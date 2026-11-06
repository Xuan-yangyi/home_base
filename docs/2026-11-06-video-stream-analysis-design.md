# Qwen3.5 35B 实时视频流分析系统 — 设计文档

> 创建日期: 2026-11-06
> 目标平台: CV84X6 (SE13-64), SDK v2.3, libsophon 0.4.15
> 模型: Qwen3.5-35B-A3B INT4 4core

---

## 1. 需求概述

基于已部署的 Qwen3.5 35B MoE 模型，构建模块化实时视频流分析 pipeline：
- **输入**: RTSP 摄像头流 或 MP4 视频文件
- **输出**: 周期性文字描述，同步输出到终端和日志文件
- **分析模式**: 逐帧快照，每帧独立分析，不保留历史上下文
- **采样间隔**: ~4 秒/次

## 2. 架构

```
┌─────────────────────────────────────────────────────────────────┐
│                     StreamPipeline (orchestrator)                │
│                                                                  │
│  ┌──────────────────┐   ┌──────────────────┐   ┌──────────────┐ │
│  │  FrameSource      │   │  InferenceEngine  │   │  Reporter     │ │
│  │  (abstract)       │   │  (abstract)       │   │  (abstract)   │ │
│  │                   │   │                   │   │               │ │
│  │  open(source)     │   │  load_model()     │   │  start()      │ │
│  │  read() → Frame   │   │  analyze() → str  │   │  report()     │ │
│  │  close()          │   │  warmup()         │   │  stop()       │ │
│  └───────┬──────────┘   └────────┬──────────┘   └──────┬───────┘ │
│          │                       │                      │         │
│     ┌────┴────┐            ┌─────┴──────┐        ┌─────┴──────┐  │
│     │ RTSP    │            │ Qwen35     │        │ Console    │  │
│     │ Source  │            │ Engine     │        │ Reporter   │  │
│     │ (cv2)   │            │ (qwen3_5)  │        │            │  │
│     │         │            │            │        │ File       │  │
│     │ MP4     │            │ (可扩展     │        │ Reporter   │  │
│     │ Source  │            │  其他模型)  │        │ (JSONL)    │  │
│     │ (cv2)   │            │            │        │            │  │
│     └─────────┘            └────────────┘        └────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

三层抽象，每层可独立替换、独立测试。

## 3. 数据流

```
RTSP/MP4 ──→ cv2.VideoCapture ──→ numpy.ndarray (BGR)
                                         │
                                    FrameSource.read()
                                         │
                                         ▼
                              ┌── Pipeline 检查 ──┐
                              │  frame is not None  │
                              └────────┬────────────┘
                                       │
                                       ▼
                              Qwen35Engine.analyze()
                                │         │
                         save to tmp  vision + LLM
                                │         │
                        图片路径   model.generate()
                                │         │
                                │    ┌────▼──────┐
                                │    │ 文本描述    │
                                │    └────┬──────┘
                                │         │
                                ▼         ▼
                         InferenceResult
                         ├── text: str
                         ├── timestamp: float
                         ├── ftl: float
                         └── tps: float
                                │
                    ┌───────────┼───────────┐
                    ▼                       ▼
              ConsoleReporter          FileReporter
               [HH:MM:SS] text         results.jsonl
```

线程模型：单线程循环（采样间隔 ≈ 推理耗时，多线程收益不大）。

## 4. 模块接口定义

### 4.1 Frame — 通用帧包装

```python
@dataclass
class Frame:
    data: np.ndarray        # BGR numpy array
    timestamp: float        # Unix timestamp
    frame_id: int           # 递增序号
```

### 4.2 FrameSource (抽象)

```python
class FrameSource(ABC):
    @abstractmethod
    def open(self, source: str) -> bool: ...
    @abstractmethod
    def read(self) -> Frame | None: ...
    @abstractmethod
    def close(self) -> None: ...
    @property
    @abstractmethod
    def fps(self) -> float: ...
    @property
    @abstractmethod
    def is_live(self) -> bool: ...
```

| 实现 | 解码后端 | 说明 |
|------|---------|------|
| `RTSPSource` | sophon-opencv `cv2.VideoCapture` | RTSP 流，设置 `CAP_PROP_BUFFERSIZE=1` |
| `MP4Source` | sophon-opencv `cv2.VideoCapture` | 本地文件，播完自动停止 |

sophon-opencv 路径: `/opt/sophon/sophon-opencv_2.3.0/`
Python 绑定: `/opt/sophon/sophon-opencv_2.3.0/opencv-python/cv2.so`

### 4.3 InferenceEngine (抽象)

```python
@dataclass
class InferenceResult:
    text: str
    timestamp: float
    ftl: float
    tps: float
    vision_time: float

class InferenceEngine(ABC):
    @abstractmethod
    def load(self, model_path: str, config_path: str, dev_id: int) -> None: ...
    @abstractmethod
    def analyze(self, frame: Frame, prompt: str) -> InferenceResult: ...
    def warmup(self) -> None: ...  # 可选预热
```

`Qwen35Engine` 实现:
1. `frame.data` 保存为临时 PNG 文件
2. 调用 `Qwen3_5.image_message(tmp_path)` 构建消息
3. 执行 `Qwen3_5.process()` + `Qwen3_5.run_round()` 获取生成文本
4. 清理临时文件，返回 `InferenceResult`
5. 每轮推理后调用 `clear_history()` 保持独立模式

### 4.4 Reporter (抽象)

```python
class Reporter(ABC):
    @abstractmethod
    def start(self) -> None: ...
    @abstractmethod
    def report(self, result: InferenceResult) -> None: ...
    @abstractmethod
    def stop(self) -> None: ...
```

实现:

| 实现 | 输出格式 |
|------|---------|
| `ConsoleReporter` | `[HH:MM:SS] 描述文字  │ FTL:1.6s TPS:22.3` |
| `FileReporter` | JSONL 每行 `{"ts":..., "text":..., "ftl":..., "tps":...}` |

### 4.5 StreamPipeline (编排器)

```python
class StreamPipeline:
    def __init__(self,
                 source: FrameSource,
                 engine: InferenceEngine,
                 reporters: list[Reporter],
                 interval: float = 4.0,
                 prompt: str = "请描述画面中的内容"):
        ...

    def run(self) -> None:
        sources = [r.start() for r in self.reporters]
        self.source.open()
        self.engine.load()
        self.engine.warmup()

        while not self._stop:
            frame = self.source.read()
            if frame is None:
                if not self.source.is_live:
                    break  # MP4 播放完毕
                time.sleep(0.5)  # RTSP 等待数据
                continue

            result = self.engine.analyze(frame, self.prompt)
            for reporter in self.reporters:
                reporter.report(result)

            time.sleep(max(0, self.interval - result.ftl))

        self.source.close()
        for r in self.reporters:
            r.stop()
```

## 5. 命令行接口

```bash
# MP4 文件分析
python3 video_pipeline.py \
    --source ./demo.mp4 \
    --model ../models/CV84X6/qwen3.5-35b-...history_dynamic_xxx.bmodel \
    --config config/ \
    --interval 4 \
    --output results.jsonl

# RTSP 摄像头分析
python3 video_pipeline.py \
    --source rtsp://admin:password@192.168.1.100:554/stream1 \
    --model ../models/CV84X6/qwen3.5-35b-...history_dynamic_xxx.bmodel \
    --config config/ \
    --interval 4 \
    --prompt "请分析画面中的安全状况"

# 仅终端输出 (不写日志)
python3 video_pipeline.py --source demo.mp4 --model ... --config config/
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--source` | 必填 | RTSP URL 或 MP4/视频文件路径 |
| `--model`, `-m` | 必填 | bmodel 路径 |
| `--config`, `-c` | `config/` | processor 配置目录 |
| `--devid`, `-d` | 0 | TPU 设备 ID |
| `--interval` | 4 | 采样间隔 (秒) |
| `--prompt`, `-p` | "请描述画面中的内容" | 每帧的提问文字 |
| `--max_tokens` | 100 | 每次生成最大 token 数 |
| `--output`, `-o` | 无 | JSONL 日志路径 (不指定则不写文件) |
| `--no-console` | False | 关闭终端输出 |

## 6. 环境配置

运行前需初始化 sophon 库路径：

```bash
# OpenCV 硬件加速
export LD_LIBRARY_PATH=/opt/sophon/sophon-opencv_2.3.0/lib:$LD_LIBRARY_PATH
export PYTHONPATH=/opt/sophon/sophon-opencv_2.3.0/opencv-python:$PYTHONPATH

# FFmpeg 硬件加速 (可选，用于 MP4 解码加速)
export LD_LIBRARY_PATH=/opt/sophon/sophon-ffmpeg_2.3.0/lib:$LD_LIBRARY_PATH
export PATH=/opt/sophon/sophon-ffmpeg_2.3.0/bin:$PATH
```

## 7. 文件清单

| 文件 | 位置 | 说明 | 新增/修改 |
|------|------|------|----------|
| `video_pipeline.py` | `/data/python/` | 主程序 + Pipeline 编排 | 新增 |
| `qwen3_5.py` | `/data/python/` | 复用 Qwen3_5 类作为推理引擎 | 不改动 |

## 8. 性能预估

基于 CV84X6 实测数据 (35B history 模型):

| 阶段 | 耗时 |
|------|------|
| 模型加载 | ~64s (仅一次) |
| Vision encoder (768x768) | ~0.4s |
| FTL + 生成 100 token | ~1.6 + 100/22 ≈ 6.1s |
| 单轮总耗时 | ~6.5s |
| 建议采样间隔 | ≥ 5s |

## 9. 扩展性

通过抽象接口，未来可轻松扩展：

- **FrameSource**: USB 摄像头、HLS 流、图片序列
- **InferenceEngine**: Qwen3-VL、Qwen2-VL 等其他模型
- **Reporter**: WebSocket 推流、HTTP API、Kafka、MQTT
- **Pipeline**: 多路视频源并发、事件触发模式 (运动检测)

## 10. 未涵盖

以下功能不在当前版本范围内：
- 多帧累积分析（滑动窗口模式）
- 实时视频推流到前端
- Web UI 集成
- 视频编码/保存