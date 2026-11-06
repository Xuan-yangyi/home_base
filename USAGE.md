# 视频流分析程序使用说明

## 两个程序

| 程序 | 用途 | 推荐场景 |
|------|------|----------|
| `extract_and_analyze.py` | 两阶段：先抽帧，再批量分析 | **推荐** — MP4文件首选 |
| `video_pipeline.py` | 实时流式：读一帧分析一帧 | RTSP流或简单MP4 |

---

## 一、extract_and_analyze.py（推荐）

### 环境准备（EVB上）

```bash
# sophon-opencv python 模块
export PYTHONPATH=/opt/sophon/sophon-opencv_2.3.0/opencv-python:$PYTHONPATH

# sophon-ffmpeg（命令行工具，用于抽帧）
export PATH=/opt/sophon/sophon-ffmpeg-latest/bin:$PATH

# qwen3_5 推理模块（需在 /data/python/ 目录下运行）
cd /data/python
```

### 完整两阶段运行

```bash
cd /data/python
python3 /path/to/extract_and_analyze.py \
    --source huangfeihong.mp4 \
    --image-dir frames/ \
    --extract-ratio 5 \
    --resize 640 \
    --analyze-step 30 \
    --model /data/models/CV84X6/qwen3.5-35b-int4-autoround_w4bf16_seq2048_cv84x6_4core_history_dynamic_20260918_122058.bmodel \
    --config config/ \
    --output results.jsonl
```

### 只做抽帧（跳过分析）

```bash
python3 extract_and_analyze.py \
    --source input.mp4 \
    --image-dir frames/ \
    --extract-only
```

### 只做分析（已有抽帧结果）

```bash
python3 extract_and_analyze.py \
    --image-dir frames/ \
    --analyze-only \
    --analyze-step 30 \
    --model /path/to/model.bmodel \
    --config config/ \
    --output results.jsonl
```

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--source` | 无 | 输入视频路径 |
| `--image-dir` | `extracted_frames/` | 抽帧输出/读取目录 |
| `--extract-ratio` | 5 | 每N帧抽1帧 |
| `--resize` | 640 | 缩放至 N×N |
| `--analyze-step` | 30 | 抽帧后再采样：每N张分析1张 |
| `--model` | 无 | bmodel路径 |
| `--config` | `config/` | processor配置目录 |
| `--prompt` | `"请描述画面中的内容"` | 分析提示词 |
| `--output` | `analysis_results.jsonl` | JSONL输出路径 |
| `--extract-only` | — | 只抽帧不分析 |
| `--analyze-only` | — | 只分析不抽帧 |

### 视频抽帧比例计算示例

| 视频 | fps | extract-ratio | 抽出帧数 | analyze-step | 分析帧数 |
|------|-----|---------------|----------|-------------|----------|
| huangfeihong.mp4 525s | 30 | 5 | 3150 | 30 | 105 |
| test2.mp4 短 | 30 | 5 | ~300 | 30 | ~10 |
| 任意视频 | — | 5 | total/5 | 30 | total/150 |

### 输出格式

每行一个JSON，包含：帧编号、文件名、时间戳、分析文本、性能指标

```json
{"id": 5, "file": "frame_000120.png", "ts": 1793958198.99, "time": "2026-11-06 17:43:18", "text": "这张图片展示了一个电影场景...", "ftl": 3.409, "tps": 19.277, "vision_time": 1.038}
```

---

## 二、video_pipeline.py

### 运行

```bash
cd /data/python
python3 video_pipeline.py \
    -s test2.mp4 \
    -m /path/to/model.bmodel \
    -c config/ \
    -i 10 \
    -o results.jsonl
```

### 参数

| 参数 | 说明 |
|------|------|
| `-s` | 视频文件路径或RTSP地址 |
| `-m` | bmodel路径 |
| `-c` | processor配置目录 |
| `-i` | 分析间隔（秒），默认10 |
| `-o` | JSONL输出路径 |

### 局限性

- 使用 **sophon-opencv** (cv2.VideoCapture + VPSS硬件解码)
- 部分H.264编码的视频会**解码失败**（VPSS不兼容），如huangfeihong.mp4
- 适合RTSP实时流和编码兼容的MP4文件

---

## 常见问题

**Q: 为什么分析结果很多"请描述画面中的内容"？**
A: stdout解析方式限制。模型在暗景/高速运动/抽象场景下没有输出"Answer:"标记，fallback到原文。这是已知edge case，不影响图像质量。

**Q: 为什么不用video_pipeline.py？**
A: 它的cv2.VideoCapture强制VPSS硬件解码，huangfeihong.mp4的H.264 profile不兼容。extract_and_analyze.py用ffmpeg抽帧避免了这个问题。

**Q: 如何清理抽帧缓存？**
A: `rm -rf frames/` 即可。3150帧约1.9GB。