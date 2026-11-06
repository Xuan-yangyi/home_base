# Video Stream Analysis Pipeline — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `video_pipeline.py` — a modular pipeline that feeds RTSP/MP4 video frames to Qwen3.5 35B for periodic image analysis, outputting text descriptions with timestamps.

**Architecture:** Single-file pipeline with three abstract layers (FrameSource, InferenceEngine, Reporter) and a StreamPipeline orchestrator. CV2 for hardware-accelerated frame capture. Existing qwen3_5.py imported unchanged for inference.

**Tech Stack:** Python 3.10, sophon-opencv 4.8.0, Qwen3.5 35B INT4 bmodel, stdlib (abc, dataclasses, argparse, json, io, signal, tempfile)

**Spec:** `docs/2026-11-06-video-stream-analysis-design.md`

## Global Constraints

- Python >= 3.10 (transformers 5.7.0 requirement)
- sophon-opencv at `/opt/sophon/sophon-opencv_2.3.0/` with `LD_LIBRARY_PATH` and `PYTHONPATH` set
- Do NOT modify existing `/data/python/qwen3_5.py`
- Single file output: `/data/python/video_pipeline.py`
- Independent frame analysis (clear_history between frames)
- Default sampling interval: 4 seconds

---

### Task 1: Abstract Base Classes and Data Types

**Files:**
- Create: `/data/python/video_pipeline.py` (lines 1-120)

**Interfaces:**
- Produces: `Frame` dataclass, `InferenceResult` dataclass, `FrameSource` ABC, `InferenceEngine` ABC, `Reporter` ABC

- [ ] **Step 1: Write the abstract base classes and dataclass types**

```python
#!/usr/bin/env python3
"""
video_pipeline.py — Real-time video stream analysis using Qwen3.5 35B.
Feeds periodic frames from RTSP/MP4 sources to the model for description.

Usage:
  python3 video_pipeline.py --source demo.mp4 --model ../models/CV84X6/model.bmodel
  python3 video_pipeline.py --source rtsp://...   --model ... --interval 4
"""

import os
import sys
import io
import json
import time
import signal
import abc
import argparse
import tempfile
import logging
import traceback
from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Optional, List

import numpy as np

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class Frame:
    """A single video frame with metadata."""
    data: np.ndarray          # BGR image (H, W, 3), uint8
    timestamp: float          # Unix timestamp at capture
    frame_id: int             # Monotonically increasing


@dataclass
class InferenceResult:
    """Result of running inference on a frame."""
    text: str                 # Generated description
    timestamp: float          # Frame capture timestamp
    ftl: float = 0.0          # First token latency (seconds)
    tps: float = 0.0          # Tokens per second
    vision_time: float = 0.0  # Vision encoder time (seconds)


# ---------------------------------------------------------------------------
# Abstract base classes
# ---------------------------------------------------------------------------

class FrameSource(ABC):
    """Abstract source of video frames."""

    @abstractmethod
    def open(self, source: str) -> bool:
        """Open the video source. Returns True on success."""
        ...

    @abstractmethod
    def read(self) -> Optional[Frame]:
        """Read the next frame. Returns None if no frame available."""
        ...

    @abstractmethod
    def close(self) -> None:
        """Close the source and release resources."""
        ...

    @property
    @abstractmethod
    def fps(self) -> float:
        """Nominal frame rate of the source."""
        ...

    @property
    @abstractmethod
    def is_live(self) -> bool:
        """True for RTSP/live streams, False for finite files."""
        ...


class InferenceEngine(ABC):
    """Abstract video analysis engine."""

    @abstractmethod
    def load(self, model_path: str, config_path: str, dev_id: int) -> None:
        """Load and initialize the model."""
        ...

    @abstractmethod
    def analyze(self, frame: Frame, prompt: str) -> InferenceResult:
        """Run inference on a single frame."""
        ...

    def warmup(self) -> None:
        """Optional warmup inference (default: no-op)."""
        pass


class Reporter(ABC):
    """Abstract analysis result reporter."""

    @abstractmethod
    def start(self) -> None:
        """Initialize the reporter (open files, print headers, etc.)."""
        ...

    @abstractmethod
    def report(self, result: InferenceResult) -> None:
        """Output a single analysis result."""
        ...

    @abstractmethod
    def stop(self) -> None:
        """Clean up reporter resources."""
        ...
```

- [ ] **Step 2: Quick syntax check**

```bash
python3 -c "import ast; ast.parse(open('/data/python/video_pipeline.py').read()); print('OK')"
```

- [ ] **Step 3: Commit**

```bash
git add /data/python/video_pipeline.py
git commit -m "feat: add abstract base classes and data types for video pipeline"
```

---

### Task 2: CV2VideoSource — Unified FrameSource for RTSP and MP4

**Files:**
- Modify: `/data/python/video_pipeline.py` (append after Task 1 code)

**Interfaces:**
- Consumes: `Frame` dataclass, `FrameSource` ABC (from Task 1)
- Produces: `CV2VideoSource` class implementing FrameSource

- [ ] **Step 1: Write CV2VideoSource**

```python
# ---------------------------------------------------------------------------
# CV2-based FrameSource (RTSP + MP4 via hardware-accelerated OpenCV)
# ---------------------------------------------------------------------------

class CV2VideoSource(FrameSource):
    """Video frame source backed by sophon-opencv cv2.VideoCapture.

    Supports:
      - RTSP streams   (rtsp://...)
      - MP4/AVI files  (/path/to/file.mp4)
      - Other formats  (anything cv2.VideoCapture can open)
    """

    def __init__(self):
        self._cap = None
        self._source = ""
        self._fps: float = 30.0
        self._frame_count: int = 0
        self._is_live: bool = False

    def open(self, source: str) -> bool:
        self._source = source
        self._is_live = source.startswith("rtsp://") or source.startswith("rtmp://")

        # Use hardware-accelerated OpenCV from sophon SDK.
        # The LD_LIBRARY_PATH/PYTHONPATH must be set before running.
        import cv2
        if self._is_live:
            self._cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
            # Reduce buffer to minimize latency for live streams
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        else:
            self._cap = cv2.VideoCapture(source)

        if not self._cap.isOpened():
            print(f"[ERROR] Cannot open video source: {source}")
            return False

        self._fps = self._cap.get(cv2.CAP_PROP_FPS)
        if self._fps <= 0:
            self._fps = 30.0
        self._frame_count = 0
        print(f"[INFO] Video source opened: fps={self._fps:.1f}, "
              f"live={self._is_live}, source={source}")
        return True

    def read(self) -> Optional[Frame]:
        import cv2
        if self._cap is None:
            return None

        ret, bgr = self._cap.read()
        if not ret or bgr is None:
            if not self._is_live:
                return None  # MP4 file ended
            # RTSP may drop frames — return None so caller retries
            return None

        self._frame_count += 1
        return Frame(
            data=bgr,
            timestamp=time.time(),
            frame_id=self._frame_count,
        )

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
            print("[INFO] Video source closed.")

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def is_live(self) -> bool:
        return self._is_live
```

- [ ] **Step 2: Syntax check**

```bash
python3 -c "import ast; ast.parse(open('/data/python/video_pipeline.py').read()); print('OK')"
```

- [ ] **Step 3: Commit**

```bash
git add /data/python/video_pipeline.py
git commit -m "feat: add CV2VideoSource for RTSP and MP4 frame capture"
```

---

### Task 3: Qwen35Engine — InferenceEngine using existing qwen3_5.py

**Files:**
- Modify: `/data/python/video_pipeline.py` (append after Task 2 code)

**Interfaces:**
- Consumes: `Frame`, `InferenceResult` datatypes, `InferenceEngine` ABC (from Task 1)
- Produces: `Qwen35Engine` class implementing InferenceEngine

- [ ] **Step 1: Write Qwen35Engine**

```python
# ---------------------------------------------------------------------------
# Qwen3.5 Inference Engine (wraps existing qwen3_5.py unchanged)
# ---------------------------------------------------------------------------

class Qwen35Engine(InferenceEngine):
    """Inference engine wrapping the existing Qwen3_5 class from qwen3_5.py.

    The existing qwen3_5.Qwen3_5.run_round() prints to stdout; we capture
    the printed output via io.StringIO redirection and parse it back.
    """

    def __init__(self):
        self._model = None
        self._temp_dir = None
        self._frame_count = 0

    def load(self, model_path: str, config_path: str, dev_id: int) -> None:
        """Import qwen3_5 and instantiate Qwen3_5 with minimal args."""
        # Add current dir to sys.path so we can import qwen3_5
        script_dir = os.path.dirname(os.path.abspath(__file__))
        if script_dir not in sys.path:
            sys.path.insert(0, script_dir)

        import qwen3_5

        # Build a minimal argparse.Namespace matching what Qwen3_5.__init__ expects
        class _Args:
            model_path = model_path
            config_path = config_path
            devid = dev_id
            video_ratio = 0.25
            log_level = "INFO"

        self._model = qwen3_5.Qwen3_5(_Args())
        self._temp_dir = tempfile.mkdtemp(prefix="qwen35_frame_")
        print(f"[INFO] Qwen3.5 engine loaded. Temp dir: {self._temp_dir}")

    def warmup(self) -> None:
        """Run a tiny warmup inference to prime the model."""
        if self._model is None:
            return
        print("[INFO] Warming up model...")
        dummy = np.zeros((64, 64, 3), dtype=np.uint8) + 128
        warmup_path = os.path.join(self._temp_dir, "_warmup.png")
        import cv2
        cv2.imwrite(warmup_path, dummy)
        try:
            self._analyze_impl(warmup_path, "warmup")
        except Exception:
            pass  # warmup failure is non-fatal
        finally:
            if os.path.exists(warmup_path):
                os.remove(warmup_path)
        print("[INFO] Warmup complete.")

    def analyze(self, frame: Frame, prompt: str) -> InferenceResult:
        """Save frame to temp PNG, run inference, return result."""
        self._frame_count += 1
        img_path = os.path.join(
            self._temp_dir,
            f"frame_{self._frame_count:05d}.png"
        )

        import cv2
        cv2.imwrite(img_path, frame.data)

        try:
            text, ftl, tps, vision_time = self._analyze_impl(img_path, prompt)
        finally:
            try:
                os.remove(img_path)
            except OSError:
                pass

        return InferenceResult(
            text=text,
            timestamp=frame.timestamp,
            ftl=ftl,
            tps=tps,
            vision_time=vision_time,
        )

    def _analyze_impl(self, img_path: str, prompt: str):
        """Core: set up messages, call run_round, capture stdout, parse output."""
        model = self._model

        # Set the user prompt that image_message() embeds
        model.input_str = prompt

        # Build image messages
        messages = model.image_message(img_path)

        # Capture stdout during run_round()
        buf = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = buf
        try:
            model.run_round(messages, media_type="image")
        finally:
            sys.stdout = old_stdout

        # Clear history for independent mode (next frame is fresh)
        model.clear_history()
        model.history_max_posid = 0

        # Parse captured output
        output = buf.getvalue()
        return self._parse_output(output)

    def _parse_output(self, output: str):
        """Parse the printed output from run_round() to extract text and metrics.

        run_round() prints:
            \nAnswer:
            <token-by-token text>
            \nFTL: X.XXX s
            \nTPS: Y.YYY token/s
            [\nTotal Tokens: N]        (only for history bmodel)
            [\nVision([...]): Z.ZZZ s] (only for image/video)

        Returns: (text: str, ftl: float, tps: float, vision_time: float)
        """
        # Find the Answer: marker
        answer_marker = "\nAnswer:\n"
        idx = output.find(answer_marker)
        if idx == -1:
            return ("", 0.0, 0.0, 0.0)

        after_answer = output[idx + len(answer_marker):]

        # The last lines are metrics: FTL, TPS, optionally Total Tokens, Vision
        lines = after_answer.split("\n")

        ftl = 0.0
        tps = 0.0
        vision_time = 0.0
        text_lines = []

        for line in lines:
            if line.startswith("FTL:"):
                # "FTL: X.XXX s"
                parts = line.split()
                if len(parts) >= 2:
                    ftl = float(parts[1])
            elif line.startswith("TPS:"):
                # "TPS: Y.YYY token/s"
                parts = line.split()
                if len(parts) >= 2:
                    tps = float(parts[1])
            elif line.startswith("Vision("):
                # "Vision([[1, 18, 28]]): Z.ZZZ s"
                parts = line.split("): ")
                if len(parts) >= 2:
                    vision_time = float(parts[1].split()[0])
            elif line.startswith("Total Tokens:"):
                # skip this line
                continue
            else:
                text_lines.append(line)

        text = "\n".join(text_lines).strip()
        return (text, ftl, tps, vision_time)
```

- [ ] **Step 2: Syntax check**

```bash
cd /data/python && python3 -c "import ast; ast.parse(open('video_pipeline.py').read()); print('Syntax OK')"
```

- [ ] **Step 3: Quick import test (just verify qwen3_5 can be imported from video_pipeline)**

```bash
cd /data/python && python3 -c "
import sys; sys.path.insert(0, '.')
import qwen3_5
print('qwen3_5 import OK')
print('Qwen3_5 class:', qwen3_5.Qwen3_5)
"
```

- [ ] **Step 4: Commit**

```bash
git add /data/python/video_pipeline.py
git commit -m "feat: add Qwen35Engine wrapping existing qwen3_5.py"
```

---

### Task 4: ConsoleReporter and FileReporter

**Files:**
- Modify: `/data/python/video_pipeline.py` (append after Task 3 code)

**Interfaces:**
- Consumes: `InferenceResult` dataclass, `Reporter` ABC (from Task 1)
- Produces: `ConsoleReporter`, `FileReporter` classes

- [ ] **Step 1: Write ConsoleReporter and FileReporter**

```python
# ---------------------------------------------------------------------------
# Reporters
# ---------------------------------------------------------------------------

class ConsoleReporter(Reporter):
    """Print analysis results to stdout with color and formatting."""

    def __init__(self):
        self._start_time: float = 0.0
        self._count: int = 0

    def start(self) -> None:
        self._start_time = time.time()
        self._count = 0
        print("=" * 60)
        print("Video Stream Analysis — Live")
        print("=" * 60)

    def report(self, result: InferenceResult) -> None:
        self._count += 1
        ts = time.strftime("%H:%M:%S", time.localtime(result.timestamp))
        elapsed = result.timestamp - self._start_time

        print(f"\n[{ts} | #{self._count:03d} | +{elapsed:5.1f}s]")
        print(f"{result.text}")
        print(f"  ── FTL:{result.ftl:.2f}s  TPS:{result.tps:.1f}  "
              f"Vision:{result.vision_time:.2f}s")

    def stop(self) -> None:
        elapsed = time.time() - self._start_time
        print(f"\n{'=' * 60}")
        print(f"Analysis finished. {self._count} frames in {elapsed:.1f}s.")
        print(f"{'=' * 60}")


class FileReporter(Reporter):
    """Write analysis results as JSONL (one JSON object per line)."""

    def __init__(self, output_path: str):
        self._path = output_path
        self._fd = None

    def start(self) -> None:
        self._fd = open(self._path, "w", encoding="utf-8")
        self._count = 0
        print(f"[INFO] Writing results to {self._path}")

    def report(self, result: InferenceResult) -> None:
        self._count += 1
        record = {
            "ts": result.timestamp,
            "time": time.strftime("%Y-%m-%d %H:%M:%S",
                                  time.localtime(result.timestamp)),
            "frame_id": self._count,
            "text": result.text,
            "ftl": round(result.ftl, 3),
            "tps": round(result.tps, 3),
            "vision_time": round(result.vision_time, 3),
        }
        json.dump(record, self._fd, ensure_ascii=False)
        self._fd.write("\n")
        self._fd.flush()

    def stop(self) -> None:
        if self._fd:
            self._fd.close()
            self._fd = None
            print(f"[INFO] Results saved to {self._path}")
```

- [ ] **Step 2: Syntax check**

```bash
cd /data/python && python3 -c "import ast; ast.parse(open('video_pipeline.py').read()); print('Syntax OK')"
```

- [ ] **Step 3: Commit**

```bash
git add /data/python/video_pipeline.py
git commit -m "feat: add ConsoleReporter and FileReporter"
```

---

### Task 5: StreamPipeline Orchestrator and CLI Entry Point

**Files:**
- Modify: `/data/python/video_pipeline.py` (append after Task 4 code)

**Interfaces:**
- Consumes: All types from Tasks 1-4
- Produces: `StreamPipeline` class, `parse_args()`, `main()`, `if __name__ == "__main__"` block

- [ ] **Step 1: Write StreamPipeline**

```python
# ---------------------------------------------------------------------------
# Pipeline Orchestrator
# ---------------------------------------------------------------------------

class StreamPipeline:
    """Orchestrates the video analysis pipeline: source → engine → reporters."""

    def __init__(
        self,
        source: FrameSource,
        engine: InferenceEngine,
        reporters: List[Reporter],
        source_url: str,
        model_path: str,
        config_path: str,
        dev_id: int = 0,
        interval: float = 4.0,
        prompt: str = "请描述画面中的内容",
    ):
        self._source = source
        self._engine = engine
        self._reporters = reporters
        self._source_url = source_url
        self._model_path = model_path
        self._config_path = config_path
        self._dev_id = dev_id
        self._interval = interval
        self._prompt = prompt
        self._stop = False

    def run(self) -> None:
        """Main loop: sample → analyze → report → wait → repeat."""
        # Set up signal handler for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        # Initialize reporters
        for r in self._reporters:
            r.start()

        # Open video source
        if not self._source.open(self._source_url):
            print("[ERROR] Failed to open video source. Exiting.")
            return

        # Load model and warmup
        self._engine.load(self._model_path, self._config_path, self._dev_id)
        try:
            self._engine.warmup()
        except Exception as e:
            print(f"[WARN] Warmup failed (non-fatal): {e}")

        print(f"[INFO] Starting analysis loop. "
              f"Interval: {self._interval}s, Prompt: {self._prompt}")

        while not self._stop:
            frame = self._source.read()

            if frame is None:
                if not self._source.is_live:
                    break  # MP4 finished
                time.sleep(0.5)  # RTSP: wait for data
                continue

            try:
                result = self._engine.analyze(frame, self._prompt)
            except Exception as e:
                print(f"[ERROR] Inference failed: {e}")
                traceback.print_exc()
                continue

            for reporter in self._reporters:
                try:
                    reporter.report(result)
                except Exception as e:
                    print(f"[ERROR] Reporter failed: {e}")

            # Sleep, accounting for inference time
            sleep_time = max(0.1, self._interval - result.ftl)
            if self._stop:
                break
            time.sleep(sleep_time)

        # Cleanup
        self._source.close()
        for r in self._reporters:
            r.stop()

    def _signal_handler(self, signum, frame):
        print("\n[INFO] Interrupted. Shutting down gracefully...")
        self._stop = True
```

- [ ] **Step 2: Write CLI argument parser and main()**

```python
# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Real-time video stream analysis using Qwen3.5 35B"
    )
    # yapf: disable
    parser.add_argument(
        "--source", "-s", type=str, required=True,
        help="Video source: RTSP URL (rtsp://...) or local file path (mp4/avi/etc.)",
    )
    parser.add_argument(
        "--model", "-m", type=str, required=True,
        help="Path to the bmodel file",
    )
    parser.add_argument(
        "--config", "-c", type=str, default="config/",
        help="Path to processor config directory (default: config/)",
    )
    parser.add_argument(
        "--devid", "-d", type=int, default=0,
        help="TPU device ID (default: 0)",
    )
    parser.add_argument(
        "--interval", type=float, default=4.0,
        help="Sampling interval in seconds (default: 4.0)",
    )
    parser.add_argument(
        "--prompt", "-p", type=str, default="请描述画面中的内容",
        help="Question/prompt for each frame analysis",
    )
    parser.add_argument(
        "--output", "-o", type=str, default=None,
        help="JSONL output file path (skip if not specified)",
    )
    parser.add_argument(
        "--no-console", action="store_true",
        help="Disable console output (only write to --output file)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Build pipeline components
    source = CV2VideoSource()
    engine = Qwen35Engine()
    reporters: List[Reporter] = []

    if not args.no_console:
        reporters.append(ConsoleReporter())
    if args.output:
        reporters.append(FileReporter(args.output))

    if not reporters:
        print("[WARN] No output configured. "
              "Add --output or remove --no-console.")

    pipeline = StreamPipeline(
        source=source,
        engine=engine,
        reporters=reporters,
        source_url=args.source,
        model_path=args.model,
        config_path=args.config,
        dev_id=args.devid,
        interval=args.interval,
        prompt=args.prompt,
    )

    pipeline.run()


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Syntax check**

```bash
cd /data/python && python3 -c "import ast; ast.parse(open('video_pipeline.py').read()); print('Syntax OK')"
```

- [ ] **Step 4: Dry-run help test (no model load)**

```bash
cd /data/python && python3 video_pipeline.py --help
```

Expected: prints usage with all the arguments listed.

- [ ] **Step 5: Commit**

```bash
git add /data/python/video_pipeline.py
git commit -m "feat: add StreamPipeline orchestrator and CLI entry point"
```

---

### Task 6: End-to-End Test with test2.mp4

**Files:**
- Read: `/data/python/video_pipeline.py` (complete)
- Read: `/data/python/qwen3_5.py` (existing, unchanged)

**Prerequisites:**
- `test2.mp4` available on EVB at `/data/test2.mp4`
- sophon-opencv env vars set
- bmodel at `/data/models/CV84X6/qwen3.5-35b-int4-autoround_w4bf16_seq2048_cv84x6_4core_history_dynamic_20260918_122058.bmodel`

- [ ] **Step 1: Upload video_pipeline.py to EVB**

```bash
sshpass -p 'linaro' scp -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -P 83 \
    /data/python/video_pipeline.py \
    linaro@172.25.202.33:/data/python/video_pipeline.py
```

- [ ] **Step 2: Upload test2.mp4 to EVB**

```bash
sshpass -p 'linaro' scp -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -P 83 \
    test2.mp4 \
    linaro@172.25.202.33:/data/test2.mp4
```

- [ ] **Step 3: Run the pipeline with test2.mp4**

Run with lowered interval (2s) for quick test on a short video:

```bash
sshpass -p 'linaro' ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p 83 \
    linaro@172.25.202.33 \
    "export LD_LIBRARY_PATH=/opt/sophon/sophon-opencv_2.3.0/lib:\$LD_LIBRARY_PATH && \
     export PYTHONPATH=/opt/sophon/sophon-opencv_2.3.0/opencv-python:\$PYTHONPATH && \
     cd /data/python && \
     timeout 120 python3 video_pipeline.py \
         --source /data/test2.mp4 \
         --model ../models/CV84X6/qwen3.5-35b-int4-autoround_w4bf16_seq2048_cv84x6_4core_history_dynamic_20260918_122058.bmodel \
         --config config/ \
         --interval 4 \
         --prompt '请详细描述画面中发生了什么' \
         --output /data/results.jsonl"
```

Expected output: Model load message (~64s), then periodic frame analysis with timestamps.

- [ ] **Step 4: Verify output**

```bash
sshpass -p 'linaro' ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p 83 \
    linaro@172.25.202.33 "cat /data/results.jsonl 2>/dev/null | head -3"
```

Expected: Valid JSONL records with `ts`, `text`, `ftl`, `tps`, `vision_time` fields.

- [ ] **Step 5: Check for issues and fix if needed**, then commit

```bash
git add /data/python/video_pipeline.py
git commit -m "fix: any issues found during end-to-end test"
```