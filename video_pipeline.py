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

        import cv2
        if self._is_live:
            self._cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
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
                return None
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
        script_dir = os.path.dirname(os.path.abspath(__file__))
        if script_dir not in sys.path:
            sys.path.insert(0, script_dir)

        import qwen3_5
        from types import SimpleNamespace

        _args = SimpleNamespace(
            model_path=model_path,
            config_path=config_path,
            devid=dev_id,
            video_ratio=0.25,
            log_level="INFO",
            do_sample=False,
        )

        self._model = qwen3_5.Qwen3_5(_args)
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
            pass
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
        model.input_str = prompt
        messages = model.image_message(img_path)

        buf = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = buf
        try:
            model.run_round(messages, media_type="image")
        finally:
            sys.stdout = old_stdout

        model.clear_history()
        model.history_max_posid = 0

        output = buf.getvalue()
        return self._parse_output(output)

    def _parse_output(self, output: str):
        """Parse the printed output from run_round() to extract text and metrics.

        run_round() prints:
            \\nAnswer:\\n
            <token-by-token text>
            \\nFTL: X.XXX s
            \\nTPS: Y.YYY token/s
            [\\nTotal Tokens: N]
            [\\nVision([...]): Z.ZZZ s]

        Returns: (text: str, ftl: float, tps: float, vision_time: float)
        """
        answer_marker = "\nAnswer:\n"
        idx = output.find(answer_marker)
        if idx == -1:
            return ("", 0.0, 0.0, 0.0)

        after_answer = output[idx + len(answer_marker):]
        lines = after_answer.split("\n")

        ftl = 0.0
        tps = 0.0
        vision_time = 0.0
        text_lines = []

        for line in lines:
            if line.startswith("FTL:"):
                parts = line.split()
                if len(parts) >= 2:
                    ftl = float(parts[1])
            elif line.startswith("TPS:"):
                parts = line.split()
                if len(parts) >= 2:
                    tps = float(parts[1])
            elif line.startswith("Vision("):
                parts = line.split("): ")
                if len(parts) >= 2:
                    vision_time = float(parts[1].split()[0])
            elif line.startswith("Total Tokens:"):
                continue
            else:
                text_lines.append(line)

        text = "\n".join(text_lines).strip()
        return (text, ftl, tps, vision_time)


# ---------------------------------------------------------------------------
# Reporters — Console and JSONL file output
# ---------------------------------------------------------------------------

class ConsoleReporter(Reporter):
    """Prints results to terminal with timestamps and metrics."""

    def start(self) -> None:
        print("[INFO] ConsoleReporter started.")

    def report(self, result: InferenceResult) -> None:
        ts = time.strftime("%H:%M:%S", time.localtime(result.timestamp))
        line = (
            f"[{ts}] {result.text}"
            f"  │ FTL:{result.ftl:.1f}s TPS:{result.tps:.1f}"
        )
        if result.vision_time > 0:
            line += f" Vision:{result.vision_time:.2f}s"
        print(line)

    def stop(self) -> None:
        print("[INFO] ConsoleReporter stopped.")


class FileReporter(Reporter):
    """Writes results as JSONL to a file."""

    def __init__(self, output_path: str):
        self._path = output_path
        self._file = None
        self._count = 0

    def start(self) -> None:
        self._count = 0
        self._file = open(self._path, "w")
        print(f"[INFO] FileReporter writing to: {self._path}")

    def report(self, result: InferenceResult) -> None:
        if self._file is None:
            return
        self._count += 1
        record = {
            "id": self._count,
            "ts": result.timestamp,
            "time": time.strftime("%Y-%m-%d %H:%M:%S",
                                  time.localtime(result.timestamp)),
            "text": result.text,
            "ftl": result.ftl,
            "tps": result.tps,
            "vision_time": result.vision_time,
        }
        self._file.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._file.flush()

    def stop(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None
            print(f"[INFO] FileReporter stopped. Wrote {self._count} records.")


# ---------------------------------------------------------------------------
# StreamPipeline — orchestrator
# ---------------------------------------------------------------------------

class StreamPipeline:
    """Orchestrates FrameSource → InferenceEngine → Reporter pipeline."""

    def __init__(self,
                 source_url: str,
                 model_path: str,
                 config_path: str,
                 dev_id: int,
                 interval: float = 4.0,
                 prompt: str = "请描述画面中的内容",
                 output_path: str = "",
                 no_console: bool = False):
        self.source_url = source_url
        self.model_path = model_path
        self.config_path = config_path
        self.dev_id = dev_id
        self.interval = interval
        self.prompt = prompt
        self.output_path = output_path
        self.no_console = no_console
        self._stop = False

        self.source: FrameSource = CV2VideoSource()
        self.engine: InferenceEngine = Qwen35Engine()
        self.reporters: List[Reporter] = []

    def _build_reporters(self) -> None:
        if not self.no_console:
            self.reporters.append(ConsoleReporter())
        if self.output_path:
            self.reporters.append(FileReporter(self.output_path))

    def run(self) -> None:
        signal.signal(signal.SIGINT, lambda sig, frame: setattr(self, '_stop', True))
        signal.signal(signal.SIGTERM, lambda sig, frame: setattr(self, '_stop', True))

        self._build_reporters()

        if not self.source.open(self.source_url):
            print("[ERROR] Cannot open video source. Aborting.")
            return

        self.engine.load(self.model_path, self.config_path, self.dev_id)
        self.engine.warmup()

        for r in self.reporters:
            r.start()

        print(f"[INFO] Pipeline running. Interval={self.interval}s, "
              f"Prompt=\"{self.prompt}\"")
        print("[INFO] Press Ctrl+C to stop.\n")

        while not self._stop:
            frame = self.source.read()
            if frame is None:
                if not self.source.is_live:
                    print("[INFO] Video source exhausted. Stopping.")
                    break
                time.sleep(0.5)
                continue

            try:
                result = self.engine.analyze(frame, self.prompt)
            except Exception as exc:
                print(f"[ERROR] Inference failed: {exc}")
                traceback.print_exc()
                continue

            for reporter in self.reporters:
                reporter.report(result)

            wait_time = max(0, self.interval - result.ftl)
            if wait_time > 0:
                time.sleep(wait_time)

        self.source.close()
        for r in self.reporters:
            r.stop()
        print("[INFO] Pipeline finished.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Real-time video stream analysis using Qwen3.5 35B"
    )
    parser.add_argument("--source", "-s", required=True,
                        help="RTSP URL or MP4/video file path")
    parser.add_argument("--model", "-m", required=True,
                        help="Path to bmodel file")
    parser.add_argument("--config", "-c", default="config/",
                        help="Processor config directory (default: config/)")
    parser.add_argument("--devid", "-d", type=int, default=0,
                        help="TPU device ID (default: 0)")
    parser.add_argument("--interval", "-i", type=float, default=4.0,
                        help="Sampling interval in seconds (default: 4)")
    parser.add_argument("--prompt", "-p", default="请描述画面中的内容",
                        help="Prompt for each frame (default: 请描述画面中的内容)")
    parser.add_argument("--output", "-o", default="",
                        help="JSONL output path (skip file output if omitted)")
    parser.add_argument("--no-console", action="store_true",
                        help="Disable console output")

    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"[ERROR] Model file not found: {args.model}")
        sys.exit(1)

    pipeline = StreamPipeline(
        source_url=args.source,
        model_path=args.model,
        config_path=args.config,
        dev_id=args.devid,
        interval=args.interval,
        prompt=args.prompt,
        output_path=args.output,
        no_console=args.no_console,
    )
    pipeline.run()


if __name__ == "__main__":
    main()