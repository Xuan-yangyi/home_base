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