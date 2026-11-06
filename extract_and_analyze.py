#!/usr/bin/env python3
"""
Two-phase video analysis:
  Phase 1: Extract frames (N:1), resize, save PNGs to directory
  Phase 2: Batch-feed saved images to Qwen3.5 35B for analysis

Usage:
  # Full pipeline
  python3 extract_and_analyze.py -s huangfeihong.mp4 -m model.bmodel -c config/

  # Extract only
  python3 extract_and_analyze.py -s huangfeihong.mp4 --extract-only

  # Analyze only (pre-extracted frames)
  python3 extract_and_analyze.py -d frames_dir/ -m model.bmodel -c config/
"""

import os
import sys
import io
import json
import time
import signal
import argparse
import glob
from types import SimpleNamespace
from typing import Optional

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Phase 1: Frame extraction
# ---------------------------------------------------------------------------

def extract_frames(video_path: str, output_dir: str,
                   extract_ratio: int = 5, resize_to: int = 640) -> int:
    """Extract every Nth frame, resize, save as PNG. Returns frame count."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {video_path}")
        return 0

    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    os.makedirs(output_dir, exist_ok=True)

    print(f"[Extract] Video: {video_path}")
    print(f"[Extract] FPS: {fps:.1f}, Total frames: {total}")
    print(f"[Extract] Ratio: 1:{extract_ratio}, Resize: {resize_to}x{resize_to}")
    print(f"[Extract] Estimated output: ~{total // extract_ratio} frames")
    print(f"[Extract] Output dir: {output_dir}")

    count = 0
    saved = 0
    t0 = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        count += 1

        if count % extract_ratio != 0:
            continue

        # Resize
        h, w = frame.shape[:2]
        if w != resize_to or h != resize_to:
            frame = cv2.resize(frame, (resize_to, resize_to),
                              interpolation=cv2.INTER_AREA)

        out_path = os.path.join(output_dir, f"frame_{saved:06d}.png")
        cv2.imwrite(out_path, frame)
        saved += 1

        if saved % 200 == 0:
            elapsed = time.time() - t0
            print(f"[Extract] {saved} frames saved ({elapsed:.1f}s)...")

    cap.release()
    elapsed = time.time() - t0
    print(f"[Extract] Done: {saved} frames saved in {elapsed:.1f}s "
          f"({saved / elapsed:.1f} fps)")
    return saved


# ---------------------------------------------------------------------------
# Phase 2: Batch inference
# ---------------------------------------------------------------------------

def load_engine(model_path: str, config_path: str, dev_id: int):
    """Import qwen3_5 and instantiate the model."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)

    import qwen3_5

    args = SimpleNamespace(
        model_path=model_path,
        config_path=config_path,
        devid=dev_id,
        video_ratio=0.25,
        log_level="INFO",
        do_sample=False,
    )
    model = qwen3_5.Qwen3_5(args)
    print(f"[Analyze] Model loaded: {model_path}")
    return model


def analyze_one(model, img_path: str, prompt: str):
    """Run inference on a single image. Returns (text, ftl, tps, vision_time)."""
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

    return _parse_output(buf.getvalue())


def _parse_output(output: str):
    """Parse printed output from run_round()."""
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


def batch_analyze(image_dir: str, model_path: str, config_path: str,
                  dev_id: int, prompt: str, output_path: str,
                  analyze_step: int = 1) -> int:
    """Walk image_dir, analyze each image (stepped), write JSONL."""
    images = sorted(glob.glob(os.path.join(image_dir, "frame_*.png")))
    images = images[::analyze_step]

    if not images:
        print(f"[Analyze] No frame_*.png found in {image_dir}")
        return 0

    print(f"[Analyze] Found {len(images)} images to process "
          f"(step={analyze_step})")
    print(f"[Analyze] Prompt: \"{prompt}\"")

    # Load model
    model = load_engine(model_path, config_path, dev_id)

    # Warmup
    print("[Analyze] Warming up...")
    dummy = np.zeros((64, 64, 3), dtype=np.uint8) + 128
    warmup_path = os.path.join(image_dir, "_warmup.png")
    cv2.imwrite(warmup_path, dummy)
    try:
        analyze_one(model, warmup_path, "warmup")
    except Exception:
        pass
    finally:
        os.remove(warmup_path)
    print("[Analyze] Warmup complete.")

    # Process
    out_file = open(output_path, "w")
    total = len(images)
    t0 = time.time()
    ftls = []
    tpss = []
    vts = []

    for i, img_path in enumerate(images, 1):
        try:
            text, ftl, tps, vision_time = analyze_one(model, img_path, prompt)
        except Exception as exc:
            print(f"[Analyze] ERROR frame {i}/{total}: {exc}")
            text, ftl, tps, vision_time = "", 0.0, 0.0, 0.0

        record = {
            "id": i,
            "file": os.path.basename(img_path),
            "ts": time.time(),
            "time": time.strftime("%Y-%m-%d %H:%M:%S",
                                  time.localtime()),
            "text": text,
            "ftl": ftl,
            "tps": tps,
            "vision_time": vision_time,
        }
        out_file.write(json.dumps(record, ensure_ascii=False) + "\n")
        out_file.flush()

        ftls.append(ftl)
        tpss.append(tps)
        vts.append(vision_time)

        elapsed = time.time() - t0
        eta = (elapsed / i) * (total - i)
        print(f"[Analyze] {i}/{total} | {text[:60]}..."
              f"  │ FTL:{ftl:.1f}s TPS:{tps:.1f} ETA:{eta:.0f}s")

    out_file.close()
    elapsed = time.time() - t0

    print(f"\n[Analyze] Done: {total} images in {elapsed:.0f}s "
          f"({elapsed / total:.1f}s/img)")
    print(f"[Analyze] Avg FTL: {sum(ftls)/len(ftls):.2f}s  "
          f"TPS: {sum(tpss)/len(tpss):.2f}  "
          f"Vision: {sum(vts)/len(vts):.3f}s")
    print(f"[Analyze] Output: {output_path}")

    return total


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Two-phase video analysis: extract frames then batch analyze"
    )
    # Source
    parser.add_argument("--source", "-s", default="",
                        help="Video file path (for extraction)")
    parser.add_argument("--image-dir", "-d", default="extracted_frames/",
                        help="Directory for extracted frames")

    # Extract params
    parser.add_argument("--extract-ratio", type=int, default=5,
                        help="Extract 1 out of N frames (default: 5)")
    parser.add_argument("--resize", type=int, default=640,
                        help="Resize frames to NxN (default: 640)")
    parser.add_argument("--extract-only", action="store_true",
                        help="Only extract frames, skip analysis")

    # Model / analysis params
    parser.add_argument("--model", "-m", default="",
                        help="Path to bmodel file")
    parser.add_argument("--config", "-c", default="config/",
                        help="Processor config directory (default: config/)")
    parser.add_argument("--devid", type=int, default=0,
                        help="TPU device ID (default: 0)")
    parser.add_argument("--prompt", "-p", default="请描述画面中的内容",
                        help="Prompt for each frame")
    parser.add_argument("--output", "-o", default="analysis_results.jsonl",
                        help="JSONL output path")
    parser.add_argument("--analyze-step", type=int, default=30,
                        help="Further sample: analyze 1 out of N extracted "
                             "frames (default: 30)")
    parser.add_argument("--analyze-only", action="store_true",
                        help="Only analyze, skip extraction")

    args = parser.parse_args()

    # Phase 1: Extract
    if not args.analyze_only:
        if not args.source:
            print("[ERROR] --source required for extraction")
            sys.exit(1)
        extract_frames(
            video_path=args.source,
            output_dir=args.image_dir,
            extract_ratio=args.extract_ratio,
            resize_to=args.resize,
        )
        if args.extract_only:
            print("[INFO] Extract only mode. Done.")
            return

    # Phase 2: Analyze
    if args.extract_only:
        return

    if not args.model:
        print("[ERROR] --model required for analysis")
        sys.exit(1)

    batch_analyze(
        image_dir=args.image_dir,
        model_path=args.model,
        config_path=args.config,
        dev_id=args.devid,
        prompt=args.prompt,
        output_path=args.output,
        analyze_step=args.analyze_step,
    )


if __name__ == "__main__":
    main()