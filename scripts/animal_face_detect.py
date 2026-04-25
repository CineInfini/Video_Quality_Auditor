#!/usr/bin/env python
"""Standalone CLI for animal detection."""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import cv2
from cineinfini.core.animal_face import YoloAnimalDetector

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True, type=Path)
    p.add_argument("--video", required=True, type=Path)
    p.add_argument("--sample-times", nargs="+", type=float, default=[10,30,60])
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    detector = YoloAnimalDetector(args.weights, conf_threshold=args.conf)
    cap = cv2.VideoCapture(str(args.video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    duration = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))/fps
    results = []
    for t in args.sample_times:
        if t>duration: continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t*fps))
        ret, frame = cap.read()
        if not ret: continue
        start = time.time()
        detections = detector.detect(frame)
        elapsed = (time.time()-start)*1000
        results.append({"time_s":t,"n":len(detections),"elapsed_ms":elapsed,"detections":[{"class":d.class_name,"conf":d.confidence} for d in detections]})
        print(f"t={t:.1f}s: {len(detections)} detections ({elapsed:.0f}ms)")
    cap.release()
    if args.output:
        args.output.write_text(json.dumps(results, indent=2))
    return 0
if __name__=="__main__": sys.exit(main())
