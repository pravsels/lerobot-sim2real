#!/usr/bin/env python3
"""
prepare_annotated_frames.py

Usage:
    ./prepare_annotated_frames.py /path/to/project_root

Expects under project_root/:
    data/recording_data.json
    frames/            (raw .jpg)
    frames_bbox/       (annotated .jpg + per-frame .json)

Creates under project_root/:
    output/            (copied images + metadata.json)
"""

import os
import json
import shutil
import argparse
import sys

def main():
    p = argparse.ArgumentParser(
        description="Build an `output/` folder of annotated frames + metadata."
    )
    p.add_argument(
        "root_dir",
        help="Project root containing data/, frames/, frames_bbox/"
    )
    args = p.parse_args()

    root       = args.root_dir
    data_dir   = os.path.join(root, "data")
    frames_dir = os.path.join(root, "frames")
    bbox_dir   = os.path.join(root, "frames_bbox")
    output_dir = os.path.join(root, "annotated_frames")

    # sanity checks
    for d in (data_dir, frames_dir, bbox_dir):
        if not os.path.isdir(d):
            print(f"❌ Required directory not found: {d}", file=sys.stderr)
            sys.exit(1)
    os.makedirs(output_dir, exist_ok=True)

    rec_path = os.path.join(data_dir, "recording_data.json")
    if not os.path.isfile(rec_path):
        print(f"❌ recording_data.json not found in {data_dir}", file=sys.stderr)
        sys.exit(1)

    # Load recording data
    with open(rec_path, "r") as f:
        recording = json.load(f)

    frames_meta = []
    for sample in recording.get("data", []):
        fname = sample.get("frame_filename")
        if not fname:
            continue

        # pick annotated image if present
        bbox_img = os.path.join(bbox_dir, fname)
        raw_img  = os.path.join(frames_dir, fname)

        if   os.path.exists(bbox_img):
            src_img = bbox_img
        elif os.path.exists(raw_img):
            src_img = raw_img
        else:
            print(f"⚠️  {fname} missing in both frames/ and frames_bbox/, skipping.")
            continue

        dst_img = os.path.join(output_dir, fname)
        shutil.copy2(src_img, dst_img)

        # load object annotations if JSON exists
        base, _ = os.path.splitext(fname)
        ann_json = os.path.join(bbox_dir, base + ".json")
        if os.path.exists(ann_json):
            with open(ann_json, "r") as jf:
                objects = json.load(jf)
        else:
            objects = []

        # collect metadata
        frames_meta.append({
            "frame_id":                 sample.get("sample_id"),
            "joint_angles":             sample.get("joint_angles"),
            "end_effector_position":    sample.get("end_effector_position"),
            "end_effector_orientation": sample.get("end_effector_orientation"),
            "frame_filename":           fname,
            "objects":                  objects
        })

    # write single metadata.json with only frames
    meta_path = os.path.join(output_dir, "metadata.json")
    with open(meta_path, "w") as mf:
        json.dump({"frames": frames_meta}, mf, indent=2)

    print(f"✅ Prepared {len(frames_meta)} frames in {output_dir}")
    print(f"ℹ️  Metadata → {meta_path}")

if __name__ == "__main__":
    main()
