#!/usr/bin/env python3
"""
bbox_viewer.py — Step through JPEGs in a directory and overlay BBox from EXIF.

Usage:
    python3 bbox_viewer.py <directory>

Keys:
    n / d / right  — next image
    p / a / left   — previous image
    q / Escape     — quit
"""

import sys
import re
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


BBOX_COLOR     = (0, 255, 0)   # green, BGR
BBOX_THICKNESS = 2
LABEL_COLOR    = (0, 255, 0)
LABEL_SCALE    = 0.55
LABEL_THICKNESS = 1


def parse_bbox(image_description: str):
    """Extract BBox=x1,y1,x2,y2 from ImageDescription string. Returns tuple or None."""
    m = re.search(r'BBox=(\d+),(\d+),(\d+),(\d+)', image_description)
    if m:
        return tuple(int(v) for v in m.groups())
    return None


def read_exif_description(path: Path) -> str | None:
    """Read ImageDescription tag (0x010E) from JPEG EXIF via Pillow."""
    try:
        img = Image.open(path)
        exif = img.getexif()
        return exif.get(0x010E)
    except Exception:
        return None


def load_image_cv(path: Path):
    """Load JPEG as BGR numpy array via OpenCV."""
    img = cv2.imread(str(path))
    return img


def draw_bbox(img, bbox, label: str):
    """Draw bounding box and label on a copy of img. Returns new array."""
    out = img.copy()
    x1, y1, x2, y2 = bbox
    cv2.rectangle(out, (x1, y1), (x2, y2), BBOX_COLOR, BBOX_THICKNESS)
    # Label just above the box; shift down if too close to top
    text_y = y1 - 8 if y1 > 20 else y1 + 18
    cv2.putText(out, label, (x1, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, LABEL_SCALE,
                LABEL_COLOR, LABEL_THICKNESS, cv2.LINE_AA)
    return out


def show_image(window_name: str, img_path: Path, index: int, total: int):
    """Load, annotate, and display one image. Returns immediately."""
    desc = read_exif_description(img_path)
    bbox = parse_bbox(desc) if desc else None

    img = load_image_cv(img_path)
    if img is None:
        print(f"  [warning] could not load {img_path.name}")
        return

    if bbox:
        label = f"BBox {bbox[0]},{bbox[1]}  {bbox[2]},{bbox[3]}"
        img = draw_bbox(img, bbox, label)
        status = f"BBox={bbox}"
    else:
        # status = "no BBox" + (f" (desc: {desc!r})" if desc else " (no EXIF desc)")
        status = "no BBox" + (f" (desc: {desc})" if desc else " (no EXIF desc)")

    title = f"[{index+1}/{total}]  {img_path.name}    {status}"
    cv2.setWindowTitle(window_name, title)
    cv2.imshow(window_name, img)


def main():
    #if len(sys.argv) < 2:
    #    print("Usage: bbox_viewer.py <directory>")
    #    sys.exit(1)
    # directory = Path(sys.argv[1])

    directory = Path(r"Z:\CAMA")

    if not directory.is_dir():
        print(f"Not a directory: {directory}")
        sys.exit(1)

    jpegs = sorted(directory.glob("*.jpg")) + sorted(directory.glob("*.jpeg"))
    jpegs = sorted(set(jpegs))   # deduplicate, keep sorted
    if not jpegs:
        print(f"No JPEG files found in {directory}")
        sys.exit(1)

    print(f"Found {len(jpegs)} JPEG(s) in {directory}")
    print("Keys: n/d=next  p/a=prev  q/Esc=quit")

    window = "bbox_viewer"
    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)

    index = 0
    show_image(window, jpegs[index], index, len(jpegs))

    while True:
        key = cv2.waitKey(0) & 0xFF
        # Arrow keys on Linux come as multi-byte; OpenCV maps them to these values:
        # right=83, left=81, up=82, down=84  (may vary by platform)
        if key in (ord('q'), 27):           # q or Escape
            break
        elif key in (ord('n'), ord('d'), 83, 3):   # next; 83=right arrow, 3=ctrl-C fallback
            index = index + 1 if index < len(jpegs) - 1 else 1  
        elif key in (ord('p'), ord('a'), 81):       # prev; 81=left arrow
            index = index - 1 if index > 0 else len(jpegs) - 1
        else:
            continue
        show_image(window, jpegs[index], index, len(jpegs))

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
