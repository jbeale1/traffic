#!/usr/bin/env python3
"""
infer_onnx_wheels.py - Run the exported YOLOv8n wheel/vehicle detector
(best.onnx) on CPU via onnxruntime, with manual preprocessing
(letterbox resize) and postprocessing (box decode + NMS + coordinate
un-letterbox), since raw ONNX Runtime has none of the convenience layer
that ultralytics.YOLO() normally provides.

YOLOv8 ONNX output shape: [1, 4 + num_classes, num_anchors]
  - 4 = box (cx, cy, w, h) in the model's input pixel space (e.g. 0-960)
  - num_classes = 3 here (vehicle, front_wheel, rear_wheel)
  - num_anchors = depends on imgsz (e.g. 18900 at 960x960)
Class scores are already post-confidence (objectness folded in) per
anchor in the standard ultralytics detection export. We take
max-over-classes for each anchor as the confidence, threshold, then NMS
per class.

Usage:
    ./infer_onnx_wheels.py best.onnx image.jpg
    ./infer_onnx_wheels.py best.onnx /path/to/dir_of_images --conf 0.5
"""

import sys
import os
import time
import glob
import argparse
import contextlib
import numpy as np
import cv2


@contextlib.contextmanager
def _suppress_stderr():
    """
    onnxruntime's GPU/device-discovery probe (e.g. "Failed to detect
    devices under /sys/class/drm/cardN") is emitted by the native C++
    library during onnxruntime's own internal environment setup, NOT
    through Python's logging machinery -- so ort.set_default_logger_severity()
    has no effect on it; it fires before/outside that mechanism entirely.
    The only reliable way to silence it is to redirect the OS-level
    stderr file descriptor around the import, since this is harmless
    noise on CPU-only systems (we always force CPUExecutionProvider
    explicitly regardless of what GPU devices were or weren't found).
    """
    stderr_fd = sys.stderr.fileno()
    saved_fd = os.dup(stderr_fd)
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull_fd, stderr_fd)
        yield
    finally:
        os.dup2(saved_fd, stderr_fd)
        os.close(devnull_fd)
        os.close(saved_fd)


with _suppress_stderr():
    import onnxruntime as ort

# Also set the logger severity for any subsequent (post-import) logging,
# e.g. messages tied to session creation rather than the initial native
# environment/device probe. Severity levels: 0=VERBOSE, 1=INFO,
# 2=WARNING, 3=ERROR, 4=FATAL.
ort.set_default_logger_severity(3)

CLASS_NAMES = ['vehicle', 'front_wheel', 'rear_wheel']
DEFAULT_IMGSZ = 960
DEFAULT_CONF = 0.5
DEFAULT_IOU = 0.45


def letterbox(img, new_shape=960, color=(114, 114, 114)):
    """
    Resize + pad image to new_shape x new_shape while preserving aspect
    ratio, matching ultralytics' own letterbox preprocessing (centered
    padding, same fill color).

    Returns (padded_img, scale, (pad_x, pad_y)) so the transform can be
    inverted on the output boxes.
    """
    h0, w0 = img.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    scale = min(new_shape[0] / h0, new_shape[1] / w0)
    new_unpad = (int(round(w0 * scale)), int(round(h0 * scale)))

    dw = new_shape[1] - new_unpad[0]
    dh = new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2

    if (w0, h0) != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right,
                              cv2.BORDER_CONSTANT, value=color)
    return img, scale, (left, top)


def preprocess(img, imgsz):
    padded, scale, pad = letterbox(img, imgsz)
    blob = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    blob = blob.transpose(2, 0, 1)[None, ...]  # HWC -> NCHW
    return np.ascontiguousarray(blob), scale, pad


def xywh_to_xyxy(boxes):
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    return np.stack([x1, y1, x2, y2], axis=1)


def nms(boxes_xyxy, scores, iou_thresh):
    """Simple greedy NMS, returns indices to keep."""
    if len(boxes_xyxy) == 0:
        return []
    x1, y1, x2, y2 = boxes_xyxy[:, 0], boxes_xyxy[:, 1], boxes_xyxy[:, 2], boxes_xyxy[:, 3]
    areas = (x2 - x1).clip(0) * (y2 - y1).clip(0)
    order = scores.argsort()[::-1]

    keep = []
    while len(order) > 0:
        i = order[0]
        keep.append(i)
        if len(order) == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = (xx2 - xx1).clip(0)
        h = (yy2 - yy1).clip(0)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        remaining = np.where(iou <= iou_thresh)[0]
        order = order[remaining + 1]
    return keep


def postprocess(output, scale, pad, orig_shape, conf_thresh, iou_thresh):
    """
    output: raw model output, shape [1, 4+num_classes, num_anchors]
    Returns a list of dicts: {cls, conf, xyxy} in ORIGINAL image coords.
    """
    pred = output[0]                      # [4+num_classes, num_anchors]
    pred = pred.transpose(1, 0)           # [num_anchors, 4+num_classes]

    boxes_xywh = pred[:, :4]
    class_scores = pred[:, 4:]
    class_ids = class_scores.argmax(axis=1)
    confs = class_scores.max(axis=1)

    mask = confs >= conf_thresh
    if not mask.any():
        return []

    boxes_xywh = boxes_xywh[mask]
    class_ids = class_ids[mask]
    confs = confs[mask]
    boxes_xyxy = xywh_to_xyxy(boxes_xywh)

    results = []
    for c in np.unique(class_ids):
        c_mask = class_ids == c
        c_boxes = boxes_xyxy[c_mask]
        c_confs = confs[c_mask]
        keep = nms(c_boxes, c_confs, iou_thresh)
        for k in keep:
            results.append({
                'cls': int(c),
                'conf': float(c_confs[k]),
                'xyxy_model': c_boxes[k].copy(),
            })

    # Undo letterbox: model-space -> original-image-space
    pad_x, pad_y = pad
    h0, w0 = orig_shape[:2]
    for r in results:
        x1, y1, x2, y2 = r['xyxy_model']
        x1 = (x1 - pad_x) / scale
        y1 = (y1 - pad_y) / scale
        x2 = (x2 - pad_x) / scale
        y2 = (y2 - pad_y) / scale
        x1 = float(np.clip(x1, 0, w0))
        y1 = float(np.clip(y1, 0, h0))
        x2 = float(np.clip(x2, 0, w0))
        y2 = float(np.clip(y2, 0, h0))
        r['xyxy'] = (x1, y1, x2, y2)
        del r['xyxy_model']

    return results


class OnnxWheelDetector:
    def __init__(self, onnx_path, imgsz=DEFAULT_IMGSZ, intra_op_threads=2, inter_op_threads=1):
        self.imgsz = imgsz
        sess_options = ort.SessionOptions()
        sess_options.log_severity_level = 3  # ERROR only, same reasoning as above
        # Cap onnxruntime's CPU thread pool. Left unset, it defaults to one
        # thread per core, which can starve other processes (camera capture,
        # Flask, LiDAR UDP listener) sharing the same cores during inference.
        sess_options.intra_op_num_threads = intra_op_threads
        sess_options.inter_op_num_threads = inter_op_threads
        # Force CPU execution provider explicitly. Also wrapped in stderr
        # suppression since the native GPU/device-discovery probe may
        # fire at session/provider setup rather than (or in addition to)
        # bare import time.
        with _suppress_stderr():
            self.session = ort.InferenceSession(
                onnx_path, sess_options=sess_options,
                providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name

    def predict(self, img, conf_thresh=DEFAULT_CONF, iou_thresh=DEFAULT_IOU):
        blob, scale, pad = preprocess(img, self.imgsz)
        output = self.session.run(None, {self.input_name: blob})[0]
        return postprocess(output, scale, pad, img.shape, conf_thresh, iou_thresh)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('onnx_path')
    parser.add_argument('image_path', help='Single image file or a directory of .jpg files')
    parser.add_argument('--imgsz', type=int, default=DEFAULT_IMGSZ)
    parser.add_argument('--conf', type=float, default=DEFAULT_CONF)
    parser.add_argument('--iou', type=float, default=DEFAULT_IOU)
    args = parser.parse_args()

    print(f"Loading ONNX model: {args.onnx_path}")
    detector = OnnxWheelDetector(args.onnx_path, imgsz=args.imgsz)

    if os.path.isdir(args.image_path):
        paths = sorted(glob.glob(os.path.join(args.image_path, '*.jpg')))
    else:
        paths = [args.image_path]

    print(f"Running inference on {len(paths)} image(s)\n")

    times = []
    for path in paths:
        img = cv2.imread(path)
        if img is None:
            print(f"  Could not load: {path}")
            continue

        t0 = time.time()
        results = detector.predict(img, conf_thresh=args.conf, iou_thresh=args.iou)
        elapsed = time.time() - t0
        times.append(elapsed)

        print(f"{path}  ({elapsed*1000:.1f} ms)")
        for r in results:
            x1, y1, x2, y2 = r['xyxy']
            print(f"  {CLASS_NAMES[r['cls']]:12s} conf={r['conf']:.3f}  "
                  f"box=({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f})")

    if times:
        times = np.array(times)
        print(f"\nTiming over {len(times)} image(s):")
        print(f"  mean={times.mean()*1000:.1f} ms  "
              f"median={np.median(times)*1000:.1f} ms  "
              f"min={times.min()*1000:.1f} ms  max={times.max()*1000:.1f} ms")
        print(f"  throughput: {1.0/times.mean():.2f} img/s")


if __name__ == '__main__':
    main()
