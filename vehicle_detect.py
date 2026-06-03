#!/home/pi/cam-env/bin/python
"""
vehicle_detect.py  —  MOG2-based vehicle detection with centered-frame capture

Runs continuously, monitoring the lores stream for vehicles crossing the frame.
When a vehicle event ends, saves the two high-res frames closest to the estimated
center position to /dev/shm/burst/, then rsyncs them to a remote host.
Configured for IMX296 (global shutter) camera


Usage:
    python3 vehicle_detect.py [--tune FILE] [--shutter US] [--gain G] [--ev EV]
"""

import dataclasses
import math
import argparse
import collections
import csv
import logging
import signal
import subprocess
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path

from flask import Flask, Response

import cv2
import numpy as np
from PIL import Image
from libcamera import Transform, controls as libctrls
from picamera2 import Picamera2

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

VERSION        = "1.075" # SAVE_MASK constant, MIN_BLOB_BOTTOM_ROW filter
FRAME_RATE     = 20.0
LORES_SIZE     = (320, 240)
HIRES_SIZE     = (1456, 1088)
JPEG_QUALITY   = 95

# Set True to save a lores MOG2 mask PNG alongside every captured JPEG.
# Useful for algorithm development; disable for normal operation.
#SAVE_MASK      = True
SAVE_MASK      = False

# Crop applied to saved full-res JPEGs (in 1456x1088 frame coordinates)
HIRES_CROP_TOP    = 76
HIRES_CROP_BOTTOM = 830

# Background blur applied outside the vehicle bounding box before JPEG save.
# Reduces file size by softening high-frequency background (trees, grass, etc.).
BLUR_KERNEL_SIZE  = 7     # must be odd; 7x7 Gaussian
BLUR_MARGIN_HIRES = 20    # unblurred padding around vehicle bbox (hires pixels)
BLUR_LEAD_MARGIN_HIRES = 200  # extra unblurred margin on the leading edge (front of vehicle)
BLUR_TRAIL_MARGIN_HIRES = 100  # extra unblurred margin on the trailing edge (rear of vehicle)
# Lores-to-hires scale factors for IMX296 (1456x1088 hires, 320x240 lores)
LORES_TO_HIRES_X  = 1456 / 320
LORES_TO_HIRES_Y  = 1088 / 240

# Lores row range that corresponds to the hires JPEG crop (HIRES_CROP_TOP..HIRES_CROP_BOTTOM).
# The mask PNG uses this vertical extent so its pixels map 1-to-1 with the saved JPEG
# when both are scaled to the same display size.
LORES_CROP_TOP    = round(HIRES_CROP_TOP    / LORES_TO_HIRES_Y)   # ~16
LORES_CROP_BOTTOM = round(HIRES_CROP_BOTTOM / LORES_TO_HIRES_Y)   # ~183

SHM_BASE       = Path("/dev/shm/burst")
SHM_MIN_MB     = 100
THUMB_PATH     = Path("/dev/shm/preview_thumb.jpg")  # latest event thumbnail
LOG_DIR        = Path("/home/pi/CAMA")                # daily CSV event logs
REMOTE_HOST    = "jbeale@jbeale-mini.local"
REMOTE_DIR     = "/mnt/bluecherry/CAMA/"

EVENT_NUM_FILE = Path("/tmp/vehicle_detect_event_num.txt")  # daily event counter

# ---------------------------------------------------------------------------
# Daily event counter persistence
# ---------------------------------------------------------------------------

def _load_event_num() -> tuple[int, str]:
    """Read last event number and its date from state file.
    Returns (event_num, date_str) or (0, '') if absent/unreadable."""
    try:
        text = EVENT_NUM_FILE.read_text().strip().split()
        return int(text[1]), text[0]
    except Exception:
        return 0, ""


def _save_event_num(event_num: int):
    """Write today's date and current event number to state file."""
    try:
        EVENT_NUM_FILE.write_text(
            f"{datetime.now().strftime('%Y%m%d')} {event_num}\n")
    except Exception as e:
        log.warning("[event_num] failed to save state: %s", e)

# MOG2 detection parameters
ROI_TOP    = 60
ROI_BOTTOM = 130

MOG2_HISTORY        = 200
MOG2_VAR_THRESHOLD  = 35 # was 50  6/2/2026
MOG2_DETECT_SHADOWS = False

MORPH_KERNEL_SIZE = 5

MIN_BLOB_AREA       = 100 # catch smaller objects like bicyclists
MIN_ASPECT_RATIO    = 0.4
MIN_BLOB_WIDTH      = 20
MIN_BLOB_HEIGHT     = 20   # lowered from 25 to avoid intermittent streak resets on dark cars
MIN_HULL_FILL_RATIO = 0.44 # lowered from 0.45 to avoid boundary ties

# Minimum ROI-relative row that the blob's bounding box must reach.
# Rejects background grass / illumination blobs that never touch the road zone.
# Calibrated from observed data: grass-only false triggers bottom out ~row 35,
# real vehicles reach ~row 75+.  Midpoint rounded down for margin.
MIN_BLOB_BOTTOM_ROW = 50
MIN_CENTROID_AREA   = 400 # was 3000, to see motorcycles and bicyclists

# Minimum foreground pixel count to log when --verbose-fg is active.
VERBOSE_FG_MIN_PIXELS = 150

MIN_CONSECUTIVE_FRAMES = 4
LOCKOUT_FRAMES         = 15

# Minimum centroid travel (lores px) for short centroid histories (< 4 points)
# where directional consistency cannot be meaningfully computed.
MIN_CENTROID_TRAVEL = 80

# Minimum directional consistency for centroid histories of 4+ points.
# Defined as |net_displacement| / total_path_length: 1.0 = perfectly monotonic,
# 0.0 = pure random jitter.  A truck+trailer sawtooth still scores high because
# each phase is internally monotonic.  Shadow flicker scores near zero.
MIN_MOTION_CONSISTENCY = 0.2

# Steps larger than this (lores px) are clipped before summing path length for
# the consistency metric.  Suppresses outlier blob jumps (e.g. from AGC transients)
# without affecting the net displacement or normal per-frame steps.
MAX_STEP_FOR_CONSISTENCY = 160

# Minimum standard deviation of cx (lores px) over the last STATIONARY_TAIL_N
# confirmed frames.  A stationary foreground object shows near-zero cx variance
# across its final frames and always triggers a timeout; real vehicles do not.
# Long vehicles (max_blob_width >= HIRES_LONG_VEHICLE_WIDTH) are exempt.
# Set to 0 to disable.
MIN_CONFIRMED_CX_STD = 5.0

# Minimum backward centroid step (lores px) that signals a tow-vehicle/trailer
# split: when the dominant blob switches from the lead vehicle (exiting one side)
# to the trailer (still crossing), the centroid jumps back against the direction
# of travel.  50 px is ~15 % of frame width, large enough to avoid noise.
CENTROID_REVERSAL_THRESHOLD = 50

LR_NORMAL  = 0.002
LR_VEHICLE = 0.0

# Two-stage learning rate boost when AGC is stuck unsettled.
# Stage 1: moderate boost after LR_BOOST_AFTER frames.
# Stage 2: aggressive boost after LR_BOOST2_AFTER frames to force convergence.
LR_BOOST            = 0.05
LR_BOOST_AFTER      = 120   # frames (~6 seconds at 20fps)
LR_BOOST2           = 0.5
LR_BOOST2_AFTER     = 600   # frames (~30 seconds at 20fps)

# Correction factor must be within this fraction of 1.0 for MOG2 to resume
# learning. Widened from 0.02 to allow adaptation after sustained exposure shifts.
AGC_SETTLED_THRESHOLD = 0.15

EDGE_MARGIN  = 3
FRAME_WIDTH  = LORES_SIZE[0]

# Metadata pipeline delay (frames): picamera2 reports metadata this many
# frames after the frame it describes.  Used to align correction factors.
METADATA_DELAY_FRAMES = 2

# Maximum event duration before forced close and background re-adaptation
MAX_EVENT_FRAMES = int(FRAME_RATE * 2.5)  # 2.5 seconds — coordinated with HIRES_BUFFER_SIZE

# Rolling buffer of high-res frames kept in RAM.
# Must be large enough to reach back to the best-centered frame even after
# the vehicle has fully exited. At 20fps and up to ~3s visible, 60 frames
# (270 MB at 1456x1088 BGR) gives ample headroom within the 2 GB /dev/shm.
HIRES_BUFFER_SIZE = 60
# Maximum fi distance between the target frame and the closest available buffer
# entry.  If the gap exceeds this, the vehicle has already left the buffer and
# saving would produce frames from the wrong (stationary) phase.
MAX_FI_SAVE_GAP  = 40   # frames (~2s at 20fps)

# Offset (in frames) applied to chosen_fi when selecting from hires_buf.
# Set negative if hires frames appear to lag lores detection.
HIRES_LAG_FRAMES = 0

HIRES_LONG_VEHICLE_WIDTH = 300  # lores px — vehicle is "long" if max blob width reaches this

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("vehicle_detect")

# ---------------------------------------------------------------------------
# Timing helpers  (from lores_burst.py)
# ---------------------------------------------------------------------------

_wall_anchor:      datetime | None = None
_sensor_anchor_ns: int      | None = None


def sensor_ts_to_datetime(sensor_ts_ns):
    if (sensor_ts_ns is not None
            and _wall_anchor is not None
            and _sensor_anchor_ns is not None):
        return _wall_anchor + timedelta(
            seconds=(sensor_ts_ns - _sensor_anchor_ns) / 1e9)
    return datetime.now()


def ts_stem(dt: datetime) -> str:
    ms = dt.microsecond // 1000
    return f"{dt.strftime('%Y%m%d_%H%M%S')}_{ms:03d}"


def shm_free_mb() -> float:
    import os
    st = os.statvfs("/dev/shm")
    return st.f_bavail * st.f_frsize / (1024 * 1024)


def _wait_for_ntp(timeout_s: int = 120, poll_s: int = 5) -> bool:
    import subprocess as sp
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            r = sp.run(
                ["timedatectl", "show", "--property=NTPSynchronized", "--value"],
                capture_output=True, text=True, timeout=5,
            )
            if r.stdout.strip() == "yes":
                log.info("[timing] NTP synchronized")
                return True
        except Exception:
            pass
        log.info("[timing] waiting for NTP…")
        time.sleep(poll_s)
    log.warning("[timing] NTP timeout — using unsynchronized clock")
    return False

# ---------------------------------------------------------------------------
# Transfer thread  (from lores_burst.py)
# ---------------------------------------------------------------------------

def _transfer_and_cleanup():
    cmd_rsync = [
        "rsync", "-a", "--remove-source-files",
        str(SHM_BASE) + "/",
        f"{REMOTE_HOST}:{REMOTE_DIR}",
    ]
    cmd_cleanup = [
        "find", str(SHM_BASE),
        "-mindepth", "1", "-maxdepth", "1",
        "-type", "d", "-empty", "-delete",
    ]
    try:
        result = subprocess.run(cmd_rsync, capture_output=True, text=True)
        if result.returncode == 0:
            log.info("[transfer] rsync complete")
        else:
            log.warning("[transfer] rsync failed (rc=%d): %s",
                        result.returncode, result.stderr.strip())
            return
        subprocess.run(cmd_cleanup, capture_output=True)
        log.info("[transfer] cleanup done")
    except Exception as e:
        log.error("[transfer] exception: %s", e)


def start_transfer():
    t = threading.Thread(target=_transfer_and_cleanup, daemon=True)
    t.start()

# ---------------------------------------------------------------------------
# Save thread
# ---------------------------------------------------------------------------

def _build_exif(meta: dict):
    """Build a Pillow ExifData object from libcamera metadata.

    Tags that are SRATIONAL (signed rational) -- ShutterSpeedValue, BrightnessValue,
    CameraTemperature -- must go into the Exif sub-IFD (0x8769) via get_ifd(), not the
    top-level dict, because Pillow encodes top-level rationals as unsigned RATIONAL.
    """
    exif = Image.Exif()
    ifd  = exif.get_ifd(0x8769)   # Exif sub-IFD

    # ExposureTime: us -> seconds as RATIONAL (tag 0x829A, top-level IFD0)
    exp_us = meta.get("ExposureTime")
    if exp_us is not None:
        exif[0x829A] = (int(exp_us), 1_000_000)
        # ShutterSpeedValue: APEX Tv = log2(1/t_s), SRATIONAL x1000 (tag 0x9201, Exif IFD)
        if exp_us > 0:
            apex_tv = math.log2(1_000_000 / exp_us)
            ifd[0x9201] = (int(round(apex_tv * 1000)), 1000)

    # Lux -> APEX BrightnessValue: Bv = log2(lux/2.5), SRATIONAL x100 (tag 0x9203, Exif IFD)
    lux = meta.get("Lux")
    if lux is not None and lux > 0:
        bv = math.log2(lux / 2.5)
        ifd[0x9203] = (int(round(bv * 100)), 100)

    # AnalogueGain -> ISOSpeedRatings (tag 0x8827, Exif IFD), SHORT
    gain = meta.get("AnalogueGain")
    if gain is not None:
        ifd[0x8827] = int(round(gain * 100))

    # SensorTemperature -> CameraTemperature (tag 0x9400, Exif IFD), SRATIONAL degC x10
    temp = meta.get("SensorTemperature")
    if temp is not None:
        ifd[0x9400] = (int(round(temp * 10)), 10)

    # ImageDescription: human-readable lux + temperature for viewers that show IFD0 only
    desc_parts = []
    if lux is not None:
        desc_parts.append(f"Lux={lux:.1f}")
    if temp is not None:
        desc_parts.append(f"SensorTemp={temp:.1f}C")
    if desc_parts:
        exif[0x010E] = " ".join(desc_parts)   # ImageDescription, top-level IFD0

    return exif


def _select_frames(hires_buf, target_fi, n=2):
    """
    Pick n hires buffer entries nearest to target_fi.
    Returns the sorted list, or None if the closest entry is more than
    MAX_FI_SAVE_GAP frames away (vehicle has left the buffer window).
    """
    if not hires_buf:
        return None
    buf_list = sorted(hires_buf, key=lambda e: abs(e[3] - target_fi))
    if abs(buf_list[0][3] - target_fi) > MAX_FI_SAVE_GAP:
        return None
    return sorted(buf_list[:n], key=lambda e: e[3])


def blur_background(hires_bgr, bbox_lores, rightward=True):
    """
    Apply a Gaussian blur to everything outside the vehicle bounding box.
    bbox_lores is (bx, by, bw, bh) in lores coordinates, or None (no blur).
    rightward indicates direction of travel; the leading edge gets an extra
    BLUR_LEAD_MARGIN_HIRES pixels of unblurred space to cover the vehicle front.
    Returns a new image with background softened.
    """
    if bbox_lores is None:
        return hires_bgr

    bx, by, bw, bh = bbox_lores
    # Vertical extent only — unblurred region spans the full image width.
    # by is relative to ROI_TOP (blob detection runs on the ROI crop),
    # so restore full-lores y before scaling.
    hy1 = int((by + ROI_TOP) * LORES_TO_HIRES_Y) - BLUR_MARGIN_HIRES
    # Always extend to bottom of saved frame.
    hy2 = HIRES_CROP_BOTTOM
    # If the blob touches the top of the ROI, extend to top of frame.
    if by == 0:
        hy1 = 0

    H, W = hires_bgr.shape[:2]
    hx1 = 0
    hx2 = W
    hy1 = max(0, hy1)

    blurred = cv2.GaussianBlur(hires_bgr, (BLUR_KERNEL_SIZE, BLUR_KERNEL_SIZE), 0)
    result  = blurred.copy()
    # Restore unblurred vehicle region
    result[hy1:hy2, hx1:hx2] = hires_bgr[hy1:hy2, hx1:hx2]
    return result


CSV_HEADER = "event,time,epoch,width(px),frames,type,velocity(px/fr)\n"


def _append_csv_log(event_count, time_str, epoch, blob_width,
                    event_frames, event_type, velocity, date_str):
    """
    Append one row to /home/pi/CAMA/YYYYMMDD_log.csv.
    Creates the file (with header) if it does not yet exist.
    date_str is 'YYYYMMDD', derived from the capture timestamp.
    """
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"{date_str}_log.csv"
    write_header = not log_path.exists()
    vel_str = f"{velocity:.4f}" if velocity is not None else ""
    row = (f"{event_count},{time_str},{epoch:.3f},"
           f"{blob_width},{event_frames},{event_type},{vel_str}\n")
    try:
        with log_path.open('a') as f:
            if write_header:
                f.write(CSV_HEADER)
            f.write(row)
    except Exception as e:
        log.warning("[csv] failed to write log: %s", e)


def _build_mask_png(fg_mask_roi, lores_w=320):
    """
    Compose a mask PNG whose pixel grid corresponds to the saved hires JPEG.
    The image is LORES_CROP_TOP..LORES_CROP_BOTTOM rows tall and lores_w wide.
    The ROI band (ROI_TOP..ROI_BOTTOM) contains the MOG2 foreground mask (white=255);
    rows outside the ROI band are black.
    Returns a PNG-encoded bytes object, or None on failure.
    """
    h = LORES_CROP_BOTTOM - LORES_CROP_TOP
    canvas = np.zeros((h, lores_w), dtype=np.uint8)
    if fg_mask_roi is not None:
        # ROI rows relative to the lores crop origin
        roi_y1 = ROI_TOP    - LORES_CROP_TOP
        roi_y2 = ROI_BOTTOM - LORES_CROP_TOP
        roi_y1 = max(0, roi_y1)
        roi_y2 = min(h, roi_y2)
        mask_h = min(roi_y2 - roi_y1, fg_mask_roi.shape[0])
        if mask_h > 0:
            canvas[roi_y1:roi_y1 + mask_h, :fg_mask_roi.shape[1]] = \
                fg_mask_roi[:mask_h, :]
    ok, buf = cv2.imencode('.png', canvas)
    return buf.tobytes() if ok else None


def _save_and_transfer(frames, event_count, rightward=True, event_meta=None,
                       _preview_state=None, fi_mask=None):
    """
    frames: list of (hires_bgr, stem, bbox_lores, meta) to save as JPEGs.
    rightward: True if vehicle travelled left→right; used for leading-edge blur margin.
    event_meta: optional dict with keys time_str, blob_width, event_frames,
                event_type, velocity — prepended to _preview_state.event_history.
    fi_mask: optional dict mapping fi → fg_mask ndarray (ROI-relative, uint8).
             When provided, a same-stem .png mask is written alongside each JPEG.
    Runs in a daemon thread.
    """
    if shm_free_mb() < SHM_MIN_MB:
        log.warning("[save] /dev/shm low (%.1f MB) — event %d discarded",
                    shm_free_mb(), event_count)
        return
    SHM_BASE.mkdir(parents=True, exist_ok=True)

    # Compute the union of all per-frame lores bboxes so every frame in this
    # batch uses the same (largest) unblurred region — prevents the blur edge
    # from cutting into the vehicle when one frame's bbox is tighter than another.
    valid_bboxes = [b for _, _, b, _ in frames if b is not None]
    if valid_bboxes:
        union_bx  = min(b[0]        for b in valid_bboxes)
        union_by  = min(b[1]        for b in valid_bboxes)
        union_bx2 = max(b[0] + b[2] for b in valid_bboxes)
        union_by2 = max(b[1] + b[3] for b in valid_bboxes)
        union_bbox = (union_bx, union_by,
                      union_bx2 - union_bx, union_by2 - union_by)
    else:
        union_bbox = None

    saved = []
    first_frame = True
    for hires_bgr, stem, _bbox_lores, meta in frames:
        fname = f"{stem}_{event_count}.jpg"
        path  = SHM_BASE / fname
        processed = blur_background(hires_bgr, union_bbox, rightward)
        cropped   = processed[HIRES_CROP_TOP:HIRES_CROP_BOTTOM, :]
        # Convert BGR→RGB for PIL, build EXIF, save as JPEG
        rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        pil_img.save(str(path), format="JPEG", quality=JPEG_QUALITY,
                     exif=_build_exif(meta) if meta else None)
        # Save 50%-scaled thumbnail of the first frame for the web preview
        if first_frame:
            th, tw = cropped.shape[:2]
            thumb = cv2.resize(cropped, (tw // 2, th // 2),
                               interpolation=cv2.INTER_AREA)
            thumb_rgb = cv2.cvtColor(thumb, cv2.COLOR_BGR2RGB)
            Image.fromarray(thumb_rgb).save(
                str(THUMB_PATH), format="JPEG", quality=JPEG_QUALITY)
            # Derive capture time from stem (YYYYMMDD_HHMMSS_mmm)
            if event_meta is not None:
                try:
                    parts    = stem.split('_')
                    date_str = parts[0]              # YYYYMMDD
                    hms      = parts[1]              # HHMMSS
                    ms_str   = parts[2]              # mmm
                    time_str = f"{hms[0:2]}:{hms[2:4]}:{hms[4:6]}.{ms_str}"
                    cap_dt   = datetime.strptime(
                        f"{date_str}{hms}{ms_str}", "%Y%m%d%H%M%S%f")
                    epoch    = cap_dt.timestamp()
                except Exception:
                    time_str = '??:??:??.???'
                    epoch    = 0.0
                    date_str = datetime.now().strftime('%Y%m%d')
                _append_csv_log(
                    event_count, time_str, epoch,
                    event_meta['blob_width'], event_meta['event_frames'],
                    event_meta['event_type'], event_meta.get('velocity'),
                    date_str)
                if _preview_state is not None:
                    info = dict(event_meta)
                    info['time_str']    = time_str
                    info['event_count'] = event_count
                    with _preview_state.lock:
                        _preview_state.event_history = (
                            [info] + _preview_state.event_history)[:5]
            first_frame = False
        saved.append(fname)
        log.info("[save] event %d  → %s", event_count, fname)

        # Write companion mask PNG if enabled and fg_mask data is available for this frame
        if SAVE_MASK and fi_mask is not None:
            mask_roi  = fi_mask.get(stem)
            png_bytes = _build_mask_png(mask_roi)
            if png_bytes:
                mask_fname = f"{stem}_{event_count}_mask.png"
                mask_path  = SHM_BASE / mask_fname
                mask_path.write_bytes(png_bytes)
                log.info("[save] event %d  → %s", event_count, mask_fname)

    start_transfer()


def start_save(frames, event_count, rightward=True, event_meta=None,
               preview_state=None, fi_mask=None):
    t = threading.Thread(target=_save_and_transfer,
                         args=(frames, event_count, rightward),
                         kwargs={'event_meta': event_meta,
                                 '_preview_state': preview_state,
                                 'fi_mask': fi_mask},
                         daemon=True)
    t.start()

# ---------------------------------------------------------------------------
# Live preview server  (Flask MJPEG, port 8080)
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class PreviewState:
    lock:          threading.Lock = dataclasses.field(default_factory=threading.Lock)
    jpeg:          bytes | None   = None   # latest encoded frame
    blob:          dict | None    = None   # latest blob dict (or None between events)
    fg_mask:       object         = None   # latest fg_mask ndarray (ROI coords)
    event_history: list           = dataclasses.field(default_factory=list)  # up to 5 recent events, newest first


_OVERLAY_COLOR = (0, 200, 0)    # green tint for motion mask pixels  (BGR)
_OVERLAY_ALPHA = 0.45
_BBOX_COLOR    = (0, 220, 255)  # cyan-yellow box                     (BGR)
_BBOX_THICK    = 2
_PREVIEW_SCALE = 1              # upscale factor (1 = native lores resolution)
_PREVIEW_PORT  = 8080


def _compose_preview(color_frame, fg_mask, blob) -> bytes:
    """
    Build the annotated preview JPEG:
      - full lores color frame, 2x upscaled (nearest neighbour)
      - green tint where fg_mask is nonzero (ROI band only)
      - bounding box if blob is not None
    Returns JPEG bytes.
    """
    vis = color_frame.copy()
    h, w = vis.shape[:2]

    # Green overlay on foreground pixels within the ROI band
    if fg_mask is not None:
        overlay  = vis.copy()
        roi_view = overlay[ROI_TOP:ROI_BOTTOM, :]
        roi_view[fg_mask > 0] = _OVERLAY_COLOR
        cv2.addWeighted(overlay, _OVERLAY_ALPHA, vis, 1.0 - _OVERLAY_ALPHA, 0, vis)

    # Bounding box — bbox coords are ROI-relative; restore full-frame y
    if blob is not None:
        bx, by, bw, bh = blob['bbox']
        x1 = bx
        y1 = by + ROI_TOP
        cv2.rectangle(vis, (x1, y1), (x1 + bw, y1 + bh), _BBOX_COLOR, _BBOX_THICK)

    # 2x nearest-neighbour upscale
    vis = cv2.resize(vis, (w * _PREVIEW_SCALE, h * _PREVIEW_SCALE),
                     interpolation=cv2.INTER_NEAREST)

    ok, buf = cv2.imencode('.jpg', vis, [cv2.IMWRITE_JPEG_QUALITY, 75])
    return buf.tobytes() if ok else b''


def _preview_writer(state: PreviewState, color_frame, fg_mask):
    """Called from the main loop at ~1 fps. Reads current blob from state."""
    with state.lock:
        blob = state.blob
    jpeg = _compose_preview(color_frame, fg_mask, blob)
    with state.lock:
        state.jpeg = jpeg


def _flask_thread(state: PreviewState, port: int):
    app = Flask(__name__)
    logging.getLogger('werkzeug').setLevel(logging.ERROR)  # silence per-request noise

    @app.route('/')
    def index():
        return Response(
            '<!doctype html>'
            '<html><head><title>CAMA preview</title>'
            '<style>'
            'body{margin:0;background:#111;display:flex;flex-direction:column;'
            'align-items:center;justify-content:center;min-height:100vh;gap:12px}'
            'img{image-rendering:pixelated;max-width:100%}'
            '#thumb{opacity:0.85}'
            '#info{color:#ccc;font-family:monospace;font-size:14px;'
            'background:#1e1e1e;padding:8px 16px;border-radius:4px;'
            'white-space:pre;letter-spacing:0.03em}'
            '</style></head>'
            '<body>'
            '<img src="/stream">'
            '<div id="info">—</div>'
            '<img id="thumb" src="/thumb">'
            '<script>'
            'function refreshInfo(){'
            '  fetch("/info").then(r=>r.json()).then(arr=>{'
            '    if(!arr.length){return;}'
            '    var lines=arr.slice().reverse().map(function(d){'
            '      return "#"+d.event_count'
            '        +"  "+d.time_str'
            '        +"  w="+d.blob_width+"px"'
            '        +"  frames="+String(d.event_frames).padStart(2,"0")'
            '        +"  type="+d.event_type'
            '        +"  vel="+(d.velocity!==null?(d.velocity>=0?"+":"")+d.velocity.toFixed(2):"?")+"px/fr";'
            '    });'
            '    document.getElementById("info").textContent=lines.join("\\n");'
            '  }).catch(()=>{});'
            '}'
            'setInterval(function(){'
            '  var t=document.getElementById("thumb");'
            '  t.src="/thumb?t="+Date.now();'
            '  refreshInfo();'
            '},5000);'
            'refreshInfo();'
            '</script>'
            '</body></html>',
            mimetype='text/html')

    def _gen():
        while True:
            with state.lock:
                frame = state.jpeg
            if frame:
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'
                       + frame + b'\r\n')
            time.sleep(0.05)   # poll at 20 Hz; image itself updates at ~2 fps

    @app.route('/stream')
    def stream():
        return Response(_gen(),
                        mimetype='multipart/x-mixed-replace; boundary=frame')

    @app.route('/thumb')
    def thumb():
        if THUMB_PATH.exists():
            return Response(THUMB_PATH.read_bytes(), mimetype='image/jpeg')
        # Return a 1x1 transparent GIF as placeholder until first event
        return Response(
            b'GIF89a\x01\x00\x01\x00\x00\xff\x00,\x00\x00\x00\x00'
            b'\x01\x00\x01\x00\x00\x02\x00;',
            mimetype='image/gif')

    @app.route('/info')
    def info():
        import json as _json
        with state.lock:
            history = list(state.event_history)
        return Response(_json.dumps(history), mimetype='application/json')

    app.run(host='0.0.0.0', port=port, threaded=True, use_reloader=False)


def start_preview_server(state: PreviewState, port: int = _PREVIEW_PORT):
    t = threading.Thread(target=_flask_thread, args=(state, port), daemon=True)
    t.start()
    log.info("[preview] serving on http://0.0.0.0:%d/", port)


# ---------------------------------------------------------------------------
# Detection helpers
# ---------------------------------------------------------------------------

def apply_correction(img, factor):
    return np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)


def find_vehicle_blob(mask):
    """
    Find largest connected component, refine with convex hull of that component.
    Returns (blob_dict, reject_reason).
    blob_dict is None if no blob passes filters.
    reject_reason is a string describing why a near-miss blob was rejected,
    or None if the blob passed or was too small to be worth reporting.
    """
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask, connectivity=8)

    best_label = None
    best_area  = 0
    for i in range(1, n_labels):
        if stats[i, cv2.CC_STAT_AREA] > best_area:
            best_area  = stats[i, cv2.CC_STAT_AREA]
            best_label = i

    if best_label is None:
        return None, None

    component_mask = (labels == best_label).astype(np.uint8) * 255
    pts = cv2.findNonZero(component_mask)
    if pts is None or len(pts) < 3:
        return None, None

    hull   = cv2.convexHull(pts)
    hx, hy, hw, hh = cv2.boundingRect(hull)
    area   = cv2.contourArea(hull)
    aspect = hw / max(hh, 1)
    fill   = area / max(hw * hh, 1)

    m = cv2.moments(hull)
    cx = (m['m10'] / m['m00']) if m['m00'] != 0 else hx + hw / 2.0
    cy = (m['m01'] / m['m00']) if m['m00'] != 0 else hy + hh / 2.0

    if area < MIN_BLOB_AREA:   return None, None
    if hw   < MIN_BLOB_WIDTH:  return None, None

    # From here, the blob is large enough to be worth reporting if rejected
    fail = []
    if hh     < MIN_BLOB_HEIGHT:     fail.append(f'h={hh}<{MIN_BLOB_HEIGHT}')
    if aspect < MIN_ASPECT_RATIO:    fail.append(f'ar={aspect:.2f}<{MIN_ASPECT_RATIO}')
    if fill   < MIN_HULL_FILL_RATIO: fail.append(f'fill={fill:.2f}<{MIN_HULL_FILL_RATIO}')
    blob_bottom = hy + hh             # ROI-relative bottom edge of bounding box
    if blob_bottom < MIN_BLOB_BOTTOM_ROW:
        fail.append(f'bottom={blob_bottom}<{MIN_BLOB_BOTTOM_ROW}(grass/sky blob)')
    if fail:
        reason = f'area={area:.0f} w={hw} h={hh} ar={aspect:.2f} fill={fill:.2f} cx={cx:.0f} FAIL({" ".join(fail)})'
        return None, reason

    touches_left  = hx <= EDGE_MARGIN
    touches_right = (hx + hw) >= (FRAME_WIDTH - EDGE_MARGIN)
    return {
        'bbox':           (hx, hy, hw, hh),
        'area':           area,
        'aspect_ratio':   aspect,
        'centroid_x':     cx,
        'centroid_y':     cy,
        'touches_left':   touches_left,
        'touches_right':  touches_right,
        'fully_interior': not touches_left and not touches_right,
    }, None


def estimate_center_frame(centroid_history):
    """
    Linear fit of centroid_x vs frame_index.
    Returns (predicted_fi: float, used_interior: bool) or (None, False).
    """
    interior = [(fi, cx) for fi, cx, full in centroid_history if full]
    subset   = interior if len(interior) >= 4 else \
               [(fi, cx) for fi, cx, _ in centroid_history] \
               if len(centroid_history) >= 4 else None
    if subset is None:
        return None, False

    used_interior = (subset is interior)
    idx    = np.array([s[0] for s in subset], dtype=np.float32)
    xs     = np.array([s[1] for s in subset], dtype=np.float32)
    coeffs = np.polyfit(idx, xs, 1)
    if abs(coeffs[0]) < 1e-3:
        return None, False
    return (FRAME_WIDTH / 2.0 - coeffs[1]) / coeffs[0], used_interior

def split_centroid_history(centroid_history):
    """
    Detect a tow-vehicle/trailer event by finding the first large backward
    centroid step (against the direction of overall travel).  Returns
    (phase1, phase2) where phase2 may be empty if no reversal is found.
    Direction of travel is inferred from the sign of overall cx displacement.
    """
    if len(centroid_history) < 4:
        return centroid_history, []

    cx_vals = [cx for _, cx, _ in centroid_history]
    direction = 1 if (cx_vals[-1] - cx_vals[0]) >= 0 else -1  # +1 rightward

    for i in range(1, len(centroid_history)):
        step = (cx_vals[i] - cx_vals[i - 1]) * direction  # negative = reversal
        if step < -CENTROID_REVERSAL_THRESHOLD:
            return centroid_history[:i], centroid_history[i:]

    return centroid_history, []


def pick_and_save(phase_history, vehicle_frame_indices, hires_buf,
                  is_long, event_count, label, fi_bbox, rightward=True,
                  event_meta=None, preview_state=None, fi_mask=None):
    """
    Given a centroid phase and the hires buffer, select two frames and save.
    label is a short string for logging ('lead' or 'trailer').
    fi_bbox maps fi → lores bbox for precise per-frame blur.
    Returns True if save was initiated.
    """
    pred, used_int = estimate_center_frame(phase_history)
    src = "interior" if used_int else "fallback"

    phase_fis = [fi for fi, _, _ in phase_history]
    fi_min = min(phase_fis) if phase_fis else None
    fi_max = max(phase_fis) if phase_fis else None

    if (pred is not None and fi_min is not None
            and fi_min <= pred <= fi_max):
        chosen_fi = min(vehicle_frame_indices, key=lambda i: abs(i - pred),
                        default=None)
        chosen_by = f"linear fit ({src})"
    else:
        # Fall back to frame closest to centre of phase fi range
        mid = (fi_min + fi_max) / 2.0 if fi_min is not None else None
        chosen_fi = (min(vehicle_frame_indices, key=lambda i: abs(i - mid),
                         default=None)
                     if mid is not None else None)
        chosen_by = "phase midpoint"

    if chosen_fi is None or not hires_buf:
        log.warning("  [%s] no frame available — skipping", label)
        return False

    lag = HIRES_LAG_FRAMES
    adjusted_fi = chosen_fi + lag
    to_save = _select_frames(hires_buf, adjusted_fi)
    if to_save is None:
        log.warning("  [%s] target fi=%d too far from buffer window — skipping",
                    label, adjusted_fi)
        return False
    log.info("  [%s] saving fi=%s  (chosen_fi=%d  pred=%s  by=%s  lag=%d)",
             label, [e[3] for e in to_save], chosen_fi,
             f"{pred:.1f}" if pred is not None else "–", chosen_by, lag)
    frames_to_save = [(bgr, stem, fi_bbox.get(entry_fi), meta) for bgr, stem, meta, entry_fi in to_save]
    start_save(frames_to_save, event_count, rightward,
               event_meta=event_meta, preview_state=preview_state,
               fi_mask=fi_mask)
    return True


def save_long_vehicle_ends(vehicle_edge_frames, hires_buf, event_count, fi_bbox,
                           rightward=True, event_meta=None, preview_state=None,
                           fi_mask=None):
    """
    For a long vehicle, save two pairs of frames:
      - front pair: nearest to when the leading edge first touched the far side
        (front of vehicle about to exit)
      - back pair: nearest to when the trailing edge last touched the near side
        (rear of vehicle just after entry)

    Direction is inferred from which side is touched first across the event.
    HIRES_LAG_FRAMES is applied to both anchors.
    fi_bbox maps fi → lores bbox for precise per-frame blur.
    """
    if not vehicle_edge_frames or not hires_buf:
        log.warning("  [long] no edge frames or empty buffer — skipping")
        return

    # Infer direction: whichever side is touched in the first frame that touches
    # either edge determines the near side (entry side).
    near_left = None
    for _, tl, tr in vehicle_edge_frames:
        if tl or tr:
            near_left = tl   # True = entered from left, False = entered from right
            break
    if near_left is None:
        # No edge touches at all — fall back to first/last frames
        front_fi = vehicle_edge_frames[-1][0]
        back_fi  = vehicle_edge_frames[0][0]
    else:
        far_left = not near_left  # far side = where front exits

        # Front anchor: first frame where leading edge touches the far side
        front_fi = None
        for fi, tl, tr in vehicle_edge_frames:
            touches_far = tl if far_left else tr
            if touches_far:
                front_fi = fi
                break

        # Back anchor: last frame where trailing edge touches the near side
        back_fi = None
        for fi, tl, tr in reversed(vehicle_edge_frames):
            touches_near = tl if near_left else tr
            if touches_near:
                back_fi = fi
                break

        if front_fi is None:
            front_fi = vehicle_edge_frames[-1][0]
            log.info("  [long] no far-edge touch found — using last frame as front anchor")
        if back_fi is None:
            back_fi = vehicle_edge_frames[0][0]
            log.info("  [long] no near-edge touch found — using first frame as back anchor")

    for label, anchor_fi in (("front", front_fi), ("back", back_fi)):
        adjusted = anchor_fi + HIRES_LAG_FRAMES
        to_save = _select_frames(hires_buf, adjusted)
        if to_save is None:
            log.warning("  [long %s] target fi=%d too far from buffer window — skipping",
                        label, adjusted)
            continue
        log.info("  [long %s] saving fi=%s  (anchor_fi=%d  lag=%d)",
                 label, [e[3] for e in to_save], anchor_fi, HIRES_LAG_FRAMES)
        frames_to_save = [(bgr, stem, fi_bbox.get(entry_fi), meta)
                          for bgr, stem, meta, entry_fi in to_save]
        start_save(frames_to_save, event_count, rightward,
                   event_meta=event_meta, preview_state=preview_state,
                   fi_mask=fi_mask)



def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--shutter", type=int,   default=None,
                   help="Fix exposure time µs (default: auto)")
    p.add_argument("--gain",    type=float, default=None,
                   help="Fix analogue gain (default: auto)")
    p.add_argument("--tune",    type=str,
                   default=str(Path(__file__).parent / "custom_pisp.json"),
                   help="Tuning JSON path")
    p.add_argument("--ev",      type=float, default=-1.0,
                   help="Exposure value bias in stops (default: %(default)s)")
    p.add_argument("--verbose-fg", action="store_true",
                   help="Log foreground pixel count every frame (diagnostic for weak detections)")
    return p.parse_args()

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    log.info("vehicle_detect  version %s", VERSION)
    args = parse_args()

    running = True

    def handle_sigint(sig, frame):
        nonlocal running
        log.info("Interrupted — stopping.")
        running = False

    signal.signal(signal.SIGINT, handle_sigint)

    # --- Camera init ---------------------------------------------------------
    kwargs = {}
    if args.tune and Path(args.tune).exists():
        kwargs["tuning"] = Picamera2.load_tuning_file(args.tune)
        log.info("[tuning] loaded %s", args.tune)
    else:
        if args.tune:
            log.warning("[tuning] not found: %s — using default", args.tune)

    picam2 = Picamera2(**kwargs)

    cam_controls = {
        "NoiseReductionMode": libctrls.draft.NoiseReductionModeEnum.Minimal,
        "FrameRate": FRAME_RATE,
        "AeExposureMode": libctrls.AeExposureModeEnum.Custom,
    }
    if args.shutter is not None:
        cam_controls["ExposureTime"] = args.shutter
    if args.gain is not None:
        cam_controls["AnalogueGain"] = args.gain
    if args.ev != 0.0:
        cam_controls["ExposureValue"] = args.ev

    config = picam2.create_video_configuration(
        main     = {"size": HIRES_SIZE, "format": "BGR888"},
        lores    = {"size": LORES_SIZE, "format": "YUV420"},
        controls = cam_controls,
        transform = Transform(hflip=1, vflip=1),
        buffer_count = 6,
    )
    picam2.configure(config)

    lores_valid_w, lores_valid_h = picam2.camera_configuration()["lores"]["size"]
    log.info("Lores valid size: %dx%d", lores_valid_w, lores_valid_h)

    # --- MOG2 setup ----------------------------------------------------------
    fgbg = cv2.createBackgroundSubtractorMOG2(
        history=MOG2_HISTORY,
        varThreshold=MOG2_VAR_THRESHOLD,
        detectShadows=MOG2_DETECT_SHADOWS,
    )
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE))

    # --- State ---------------------------------------------------------------
    # Rolling buffer of (hires_bgr, stem, meta, fi) — most recent HIRES_BUFFER_SIZE frames
    hires_buf: collections.deque = collections.deque(maxlen=HIRES_BUFFER_SIZE)

    # Metadata ring buffer for delay compensation
    meta_buf: collections.deque = collections.deque(maxlen=METADATA_DELAY_FRAMES + 1)

    consecutive           = 0
    vehicle_active        = False
    lockout_remaining     = 0
    event_frames          = 0      # frames since vehicle_active became True
    centroid_history      = []     # (fi, cx, fully_interior)
    vehicle_frame_indices = []     # fi of every VEHICLE-tagged frame
    vehicle_edge_frames   = []     # (fi, touches_left, touches_right) for same frames
    fi_bbox               = {}     # fi → (bx, by, bw, bh) lores bbox for event frames
    fi_mask               = {}     # stem → fg_mask ndarray (ROI-relative) for event frames
    best_center_idx       = None   # global frame index
    best_center_dist      = float('inf')
    best_center_cx        = None
    max_blob_width        = 0      # max hull width seen during this event
    stuck_frames          = 0      # frames of continuous non-vehicle fg activity
    frame_count           = 0
    fi                    = 0      # detection frame index (increments every frame)

    # Load daily event counter — reset to 0 if date has changed since last run
    _saved_num, _saved_date = _load_event_num()
    today = datetime.now().date()
    event_count = _saved_num if _saved_date == today.strftime('%Y%m%d') else 0
    log.info("[event_num] starting at event %d for %s", event_count + 1, today)

    SHM_BASE.mkdir(parents=True, exist_ok=True)

    # --- Preview server ------------------------------------------------------
    preview_state = PreviewState()
    start_preview_server(preview_state)

    # --- NTP + sensor anchor -------------------------------------------------
    picam2.start()
    _wait_for_ntp()
    global _wall_anchor, _sensor_anchor_ns
    _anchor_meta      = picam2.capture_metadata()
    _wall_anchor      = datetime.now()
    _sensor_anchor_ns = _anchor_meta.get("SensorTimestamp")
    if _sensor_anchor_ns is not None:
        log.info("[timing] sensor clock anchored")
    else:
        log.warning("[timing] SensorTimestamp unavailable — using datetime.now()")

    log.info("Warming up MOG2 background model (%d frames)…", MOG2_HISTORY)

    # Target the first whole-minute boundary after startup for status logging
    # and fi/day reset checks.  Thereafter advance by exactly 60 seconds each time.
    _t = datetime.now()
    _next_minute = _t.replace(second=0, microsecond=0) + timedelta(seconds=60)

    # --- Capture loop --------------------------------------------------------
    while running:
        request = picam2.capture_request()
        try:
            meta         = request.get_metadata()
            sensor_ts_ns = meta.get("SensorTimestamp")
            dt           = sensor_ts_to_datetime(sensor_ts_ns)
            stem         = ts_stem(dt)

            lores_arr = request.make_array("lores")
            hires_bgr = request.make_array("main")        # labelled BGR888 but actually RGB
            hires_bgr = cv2.cvtColor(hires_bgr, cv2.COLOR_RGB2BGR)  # fix to true BGR for imwrite
        finally:
            request.release()

        frame_count += 1

        # Push metadata into delay buffer; retrieve the delayed entry
        meta_buf.append(meta)
        delayed_meta = meta_buf[0] if len(meta_buf) == meta_buf.maxlen else meta

        exp    = delayed_meta.get("ExposureTime") or 1.0
        ag     = delayed_meta.get("AnalogueGain")  or 1.0
        dg     = delayed_meta.get("DigitalGain")   or 1.0
        ev     = float(exp) * float(ag) * float(dg)

        # Baseline EV anchored to first frame; re-anchored after prolonged
        # unsettled state to handle gradual lighting changes (e.g. sunset).
        if frame_count == 1:
            ev_baseline = ev
        factor = ev_baseline / ev if ev > 0 else 1.0

        # Store high-res frame in rolling buffer (includes fi for later selection)
        hires_buf.append((hires_bgr.copy(), stem, meta, fi))

        # Convert lores YUV420 to BGR for MOG2
        buf_w  = lores_arr.shape[1]
        yuv    = lores_arr[:lores_valid_h * 3 // 2, :buf_w]
        lores_bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)
        lores_bgr = lores_bgr[:lores_valid_h, :lores_valid_w]

        # Apply exposure correction and crop to ROI
        color = apply_correction(lores_bgr, factor)
        roi   = color[ROI_TOP:ROI_BOTTOM, :]

        # MOG2 learning rate — freeze during vehicle events; boost when AGC
        # is stuck unsettled; re-anchor ev_baseline after stage-2 boost so
        # gradual lighting changes (sunset) don't keep the factor permanently off.
        agc_settled = abs(factor - 1.0) < AGC_SETTLED_THRESHOLD

        if vehicle_active:
            lr = LR_VEHICLE
            stuck_frames = 0
        elif agc_settled:
            lr = LR_NORMAL
            stuck_frames = 0
        else:
            stuck_frames += 1
            if stuck_frames >= LR_BOOST2_AFTER:
                lr = LR_BOOST2
                if stuck_frames == LR_BOOST2_AFTER:
                    log.info("[adapt] stuck %d frames (factor=%.4f) — stage-2 boost LR=%.2f",
                             stuck_frames, factor, LR_BOOST2)
                # After running stage-2 boost for a full LR_BOOST2_AFTER interval,
                # re-anchor the baseline so factor returns near 1.0 and normal
                # operation can resume. This handles gradual lighting drift.
                if stuck_frames >= LR_BOOST2_AFTER * 2:
                    ev_baseline = ev
                    stuck_frames = 0
                    lr = LR_NORMAL
                    log.info("[adapt] re-anchoring ev_baseline=%.1f after prolonged boost", ev)
                else:
                    lr = LR_BOOST2
            elif stuck_frames >= LR_BOOST_AFTER:
                lr = LR_BOOST
                if stuck_frames == LR_BOOST_AFTER:
                    log.info("[adapt] stuck %d frames (factor=%.4f) — stage-1 boost LR=%.3f",
                             stuck_frames, factor, LR_BOOST)
            else:
                lr = LR_VEHICLE
        fg_mask = fgbg.apply(roi, learningRate=lr)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN,  kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)

        if args.verbose_fg:
            fg_count = cv2.countNonZero(fg_mask)
            if fg_count >= VERBOSE_FG_MIN_PIXELS:
                log.info("  fi=%d  fg_pixels=%d  factor=%.4f", fi, fg_count, factor)

        # Blob detection
        if lockout_remaining > 0:
            lockout_remaining -= 1
            blob          = None
            reject_reason = None
            with preview_state.lock:
                preview_state.blob = None
        else:
            blob, reject_reason = find_vehicle_blob(fg_mask)
            if reject_reason:
                log.info("  fi=%d  REJECTED  %s", fi, reject_reason)
            with preview_state.lock:
                preview_state.blob = blob

        # --- Event logic -----------------------------------------------------
        if blob:
            consecutive += 1
            cx = blob['centroid_x']

            # Clear centroid history at the start of each new detection run so
            # stale entries from previous unconfirmed blobs don't corrupt the
            # motion consistency calculation.
            if consecutive == 1:
                centroid_history = []

            bx, by, bw, bh = blob['bbox']
            fi_bbox[fi] = (bx, by, bw, bh)
            fi_mask[stem] = fg_mask.copy() if SAVE_MASK else None
            edges = ('L' if blob['touches_left'] else '.') + \
                    ('R' if blob['touches_right'] else '.')

            if blob['area'] >= MIN_CENTROID_AREA:
                centroid_history.append((fi, cx, blob['fully_interior']))

            if consecutive >= MIN_CONSECUTIVE_FRAMES:
                if not vehicle_active:
                    log.info("VEHICLE START  frame=%d", frame_count)
                vehicle_active  = True
                event_frames   += 1
                vehicle_frame_indices.append(fi)
                vehicle_edge_frames.append((fi, blob['touches_left'], blob['touches_right']))
                max_blob_width  = max(max_blob_width, bw)
                log.info("  fi=%d  cx=%.1f  w=%d  h=%d  ar=%.2f  edges=%s",
                         fi, cx, bw, bh, blob['aspect_ratio'], edges)

                if blob['fully_interior']:
                    dist = abs(cx - FRAME_WIDTH / 2)
                    if dist < best_center_dist:
                        best_center_dist = dist
                        best_center_idx  = fi
                        best_center_cx   = cx
            else:
                log.info("  fi=%d  cx=%.1f  w=%d  h=%d  ar=%.2f  edges=%s  [pre-confirm %d/%d]",
                         fi, cx, bw, bh, blob['aspect_ratio'], edges,
                         consecutive, MIN_CONSECUTIVE_FRAMES)

            # Force-close event if it runs too long
            if vehicle_active and event_frames >= MAX_EVENT_FRAMES:
                log.warning("VEHICLE TIMEOUT  frame=%d — forcing close", frame_count)
                blob = None   # fall through to close logic below
                # Re-adapt MOG2 quickly to current background
                for _ in range(10):
                    fgbg.apply(roi, learningRate=0.5)

        if not blob and vehicle_active:
            # --- Vehicle event ended -----------------------------------------
            pred, used_int = estimate_center_frame(centroid_history)
            src = "interior" if used_int else "fallback"

            # Select best frame index using linear fit if available and
            # prediction falls within the observed vehicle frame range.
            if (pred is not None and vehicle_frame_indices and
                    min(vehicle_frame_indices) <= pred <= max(vehicle_frame_indices)):
                chosen_fi = min(vehicle_frame_indices, key=lambda i: abs(i - pred))
                chosen_by = f"linear fit ({src})"
            else:
                chosen_fi = best_center_idx
                chosen_by = "centroid proximity"

            # Check for tow-vehicle/trailer split before motion quality check,
            # so a valid split can bypass the consistency gate.
            phase1, phase2 = split_centroid_history(centroid_history)

            def _phase_consistent(phase):
                """Return True if this phase shows directed motion rather than jitter."""
                cx_p = [cx for _, cx, _ in phase]
                if len(cx_p) < 2:
                    return False
                steps    = [cx_p[i] - cx_p[i-1] for i in range(1, len(cx_p))]
                path_len = sum(min(abs(s), MAX_STEP_FOR_CONSISTENCY) for s in steps)
                net_disp = abs(cx_p[-1] - cx_p[0])
                if path_len == 0:
                    return False
                consistency = net_disp / path_len
                travel = max(cx_p) - min(cx_p)
                return (consistency >= MIN_MOTION_CONSISTENCY
                        and travel >= MIN_CENTROID_TRAVEL)

            p1_consistent = _phase_consistent(phase1)
            p2_consistent = phase2 and _phase_consistent(phase2)

            valid_split = (phase2
                           and len(phase1) >= MIN_CONSECUTIVE_FRAMES
                           and len(phase2) >= MIN_CONSECUTIVE_FRAMES
                           and p1_consistent
                           and p2_consistent)

            # phase1_only: phase1 is a valid directed vehicle but phase2 is an
            # AGC artifact or stationary background patch (fails consistency).
            # Re-evaluate all motion quality checks using phase1 points only so
            # the artifact tail cannot poison the stationarity or consistency tests.
            phase1_only = (phase2
                           and len(phase1) >= MIN_CONSECUTIVE_FRAMES
                           and p1_consistent
                           and not p2_consistent)

            # Use phase1 points for motion quality when phase1_only, else full history.
            eval_history = phase1 if phase1_only else centroid_history
            if phase1_only:
                log.info("  [phase1_only] phase2 failed consistency — "
                         "evaluating motion on phase1 (%d pts) only", len(phase1))

            # Check motion quality — reject stationary blobs and random jitter.
            # Skip for valid tow+trailer splits or phase1_only saves: phase
            # consistency is already confirmed above.
            cx_vals = [cx for _, cx, _ in eval_history]
            n_cx = len(cx_vals)
            # Direction of travel for leading-edge blur margin
            rightward = (cx_vals[-1] >= cx_vals[0]) if n_cx >= 2 else True

            # Stationarity check on the tail of confirmed frames.
            # A stationary foreground object (parked car, shadow, post) shows
            # near-zero cx variance in its final confirmed frames and always
            # triggers a timeout.  Real vehicles are still moving (or just exited)
            # at event end, so their tail cx std is well above threshold.
            # Long vehicles (max_blob_width >= HIRES_LONG_VEHICLE_WIDTH) are exempt:
            # their blob spans the full frame width, producing few or no interior
            # (fully_interior) centroid points, and a steady cx is expected while
            # the vehicle fills the frame.
            STATIONARY_TAIL_N = 10
            is_long_vehicle = max_blob_width >= HIRES_LONG_VEHICLE_WIDTH
            confirmed_fi_set = set(vehicle_frame_indices)
            confirmed_cx = [cx for fi, cx, _ in eval_history if fi in confirmed_fi_set]
            if (MIN_CONFIRMED_CX_STD > 0
                    and not is_long_vehicle
                    and len(confirmed_cx) >= STATIONARY_TAIL_N):
                tail_cx = confirmed_cx[-STATIONARY_TAIL_N:]
                confirmed_cx_std = float(np.std(tail_cx))
                stationary = confirmed_cx_std < MIN_CONFIRMED_CX_STD
            else:
                confirmed_cx_std = None
                stationary = False

            if n_cx >= 4:
                steps      = [cx_vals[i] - cx_vals[i - 1] for i in range(1, n_cx)]
                path_len   = sum(min(abs(s), MAX_STEP_FOR_CONSISTENCY) for s in steps)
                net_disp   = abs(cx_vals[-1] - cx_vals[0])
                consistency = (net_disp / path_len) if path_len > 0 else 0.0
                travel      = max(cx_vals) - min(cx_vals)   # kept for log only
                motion_ok   = (not stationary) and (valid_split or phase1_only
                               or consistency >= MIN_MOTION_CONSISTENCY)
                motion_desc = (f"consistency={consistency:.2f}  "
                               f"net={net_disp:.1f}  path={path_len:.1f}  "
                               f"travel={travel:.1f}"
                               + (f"  cx_std={confirmed_cx_std:.1f}" if confirmed_cx_std is not None else ""))
            else:
                travel      = (max(cx_vals) - min(cx_vals)) if n_cx >= 2 else 0.0
                consistency = None
                motion_ok   = (not stationary) and (valid_split or phase1_only
                               or travel >= MIN_CENTROID_TRAVEL)
                motion_desc = (f"travel={travel:.1f}  (short history, n={n_cx})"
                               + (f"  cx_std={confirmed_cx_std:.1f}" if confirmed_cx_std is not None else ""))

            log.info("VEHICLE END  event=%d  best_fi=%s  pred=%.1f  src=%s  by=%s  %s",
                     event_count + 1,
                     chosen_fi if chosen_fi is not None else '–',
                     pred if pred is not None else float('nan'),
                     src, chosen_by, motion_desc)

            if not motion_ok:
                reason = "stationary object" if stationary else "likely false trigger"
                log.warning("VEHICLE DISCARDED  %s — %s; "
                            "resetting background model", motion_desc, reason)
                # Force MOG2 to rapidly re-learn the current frame as background
                for _ in range(20):
                    fgbg.apply(roi, learningRate=0.5)
                ev_baseline = ev   # re-anchor EV baseline to current exposure
                stuck_frames = 0

            elif chosen_fi is not None and hires_buf:
                is_long = max_blob_width >= HIRES_LONG_VEHICLE_WIDTH

                _, _last_date = _load_event_num()
                if _last_date != datetime.now().strftime('%Y%m%d'):
                    event_count = 0
                event_count += 1
                _save_event_num(event_count)

                # Compute signed velocity (lores px/frame) from eval_history
                # (phase1 only when phase1_only, else full centroid history).
                _velocity = None
                if len(eval_history) >= 4:
                    _cx_idx = np.array([f for f, _, _ in eval_history], dtype=np.float32)
                    _cx_val = np.array([c for _, c, _ in eval_history], dtype=np.float32)
                    _vel_coeffs = np.polyfit(_cx_idx, _cx_val, 1)
                    _velocity = float(_vel_coeffs[0])

                _event_type = ('split' if valid_split
                               else 'phase1' if phase1_only
                               else 'long' if is_long
                               else 'normal')
                _event_meta = {
                    'blob_width':   max_blob_width,
                    'event_frames': len(vehicle_frame_indices),
                    'event_type':   _event_type,
                    'velocity':     _velocity,
                }

                if valid_split:
                    # Two-phase event: save lead vehicle then trailer
                    log.info("  [split] tow+trailer detected  "
                             "(phase1=%d pts, phase2=%d pts)",
                             len(phase1), len(phase2))
                    pick_and_save(phase1, vehicle_frame_indices, hires_buf,
                                  is_long, event_count, "lead", fi_bbox, rightward,
                                  event_meta=_event_meta, preview_state=preview_state,
                                  fi_mask=fi_mask)
                    pick_and_save(phase2, vehicle_frame_indices, hires_buf,
                                  is_long, event_count, "trailer", fi_bbox, rightward,
                                  fi_mask=fi_mask)
                elif phase1_only:
                    # Vehicle followed by AGC artifact: save using phase1 only
                    log.info("  [phase1_only] saving phase1 as single vehicle "
                             "(%d pts)", len(phase1))
                    pick_and_save(phase1, vehicle_frame_indices, hires_buf,
                                  is_long, event_count, "phase1", fi_bbox, rightward,
                                  event_meta=_event_meta, preview_state=preview_state,
                                  fi_mask=fi_mask)
                elif is_long:
                    # Long single vehicle: save front and back pairs
                    save_long_vehicle_ends(vehicle_edge_frames, hires_buf,
                                          event_count, fi_bbox, rightward,
                                          event_meta=_event_meta,
                                          preview_state=preview_state,
                                          fi_mask=fi_mask)
                else:
                    # Normal single-vehicle save
                    lag = HIRES_LAG_FRAMES
                    adjusted_fi = chosen_fi + lag
                    to_save = _select_frames(hires_buf, adjusted_fi)
                    if to_save is None:
                        log.warning("  target fi=%d too far from buffer window — skipping",
                                    adjusted_fi)
                    else:
                        log.info("  saving fi=%s  (chosen_fi=%d  lag=%d  max_w=%d)",
                                 [e[3] for e in to_save], chosen_fi, lag,
                                 max_blob_width)
                        frames_to_save = [(bgr, stem, fi_bbox.get(entry_fi), meta)
                                          for bgr, stem, meta, entry_fi in to_save]
                        start_save(frames_to_save, event_count, rightward,
                                   event_meta=_event_meta,
                                   preview_state=preview_state,
                                   fi_mask=fi_mask)
            elif motion_ok and vehicle_frame_indices and hires_buf:
                # No fully-interior best frame (vehicle spanned the full width
                # throughout).
                # Reject if every confirmed frame touched both edges — this is
                # a global illumination event (cloud shadow, sun glint) not a vehicle.
                all_lr = all(tl and tr for _, tl, tr in vehicle_edge_frames)
                if all_lr:
                    log.warning("VEHICLE DISCARDED  all frames edges=LR — "
                                "global illumination event, not a vehicle; "
                                "resetting background model")
                    for _ in range(20):
                        fgbg.apply(roi, learningRate=0.5)
                    ev_baseline = ev
                    stuck_frames = 0
                else:
                    is_long = max_blob_width >= HIRES_LONG_VEHICLE_WIDTH
                    _, _last_date = _load_event_num()
                    if _last_date != datetime.now().strftime('%Y%m%d'):
                        event_count = 0
                    event_count += 1
                    _save_event_num(event_count)

                    _velocity = None
                    if len(eval_history) >= 4:
                        _cx_idx = np.array([f for f, _, _ in eval_history], dtype=np.float32)
                        _cx_val = np.array([c for _, c, _ in eval_history], dtype=np.float32)
                        _vel_coeffs = np.polyfit(_cx_idx, _cx_val, 1)
                        _velocity = float(_vel_coeffs[0])

                    _event_type = 'long' if is_long else 'normal'
                    _event_meta = {
                        'blob_width':   max_blob_width,
                        'event_frames': len(vehicle_frame_indices),
                        'event_type':   _event_type,
                        'velocity':     _velocity,
                    }

                    if is_long:
                        save_long_vehicle_ends(vehicle_edge_frames, hires_buf,
                                              event_count, fi_bbox, rightward,
                                              event_meta=_event_meta,
                                              preview_state=preview_state,
                                              fi_mask=fi_mask)
                    else:
                        # Save two frames nearest the fi midpoint
                        mid_fi = (min(vehicle_frame_indices) + max(vehicle_frame_indices)) / 2.0
                        to_save = _select_frames(hires_buf, mid_fi)
                        if to_save is None:
                            log.warning("  mid_fi=%.1f too far from buffer window — skipping",
                                        mid_fi)
                        else:
                            log.info("  saving fi=%s  (mid_fi=%.1f  no interior frames)",
                                     [e[3] for e in to_save], mid_fi)
                            frames_to_save = [(bgr, stem, fi_bbox.get(entry_fi), meta)
                                              for bgr, stem, meta, entry_fi in to_save]
                            start_save(frames_to_save, event_count, rightward,
                                       event_meta=_event_meta,
                                       preview_state=preview_state,
                                       fi_mask=fi_mask)
            else:
                log.warning("VEHICLE END  no best frame available — skipping save")

            # Reset state
            vehicle_active        = False
            consecutive           = 0
            event_frames          = 0
            centroid_history      = []
            vehicle_frame_indices = []
            vehicle_edge_frames   = []
            fi_bbox               = {}
            fi_mask               = {}
            best_center_idx       = None
            best_center_dist      = float('inf')
            best_center_cx        = None
            max_blob_width        = 0
            stuck_frames          = 0
            lockout_remaining     = LOCKOUT_FRAMES

        elif not blob:
            # No event, no vehicle: reset consecutive counter
            consecutive  = 0
            event_frames = 0

        _now = datetime.now()
        if _now >= _next_minute:
            _next_minute += timedelta(seconds=60)
            log.info("frame=%d  events=%d  free_shm=%.0f MB  lr=%.4f",
                     frame_count, event_count, shm_free_mb(), lr)
            _new_day = _now.date()
            if _new_day != today:
                today = _new_day
                fi    = 0
                log.info("[daily] new day %s — fi reset to 0", today)

        # Write preview frame at ~4 fps
        if fi % int(FRAME_RATE / 4) == 0:
            _preview_writer(preview_state, color, fg_mask)

        fi += 1

    picam2.stop()
    log.info("Done. %d events captured, %d frames processed.",
             event_count, frame_count)


if __name__ == "__main__":
    main()
