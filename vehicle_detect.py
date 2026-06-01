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

import cv2
import numpy as np
from libcamera import Transform, controls as libctrls
from picamera2 import Picamera2

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

VERSION        = "1.031"
FRAME_RATE     = 20.0
LORES_SIZE     = (320, 240)
HIRES_SIZE     = (1456, 1088)
JPEG_QUALITY   = 95

# Crop applied to saved full-res JPEGs (in 1456x1088 frame coordinates)
HIRES_CROP_TOP    = 76
HIRES_CROP_BOTTOM = 830

SHM_BASE       = Path("/dev/shm/burst")
SHM_MIN_MB     = 100
REMOTE_HOST    = "jbeale@jbeale-mini.local"
REMOTE_DIR     = "/mnt/bluecherry/CAMA/"

PRINT_INTERVAL = 1200   # frames between status log lines (~1 minute at 20fps)

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
MOG2_VAR_THRESHOLD  = 50
MOG2_DETECT_SHADOWS = False

MORPH_KERNEL_SIZE = 5

MIN_BLOB_AREA       = 100 # catch smaller objects like bicyclists
MIN_ASPECT_RATIO    = 0.8
MIN_BLOB_WIDTH      = 50
MIN_BLOB_HEIGHT     = 20   # lowered from 25 to avoid intermittent streak resets on dark cars
MIN_HULL_FILL_RATIO = 0.44 # lowered from 0.45 to avoid boundary ties
MIN_CENTROID_AREA   = 800 # was 3000, to see motorcycles and bicyclists

# Minimum foreground pixel count to log when --verbose-fg is active.
VERBOSE_FG_MIN_PIXELS = 150

MIN_CONSECUTIVE_FRAMES = 4
LOCKOUT_FRAMES         = 15

# Minimum centroid travel (lores px) across the full event to accept it as a
# real vehicle pass. Stationary or near-stationary blobs (e.g. lighting artefacts)
# are discarded and trigger a background reset.
MIN_CENTROID_TRAVEL = 80

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
MAX_EVENT_FRAMES = int(FRAME_RATE * 8)   # 8 seconds

# Rolling buffer of high-res frames kept in RAM.
# Must be large enough to reach back to the best-centered frame even after
# the vehicle has fully exited. At 20fps and up to ~3s visible, 60 frames
# (270 MB at 1456x1088 BGR) gives ample headroom within the 2 GB /dev/shm.
HIRES_BUFFER_SIZE = 60

# Offset (in frames) applied to chosen_fi when selecting from hires_buf.
# Set negative if hires frames appear to lag lores detection.
HIRES_LAG_FRAMES = 0

# For long vehicles (hull width >= HIRES_LONG_VEHICLE_WIDTH lores px),
# apply this additional offset so the front hasn't yet exited the frame.
HIRES_LAG_LONG_FRAMES    = -1
HIRES_LONG_VEHICLE_WIDTH = 300  # lores px

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

def _save_and_transfer(frames, event_count):
    """
    frames: list of (hires_bgr, stem) to save as JPEGs.
    Runs in a daemon thread.
    """
    if shm_free_mb() < SHM_MIN_MB:
        log.warning("[save] /dev/shm low (%.1f MB) — event %d discarded",
                    shm_free_mb(), event_count)
        return
    SHM_BASE.mkdir(parents=True, exist_ok=True)
    saved = []
    for hires_bgr, stem in frames:
        fname = f"{stem}_{event_count}.jpg"
        path  = SHM_BASE / fname
        cropped = hires_bgr[HIRES_CROP_TOP:HIRES_CROP_BOTTOM, :]
        cv2.imwrite(str(path), cropped,
                    [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        saved.append(fname)
        log.info("[save] event %d  → %s", event_count, fname)
    start_transfer()


def start_save(frames, event_count):
    t = threading.Thread(target=_save_and_transfer,
                         args=(frames, event_count), daemon=True)
    t.start()

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

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

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
    # Rolling buffer of (hires_bgr, stem, meta) — most recent HIRES_BUFFER_SIZE frames
    hires_buf: collections.deque = collections.deque(maxlen=HIRES_BUFFER_SIZE)

    # Metadata ring buffer for delay compensation
    meta_buf: collections.deque = collections.deque(maxlen=METADATA_DELAY_FRAMES + 1)

    consecutive           = 0
    vehicle_active        = False
    lockout_remaining     = 0
    event_frames          = 0      # frames since vehicle_active became True
    centroid_history      = []     # (fi, cx, fully_interior)
    vehicle_frame_indices = []     # fi of every VEHICLE-tagged frame
    best_center_idx       = None   # global frame index
    best_center_dist      = float('inf')
    best_center_cx        = None
    max_blob_width        = 0      # max hull width seen during this event
    stuck_frames          = 0      # frames of continuous non-vehicle fg activity
    frame_count           = 0
    fi                    = 0      # detection frame index (increments every frame)

    # Load daily event counter — reset to 0 if date has changed since last run
    _saved_num, _saved_date = _load_event_num()
    today = datetime.now().strftime('%Y%m%d')
    event_count = _saved_num if _saved_date == today else 0
    log.info("[event_num] starting at event %d for %s", event_count + 1, today)

    SHM_BASE.mkdir(parents=True, exist_ok=True)

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
            blob        = None
            reject_reason = None
        else:
            blob, reject_reason = find_vehicle_blob(fg_mask)
            if reject_reason:
                log.info("  fi=%d  REJECTED  %s", fi, reject_reason)

        # --- Event logic -----------------------------------------------------
        if blob:
            consecutive += 1
            cx = blob['centroid_x']

            bx, by, bw, bh = blob['bbox']
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

            # Check centroid travel before logging or saving
            cx_vals = [cx for _, cx, _ in centroid_history]
            travel  = (max(cx_vals) - min(cx_vals)) if len(cx_vals) >= 2 else 0.0

            log.info("VEHICLE END  event=%d  best_fi=%s  pred=%.1f  src=%s  by=%s  travel=%.1f",
                     event_count + 1,
                     chosen_fi if chosen_fi is not None else '–',
                     pred if pred is not None else float('nan'),
                     src, chosen_by, travel)

            # Check centroid travel — stationary blobs are lighting artefacts, not vehicles.

            if travel < MIN_CENTROID_TRAVEL:
                log.warning("VEHICLE DISCARDED  travel=%.1f < %d — likely false trigger; "
                            "resetting background model", travel, MIN_CENTROID_TRAVEL)
                # Force MOG2 to rapidly re-learn the current frame as background
                for _ in range(20):
                    fgbg.apply(roi, learningRate=0.5)
                ev_baseline = ev   # re-anchor EV baseline to current exposure
                stuck_frames = 0

            elif chosen_fi is not None and hires_buf:
                # Pick two buffer entries whose fi is closest to chosen_fi,
                # adjusted for any hires/lores pipeline lag, with an extra
                # offset for long vehicles so the front hasn't yet exited.
                is_long = max_blob_width >= HIRES_LONG_VEHICLE_WIDTH
                lag = HIRES_LAG_FRAMES + (HIRES_LAG_LONG_FRAMES if is_long else 0)
                adjusted_fi = chosen_fi + lag
                buf_list = list(hires_buf)  # each entry: (bgr, stem, meta, fi)
                buf_list.sort(key=lambda e: abs(e[3] - adjusted_fi))
                to_save  = buf_list[:2]
                to_save.sort(key=lambda e: e[3])  # chronological order
                log.info("  saving fi=%s  (chosen_fi=%d  lag=%d  long=%s  max_w=%d)",
                         [e[3] for e in to_save], chosen_fi, lag,
                         is_long, max_blob_width)
                _, _last_date = _load_event_num()
                if _last_date != datetime.now().strftime('%Y%m%d'):
                    event_count = 0
                event_count += 1
                _save_event_num(event_count)
                frames_to_save = [(bgr, stem) for bgr, stem, _, _ in to_save]
                start_save(frames_to_save, event_count)
            else:
                log.warning("VEHICLE END  no best frame available — skipping save")

            # Reset state
            vehicle_active        = False
            consecutive           = 0
            event_frames          = 0
            centroid_history      = []
            vehicle_frame_indices = []
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

        if frame_count % PRINT_INTERVAL == 0:
            log.info("frame=%d  events=%d  free_shm=%.0f MB  lr=%.4f",
                     frame_count, event_count, shm_free_mb(), lr)

        fi += 1

    picam2.stop()
    log.info("Done. %d events captured, %d frames processed.",
             event_count, frame_count)


if __name__ == "__main__":
    main()
