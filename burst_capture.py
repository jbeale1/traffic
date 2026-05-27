#!/usr/bin/env python3
"""
burst_capture.py  —  Zone-tracked vehicle capture for Pi 5 
using IMX296 (Global Shutter) or IMX477 (HQ Camera)

Background mode: monitors NUM_ZONES horizontal zones of the lores YUV stream at 10 fps
using per-zone EMA background models.  When a vehicle enters the frame its centroid is
tracked across zones; as soon as the centroid crosses the horizontal midpoint the two
flanking frames (100 ms apart) are saved — giving a well-centred pair without manual
selection.  Zone models are frozen during an event and reanchored afterward.  An 8-second
timeout abandons stalled events and reinitialises all zone models.

Each event produces three files named:
  YYYYMMDD_HHMMSS_sss_NNN.jpg  — ISP-processed JPEG (4056x1600 after crop), deleted after transfer
                                 (lower res with other sensors)

Usage:
  python3 burst_capture.py [--shutter US] [--gain G] [--tune FILE]

Options:
  --shutter US    Fix exposure time in microseconds (default: auto)
  --gain G        Fix analogue gain (default: auto)
  --tune FILE     Path to libcamera tuning JSON (default: system IMX477 tuning)
  --threshold Z   Z-score threshold for zone motion detection (default: 4.0)
  --cooldown S    Seconds to ignore motion after an event (default: 0.8)
  --bursts N      Stop after N events (default: 0 = run forever)
  --fmt           Output image format: jpeg or tiff (default: jpeg)
  --shm-min-mb M  Minimum free /dev/shm space in MB before pausing capture (default: 50)
"""

import argparse
import math
import logging
import os
import queue
import shutil
import signal
import subprocess
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import cv2
from picamera2 import Picamera2
from libcamera import controls as libctrls


from config import REMOTE_HOST, REMOTE_DIR

VERSION = "1.56"  # use Pi global shutter camera
# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SAVE_LORES_DEBUG = False        # set True to save/transfer *_lores.jpg debug images
FRAME_RATE   = 20.0          # lores stream frame rate (fps)
LORES_SIZE   = (400, 300)      # lores stream size requested; actual buffer may be wider
#RAW_SIZE     = (4056, 2160)    # 4056x2160 sensor mode; ISP crops to this from full 4056x3040 frame
RAW_SIZE     = (1456, 1088)    # full frame size of Pi global shutter camera (IMX296)
CROP_OFFSET  = int(-130 * (1088/2160))            # vertical offset +moves image down in final (rotated) image
CROP_H       = int(1600 * (1088/2160))           # final image height after software crop of 2160-row raw frame
NUM_ZONES    = 12               # number of horizontal detection zones across lores frame
ZONE_W       = int(LORES_SIZE[0] / (NUM_ZONES - 1))  # width in pixels of each zone
EVENT_TIMEOUT_S = 8.0          # abandon event and rewarm if vehicle doesn't clear in this time
EVENT_MIN_TRAVEL = 50          # min centroid travel (lores px) to save even without midpoint crossing
SHM_DIR      = Path("/dev/shm/burst")
BG_ALPHA     = 0.02            # EMA coefficient for background model (slow adapt)
BG_WARMUP    = 60              # frames before motion detection is armed
SHM_MIN_MB   = 50              # default minimum free space in /dev/shm (MB)
EVENT_NUM_FILE = Path("/tmp/burst_capture_event_num.txt")  # persists daily event counter
PRINT_INTERVAL = 300           # frames between status prints 


# Background subtraction (applied to vehicle frames in worker)
BG_SUBTRACT       = False # True: zero background pixels in vehicle frames and save bg JPEG;
                          # False: save vehicle frames unmodified (mask PNGs still generated)
BG_DIFF_THRESHOLD = 25   # max per-channel absolute diff to call a pixel background
BG_OPEN_PX        = 5    # morphological open radius: removes isolated noise specks
BG_CLOSE_PX       = 25   # morphological close radius: bridges gaps before component filter
BG_CLOSE2_PX      = 25   # second close radius: fills near-threshold holes after component filter
BG_ROOF_POLY_DEG  = 4    # polynomial degree for roofline smoothing
BG_ROOF_TOP_FRAC  = 0.60 # roofline smoothing only active above this fraction of frame height
BG_ROOF_MAX_FILL  = 30   # max rows the roofline smoother may fill downward per column
BG_TOP_MEDIAN_W   = 51   # column window for median-filtering the top-edge profile (must be odd)
BG_TOP_SPIKE_TOL  = 3    # rows: top-edge pixels this far above the median profile are suppressed

BG_SHADOW_BOTTOM_FRAC = 0.10   # fraction of mask height to treat as the shadow zone
BG_SHADOW_DARK_MARGIN = 8      # shadow pixel must be darker than bg by at least this DN
BG_GROUND_MARGIN_PX = 8   # rows below the vehicle floor to tolerate (wheels dip here)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("burst")

# ---------------------------------------------------------------------------
# Daily event counter persistence
# ---------------------------------------------------------------------------

def _load_event_num() -> int:
    """Read today's last event number from the state file; return 0 if absent or stale."""
    try:
        text = EVENT_NUM_FILE.read_text().strip().split()
        date_str, num_str = text[0], text[1]
        if date_str == datetime.now().strftime("%Y%m%d"):
            return int(num_str)
    except Exception:
        pass
    return 0

def _save_event_num(event_num: int):
    """Write today's date and current event number to the state file."""
    try:
        EVENT_NUM_FILE.write_text(f"{datetime.now().strftime('%Y%m%d')} {event_num}\n")
    except Exception as e:
        log.warning("[event_num] failed to save state: %s", e)

# ---------------------------------------------------------------------------
# Timing: NTP sync + sensor-clock → wall-clock anchor
# ---------------------------------------------------------------------------

def _wait_for_time_sync(timeout_s: int = 120, poll_interval_s: int = 5) -> bool:
    """Block until systemd reports NTP-synchronized, or timeout."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            result = subprocess.run(
                ["timedatectl", "show", "--property=NTPSynchronized", "--value"],
                capture_output=True, text=True, timeout=5,
            )
            if result.stdout.strip() == "yes":
                log.info("[timing] NTP synchronized, anchoring wall clock")
                return True
        except Exception:
            pass
        log.info("[timing] Waiting for NTP sync…")
        time.sleep(poll_interval_s)
    log.warning("[timing] NTP sync timeout after %ds, using unsynchronized clock", timeout_s)
    return False


_wall_anchor: datetime | None = None
_sensor_anchor_ns: int | None = None


def sensor_ts_to_datetime(sensor_ts_ns: int | None) -> datetime:
    if sensor_ts_ns is not None and _wall_anchor is not None and _sensor_anchor_ns is not None:
        return _wall_anchor + timedelta(seconds=(sensor_ts_ns - _sensor_anchor_ns) / 1e9)
    return datetime.now()


def ts_stem(dt: datetime) -> str:
    ms = dt.microsecond // 1000
    return f"{dt.strftime('%Y%m%d_%H%M%S')}_{ms:03d}"

# ---------------------------------------------------------------------------
# /dev/shm free-space check
# ---------------------------------------------------------------------------

def shm_free_mb() -> float:
    st = os.statvfs("/dev/shm")
    return st.f_bavail * st.f_frsize / (1024 * 1024)

# ---------------------------------------------------------------------------
# Background model — per-column, per-stripe
# Tracks two signals per column:
#   mean  — average brightness (sensitive to object presence)
#   cv    — coefficient of variation (std/mean), illumination-invariant texture
# A cloud shadow scales both std and mean together, leaving CV nearly constant.
# A vehicle introduces a different reflectance pattern, changing CV as well as mean.
# Triggering requires BOTH signals to be anomalous in the same column, reducing
# false triggers from smooth illumination changes.
# ---------------------------------------------------------------------------

class StripeBackground:
    def __init__(self, width: int, alpha: float = BG_ALPHA):
        self.alpha  = alpha
        self.mean_m = None   # EMA of column means
        self.mad_m  = None   # EMA of |delta| for means
        self.mean_v = None   # EMA of column CV (std/mean)
        self.mad_v  = None   # EMA of |delta| for CV
        self.n      = 0
        self._reanchor_frames = 0

    def reset(self):
        """Fully reset the background model (clears EMA state, re-enters warmup)."""
        self.mean_m = None
        self.mad_m  = None
        self.mean_v = None
        self.mad_v  = None
        self.n      = 0
        self._reanchor_frames = 0

    def reanchor(self, frames: int = 10):
        """Call after a burst ends; forces fast re-adaptation for N frames."""
        self._reanchor_frames = frames

    def update(self, col_means: np.ndarray, col_cvs: np.ndarray):
        if self.mean_m is None:
            self.mean_m = col_means.astype(np.float32)
            self.mad_m  = np.ones_like(self.mean_m) * 8.0
            self.mean_v = col_cvs.astype(np.float32)
            self.mad_v  = np.ones_like(self.mean_v) * 0.02
        else:
            a = 0.5 if self._reanchor_frames > 0 else self.alpha
            if self._reanchor_frames > 0:
                self._reanchor_frames -= 1
            self.mad_m  = a * np.abs(col_means - self.mean_m) + (1 - a) * self.mad_m
            self.mean_m = a * col_means + (1 - a) * self.mean_m
            self.mad_v  = a * np.abs(col_cvs - self.mean_v)   + (1 - a) * self.mad_v
            self.mean_v = a * col_cvs  + (1 - a) * self.mean_v
        self.n += 1

    def z_scores(self, col_means: np.ndarray, col_cvs: np.ndarray):
        """Return (z_mean, z_cv) arrays for the given column statistics."""
        std_m = np.maximum(self.mad_m * 1.4826, 1.0)
        std_v = np.maximum(self.mad_v * 1.4826, 0.005)
        return (np.abs(col_means - self.mean_m) / std_m,
                np.abs(col_cvs   - self.mean_v)  / std_v)

    @property
    def ready(self) -> bool:
        return self.n >= BG_WARMUP


# ---------------------------------------------------------------------------
# Zone tracker — NUM_ZONES evenly-spaced horizontal zones across the lores frame.
# Each zone has its own StripeBackground model.  During a vehicle event all zone
# models are frozen; only unoccupied zones update during normal monitoring.
# ---------------------------------------------------------------------------

class ZoneTracker:
    """Tracks vehicle presence across NUM_ZONES horizontal zones.

    Each call to feed() returns:
      centroid  — float x-position (in lores pixels) of the midpoint between the
                  leftmost and rightmost occupied zone centres, or None if no zone
                  is occupied.
      occupied  — list of occupied zone indices (0 = leftmost)
      reason    — human-readable status string
    """

    def __init__(self, threshold: float, lores_valid_w: int,
                 num_zones: int = NUM_ZONES, zone_w: int = ZONE_W):
        self.threshold     = threshold
        self.lores_valid_w = lores_valid_w
        self.num_zones     = num_zones
        self.zone_w        = zone_w
        # Distribute zone start-x positions evenly across the valid frame width.
        span = lores_valid_w - zone_w
        self.zone_starts = [
            int(round(span * i / max(num_zones - 1, 1)))
            for i in range(num_zones)
        ]
        self.zone_centers = [x + zone_w // 2 for x in self.zone_starts]
        self.frame_center = lores_valid_w / 2.0
        self.bgs = [StripeBackground(zone_w) for _ in range(num_zones)]

    def rewarm(self):
        """Reset all zone models (re-enter warmup)."""
        for bg in self.bgs:
            bg.reset()

    def reanchor(self, frames: int = 4):
        """Fast re-adaptation after an event."""
        for bg in self.bgs:
            bg.reanchor(frames)

    @property
    def ready(self) -> bool:
        return all(bg.ready for bg in self.bgs)

    def _zone_stats(self, y_plane: np.ndarray, x_start: int):
        """Return (col_means, col_cvs) for one zone stripe (road band only)."""
        stripe = y_plane[100:200, x_start : x_start + self.zone_w]
        m = stripe.mean(axis=0)
        s = stripe.std(axis=0)
        cv = s / np.maximum(m, 1.0)
        return m, cv

    def feed(self, y_plane: np.ndarray, freeze: bool = False,
             anchor_zone: int | None = None
             ) -> tuple[float | None, list[int], str]:
        """Process one lores frame.

        Args:
            y_plane:     lores Y plane.
            freeze:      if True, skip background model updates (called during an event).
            anchor_zone: index of the zone that started the current event, or None
                         during monitoring.  When supplied, any occupied zone that is
                         separated from the contiguous block containing anchor_zone by
                         more than one quiet zone is treated as noise (e.g. a moving
                         tree shadow) and excluded from the occupied list and centroid.

        Returns:
            (centroid, occupied_zones, reason)
        """
        occupied = []
        max_z = 0.0

        for i, (x0, bg) in enumerate(zip(self.zone_starts, self.bgs)):
            m, cv = self._zone_stats(y_plane, x0)

            if not bg.ready:
                bg.update(m, cv)
                continue

            zm, zv = bg.z_scores(m, cv)
            hit = int(((zm > self.threshold) & (zv > self.threshold)).sum())
            max_z = max(max_z, zm.max(), zv.max())

            if hit >= 3:
                occupied.append(i)
            elif not freeze:
                bg.update(m, cv)

        if not self.ready:
            return None, [], "warming up"

        if not occupied:
            return None, [], f"clear  max_z={max_z:.1f}"

        # Spatial continuity filter: when an anchor zone is known, keep only
        # the contiguous block of occupied zones that includes the anchor.
        # Zones separated from that block by more than one quiet zone are
        # discarded as noise (moving shadows, etc.).
        if anchor_zone is not None and anchor_zone in occupied:
            occupied_set = set(occupied)
            # Grow outward from anchor_zone, allowing at most one quiet-zone gap.
            kept = {anchor_zone}
            for direction in (-1, +1):
                cursor = anchor_zone
                while True:
                    cursor += direction
                    if cursor < 0 or cursor >= self.num_zones:
                        break
                    if cursor in occupied_set:
                        kept.add(cursor)
                    else:
                        # one-zone gap: look one further before giving up
                        cursor += direction
                        if cursor < 0 or cursor >= self.num_zones:
                            break
                        if cursor in occupied_set:
                            kept.add(cursor)
                        else:
                            break   # two consecutive quiet zones — stop expanding
            noise = [z for z in occupied if z not in kept]
            occupied = sorted(kept)
            if noise:
                max_z_tag = f"  max_z={max_z:.1f}"
                noise_tag = f"  noise_zones={noise}"
            else:
                max_z_tag = f"  max_z={max_z:.1f}"
                noise_tag = ""
        else:
            max_z_tag = f"  max_z={max_z:.1f}"
            noise_tag = ""

        if not occupied:
            return None, [], f"clear (filtered){max_z_tag}"

        # Centroid = mean of all occupied zone centres (robust to a single
        # edge zone firing late, which would skew a leftmost/rightmost midpoint)
        centroid = float(np.mean([self.zone_centers[i] for i in occupied]))
        offset   = centroid - self.frame_center
        reason   = (f"zones={occupied}  centroid={centroid:.0f}"
                    f"  offset={offset:+.0f}{max_z_tag}{noise_tag}")
        return centroid, occupied, reason


# ---------------------------------------------------------------------------
# Conversion helpers
# ---------------------------------------------------------------------------

def load_frame(npy_path: Path) -> np.ndarray:
    """Load a saved frame and return as RGB uint8.
    Despite being labelled BGR888 in picamera2, the main stream array is
    actually RGB-ordered in memory — see https://forums.raspberrypi.com/viewtopic.php?t=397177"""
    return np.load(str(npy_path))

def rsync_file(local_path: Path) -> bool:
    global _rsync_proc
    proc = subprocess.Popen(
        ["rsync", "--remove-source-files", "-q",
         str(local_path), f"{REMOTE_HOST}:{REMOTE_DIR}"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    with _rsync_lock:
        _rsync_proc = proc
    returncode = proc.wait()
    with _rsync_lock:
        _rsync_proc = None
    if returncode != 0:
        _, stderr = proc.communicate()
        log.warning("rsync failed for %s: %s", local_path.name,
                    stderr.decode().strip())
        return False
    return True



def _build_exif(meta: dict):
    """Build a Pillow ExifData object from libcamera metadata.

    Tags that are SRATIONAL (signed rational) -- ShutterSpeedValue, BrightnessValue,
    CameraTemperature -- must go into the Exif sub-IFD (0x8769) via get_ifd(), not the
    top-level dict, because Pillow encodes top-level rationals as unsigned RATIONAL.
    """
    from PIL import Image
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


# ---------------------------------------------------------------------------
# Background worker: convert → transfer → delete npy
# ---------------------------------------------------------------------------

# Each item on the queue is a list of (npy_path, meta_dict) tuples for one burst.
_work_queue: queue.Queue = queue.Queue()
_stop_event = threading.Event()
_rsync_proc: subprocess.Popen | None = None
_rsync_lock = threading.Lock()
# Pairs of (npy_path, out_path) whose rsync failed; retried after next success.
_failed_transfers: list[tuple[Path, Path]] = []


def _load_and_crop(npy_path: Path) -> np.ndarray:
    """Load a raw frame, rotate 180°, and apply the standard vertical crop."""
    rgb = load_frame(npy_path)
    rgb = np.rot90(rgb, 2)
    margin = CROP_OFFSET + (rgb.shape[0] - CROP_H) // 2
    return rgb[margin : margin + CROP_H, :]


def _build_fg_mask(bg_rgb: np.ndarray, car_rgb: np.ndarray) -> np.ndarray:
    """Return a uint8 mask (255=foreground, 0=background) for car_rgb given bg_rgb.

    Pipeline:
      1. Per-channel absolute difference, max across channels.
      2. Threshold → binary mask.
      3. Morphological open  — removes small isolated noise specks.
      4. Morphological close — bridges narrow gaps within the vehicle silhouette.
      5. Connected-component filter — keeps only the largest component, discarding
         disconnected blobs; done before hole-fill to prevent border flood leakage.
      6. Second morphological close (larger kernel) — fills near-threshold holes and
         concavities (e.g. bumper recesses) that are physically adjacent to solid
         foreground; safe to use a larger kernel here since spurious blobs are gone.
      7. Polynomial roofline smoothing — fits a polynomial to the topmost white pixel
         per column within the mask's horizontal extent and the top BG_ROOF_TOP_FRAC
         of the frame; fills gaps between the smooth fit and the first actual white
         pixel, eliminating small concave bites on the upper vehicle silhouette.
      8. Flood-fill from border — fills any remaining fully-enclosed interior holes
         without convexifying the outline.
    """
    # diff = np.abs(car_rgb.astype(np.int16) - bg_rgb.astype(np.int16))
    # mask = (diff.max(axis=2) > BG_DIFF_THRESHOLD).astype(np.uint8) * 255
    gray     = cv2.cvtColor(car_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    lap      = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    tex      = cv2.GaussianBlur(lap * lap, (21, 21), 0)
    tex_norm = tex / (np.percentile(tex, 95) + 1e-6)   # robust normalisation
    tex_norm = np.clip(tex_norm, 0.0, 1.0)

    diff         = np.abs(car_rgb.astype(np.int16) - bg_rgb.astype(np.int16))
    diff_max     = diff.max(axis=2).astype(np.float32)
    # Raise threshold in high-texture (background-like) regions, lower it in smooth (vehicle-like) regions
    adaptive_thr = BG_DIFF_THRESHOLD * (0.5 + tex_norm)
    mask         = (diff_max > adaptive_thr).astype(np.uint8) * 255

    k_open   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                         (BG_OPEN_PX   * 2 + 1, BG_OPEN_PX   * 2 + 1))
    k_close  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                         (BG_CLOSE_PX  * 2 + 1, BG_CLOSE_PX  * 2 + 1))
    k_close2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                         (BG_CLOSE2_PX * 2 + 1, BG_CLOSE2_PX * 2 + 1))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k_open)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close)

    # Component filter first: prevents hole-fill from leaking via border-touching blobs
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n_labels > 1:
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        mask = np.where(labels == largest, np.uint8(255), np.uint8(0))

    # Second close on the clean single-component mask
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close2)

    # Ground-line clamp: suppress shadow/wake pixels below the vehicle floor.
    # Find the lowest row that has foreground within the core horizontal body span,
    # then remove any foreground pixels that are (a) below that row AND (b) outside
    # the central horizontal extent of the full mask, OR darker than the bg.
    # This removes trailing/leading ground shadows that survive the morphological steps.
    h, w = mask.shape
    fg_ys2, fg_xs2 = np.where(mask > 127)
    if len(fg_ys2) > 0:
        x_min2, x_max2 = int(fg_xs2.min()), int(fg_xs2.max())
        core_x_lo = x_min2 + (x_max2 - x_min2) // 6
        core_x_hi = x_max2 - (x_max2 - x_min2) // 6
        # Ground row = lowest fg row within the core horizontal span
        core_cols = (fg_xs2 >= core_x_lo) & (fg_xs2 <= core_x_hi)
        if core_cols.any():
            ground_row = int(fg_ys2[core_cols].max())
            floor_limit = ground_row + BG_GROUND_MARGIN_PX
            # Suppress fg pixels below the floor that are also outside the full body x-span
            # or darker than bg (i.e. shadow / wake, not a tyre)
            for r in range(floor_limit, h):
                fg_in_row = np.where(mask[r, :] > 127)[0]
                if len(fg_in_row) == 0:
                    continue
                car_lum = car_rgb[r, fg_in_row].mean(axis=1).astype(np.float32)
                bg_lum  =  bg_rgb[r, fg_in_row].mean(axis=1).astype(np.float32)
                # Remove if darker than bg (shadow) regardless of x position
                shadow = car_lum < bg_lum - BG_SHADOW_DARK_MARGIN
                mask[r, fg_in_row[shadow]] = 0
                # Also remove anything laterally outside the full body span even if not darker
                outside_body = (fg_in_row < x_min2) | (fg_in_row > x_max2)
                mask[r, fg_in_row[outside_body]] = 0


    # Roofline smoothing: fit a polynomial to the topmost white pixel per column
    # within the mask's horizontal extent and upper BG_ROOF_TOP_FRAC of the frame,
    # then fill downward from the smooth curve to the first actual white pixel.
    # Only adds pixels; never removes existing foreground.
    h, w = mask.shape
    top_limit = int(h * BG_ROOF_TOP_FRAC)
    fg_ys, fg_xs = np.where(mask > 127)
    if len(fg_xs) > 0:
        x_min, x_max = int(fg_xs.min()), int(fg_xs.max())
        col_range = np.arange(x_min, x_max + 1)
        top_row = np.full(len(col_range), np.nan)
        for i, col in enumerate(col_range):
            col_hits = np.where(mask[:top_limit, col] > 127)[0]
            if len(col_hits) > 0:
                top_row[i] = col_hits.min()
        valid = ~np.isnan(top_row)
        if valid.sum() >= BG_ROOF_POLY_DEG + 1:
            fit_x  = col_range[valid].astype(np.float64)
            fit_y  = top_row[valid]
            x_mid  = fit_x.mean()
            coeffs = np.polyfit(fit_x - x_mid, fit_y, BG_ROOF_POLY_DEG)
            poly   = np.poly1d(coeffs)
            smoothed = np.clip(poly(col_range - x_mid), 0, top_limit).astype(int)
            for i, col in enumerate(col_range):
                col_hits = np.where(mask[:top_limit, col] > 127)[0]
                if len(col_hits) == 0:
                    continue
                first_white = col_hits.min()
                roof_row = smoothed[i]
                if roof_row < first_white and (first_white - roof_row) <= BG_ROOF_MAX_FILL:
                    mask[roof_row:first_white, col] = 255

    # Top-edge spike suppression: median-filter the topmost-white-pixel profile and
    # zero any foreground pixels that protrude more than BG_TOP_SPIKE_TOL rows above
    # the smoothed profile.  Purely subtractive — never adds pixels.
    fg_ys3, fg_xs3 = np.where(mask > 127)
    if len(fg_xs3) > 0:
        x_min3, x_max3 = int(fg_xs3.min()), int(fg_xs3.max())
        col_range3 = np.arange(x_min3, x_max3 + 1)
        top_edge = np.full(len(col_range3), np.nan)
        for i, col in enumerate(col_range3):
            hits = np.where(mask[:, col] > 127)[0]
            if len(hits):
                top_edge[i] = hits.min()
        valid3 = ~np.isnan(top_edge)
        if valid3.sum() > BG_TOP_MEDIAN_W:
            filled = top_edge.copy()
            last = np.nan
            for i in range(len(filled)):
                if not np.isnan(filled[i]):
                    last = filled[i]
                elif not np.isnan(last):
                    filled[i] = last
            last = np.nan
            for i in range(len(filled) - 1, -1, -1):
                if not np.isnan(filled[i]):
                    last = filled[i]
                elif not np.isnan(last):
                    filled[i] = last
            half = BG_TOP_MEDIAN_W // 2
            padded = np.pad(filled, half, mode='edge')
            smoothed3 = np.array([np.median(padded[i:i + BG_TOP_MEDIAN_W])
                                  for i in range(len(filled))])
            for i, col in enumerate(col_range3):
                if np.isnan(top_edge[i]):
                    continue
                ref_row = int(smoothed3[i])
                actual_top = int(top_edge[i])
                # Spike suppression: white pixels above the smooth profile
                spike_limit = ref_row - BG_TOP_SPIKE_TOL
                if spike_limit > 0 and actual_top < spike_limit:
                    mask[actual_top:spike_limit, col] = 0
                    actual_top = spike_limit
                # Gap fill: black notch between smooth profile and first white pixel
                if actual_top > ref_row + BG_TOP_SPIKE_TOL:
                    fill_to = min(actual_top, ref_row + BG_ROOF_MAX_FILL)
                    mask[ref_row:fill_to, col] = 255

    # Flood-fill exterior, then OR enclosed holes into mask
    h, w  = mask.shape
    inv   = cv2.bitwise_not(mask)
    bordered = cv2.copyMakeBorder(inv, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=255)
    cv2.floodFill(bordered, None, (0, 0), 0)
    holes = bordered[1:h + 1, 1:w + 1]

    # Shadow removal: any foreground pixels in the bottom BG_SHADOW_BOTTOM_FRAC of the
    # mask's vertical extent that extend beyond the horizontal bounds of the upper body
    # are removed, provided they are also darker than the background.
    fg_rows = np.where(mask.max(axis=1) > 127)[0]
    if len(fg_rows) > 0:
        mask_top    = int(fg_rows.min())
        mask_bottom = int(fg_rows.max())
        mask_h      = mask_bottom - mask_top + 1

        split_row = mask_bottom - int(mask_h * BG_SHADOW_BOTTOM_FRAC)

        # Horizontal extent of the upper body portion
        upper_cols = np.where(mask[mask_top : split_row, :].max(axis=0) > 127)[0]
        if len(upper_cols) >= 2:
            body_x_min = int(upper_cols.min())
            body_x_max = int(upper_cols.max())

            # In the bottom strip, zero pixels outside the body extent that are darker than bg
            for r in range(split_row, mask_bottom + 1):
                outside = np.where(
                    (mask[r, :] > 127) &
                    (np.arange(w) < body_x_min) | 
                    ((mask[r, :] > 127) & (np.arange(w) > body_x_max))
                )[0]
                if len(outside) == 0:
                    continue
                car_lum = car_rgb[r, outside].mean(axis=1).astype(np.float32)
                bg_lum  =  bg_rgb[r, outside].mean(axis=1).astype(np.float32)
                dark    = car_lum < bg_lum - BG_SHADOW_DARK_MARGIN
                mask[r, outside[dark]] = 0

    return cv2.bitwise_or(mask, holes)


def _worker(fmt: str):
    """Drain the work queue: convert each frame, rsync, delete originals.

    Each job is a list of (npy_path, meta, event_num) tuples where:
      index 0  — background frame: saved as-is (JPEG/TIFF), no subtraction.
      index 1+ — vehicle frames: background-subtracted before encoding; pixels
                 outside the foreground mask are set to black.
    If the job has fewer than 2 frames (shm was low at capture time) the
    vehicle frames are saved without subtraction as a fallback.
    """

    while True:
        try:
            item = _work_queue.get(timeout=0.2)
        except queue.Empty:
            if _stop_event.is_set():
                break
            continue
        if item is None:          # sentinel → exit
            break
        if _stop_event.is_set():
            _work_queue.task_done()
            break

        frames = item   # list of (npy_path, meta, event_num)
        converted: list[Path] = []

        # Load and crop the background frame first (index 0); keep it in memory
        # so it can be reused for each vehicle frame subtraction.
        bg_rgb: np.ndarray | None = None
        if len(frames) >= 1:
            bg_npy_path = frames[0][0]
            try:
                bg_rgb = _load_and_crop(bg_npy_path)
            except Exception as e:
                log.error("[worker] failed to load background frame %s: %s",
                          bg_npy_path.name, e)

        for idx, (npy_path, meta, event_num) in enumerate(frames):
            try:
                rgb = _load_and_crop(npy_path)

                if idx == 0:
                    # Background frame — save only when BG_SUBTRACT is enabled
                    if not BG_SUBTRACT:
                        npy_path.unlink(missing_ok=True)
                        continue
                    out_stem = f"{npy_path.stem}_{event_num}_bg"
                    mask = None
                else:
                    # Vehicle frame — build mask always (needed for PNG); apply
                    # pixel zeroing only when BG_SUBTRACT is enabled.
                    if bg_rgb is not None:
                        t0_mask = time.monotonic()
                        mask = _build_fg_mask(bg_rgb, rgb)
                        if BG_SUBTRACT:
                            rgb = rgb.copy()
                            rgb[mask == 0] = 0
                        log.info("[worker] mask frame %d  mask=%.1f%%  t=%.2fs",
                                 idx, (mask > 0).mean() * 100, time.monotonic() - t0_mask)
                    else:
                        log.warning("[worker] no background available for frame %d "
                                    "— saving without subtraction", idx)
                        mask = None
                    out_stem = f"{npy_path.stem}_{event_num}"

                if fmt == "tiff":
                    import imageio
                    out_path = npy_path.with_name(f"{out_stem}.tiff")
                    imageio.imwrite(str(out_path), rgb)
                else:
                    from PIL import Image as PilImage
                    out_path = npy_path.with_name(f"{out_stem}.jpg")
                    pil_img = PilImage.fromarray(rgb)
                    pil_img.save(str(out_path), format="JPEG", quality=92,
                                 exif=_build_exif(meta))

                log.info("[worker] converted %s → %s", npy_path.name, out_path.name)
                converted.append((npy_path, out_path))

                # Save mask as 8-bit grayscale PNG (white=foreground, black=background)
                if mask is not None:
                    mask_path = npy_path.with_name(f"{out_stem}_mask.png")
                    PilImage.fromarray(mask, mode="L").save(str(mask_path), format="PNG")
                    log.info("[worker] saved mask %s", mask_path.name)
                    converted.append((None, mask_path))  # None: no npy to clean up
            except Exception as e:
                log.error("[worker] conversion failed for %s: %s", npy_path.name, e)

        for npy_path, out_path in converted:
            ok = rsync_file(out_path)
            if ok:
                log.info("[worker] transferred %s", out_path.name)
                if npy_path is not None:
                    npy_path.unlink(missing_ok=True)
                if SAVE_LORES_DEBUG:
                    lores_debug = npy_path.with_name(npy_path.stem + "_lores.jpg")
                    if lores_debug.exists():
                        rsync_file(lores_debug)   # transfers and removes
                    lores_pre_debug = npy_path.with_name(npy_path.stem + "_lores_pre.jpg")
                    if lores_pre_debug.exists():
                        rsync_file(lores_pre_debug)   # transfers and removes
                # Flush any previously failed transfers now that connectivity is back.
                if _failed_transfers:
                    log.info("[worker] retrying %d previously failed transfer(s)", len(_failed_transfers))
                    still_failed: list[tuple[Path | None, Path]] = []
                    for f_npy, f_out in list(_failed_transfers):
                        if not f_out.exists():
                            log.warning("[worker] retry skipped %s — file no longer exists", f_out.name)
                            continue
                        if rsync_file(f_out):
                            log.info("[worker] retry transferred %s", f_out.name)
                            if f_npy is not None:
                                f_npy.unlink(missing_ok=True)
                        else:
                            log.warning("[worker] retry still failed for %s", f_out.name)
                            still_failed.append((f_npy, f_out))
                    _failed_transfers.clear()
                    _failed_transfers.extend(still_failed)
            else:
                npy_label = npy_path.name if npy_path is not None else "(no npy)"
                log.warning("[worker] transfer failed; keeping %s and %s",
                            npy_label, out_path.name)
                _failed_transfers.append((npy_path, out_path))

        _work_queue.task_done()


# ---------------------------------------------------------------------------
# Burst saver — enqueues conversion job
# ---------------------------------------------------------------------------

class BurstSaver:
    def __init__(self):
        self.event_num = _load_event_num()
        if self.event_num:
            log.info("[event_num] resuming from event %d (today)", self.event_num)
        else:
            log.info("[event_num] starting fresh event numbering for today")

    def save(self, frames: list[tuple], shm_min_mb: float) -> int:
        """Enqueue pre-saved (npy_path, meta) tuples for conversion.
        Frame arrays are written at capture time (while the request is live);
        this method handles queuing only. Space was already checked at write time."""
        self.event_num += 1
        _save_event_num(self.event_num)
        SHM_DIR.mkdir(parents=True, exist_ok=True)

        # Enqueue all frames as a single job so they are converted and transferred together
        _work_queue.put([(npy_path, meta, self.event_num) for npy_path, meta in frames])
        return len(frames)

# ---------------------------------------------------------------------------
# Startup flush: process any leftover .npy frames from a previous run
# ---------------------------------------------------------------------------

def _flush_leftover_frames(fmt: str):
    """Delete any stale .npy/.jpg frames left in SHM_DIR from a previous run."""
    if not SHM_DIR.exists():
        return
    leftover = sorted(SHM_DIR.glob("*.npy")) + sorted(SHM_DIR.glob("*.jpg"))
    if not leftover:
        return
    log.info("[flush] found %d leftover file(s) in %s — removing", len(leftover), SHM_DIR)
    for path in leftover:
        log.info("[flush] removing %s", path.name)
        path.unlink(missing_ok=True)
    log.info("[flush] done")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--shutter",    type=int,   default=None,  help="Exposure time µs")
    p.add_argument("--gain",       type=float, default=None,  help="Analogue gain")
    p.add_argument("--tune", type=str,
                   default=str(Path(__file__).parent / "imx477_pisp.json"),
                   help="Tuning JSON path")
    p.add_argument("--threshold",  type=float, default=4.0,   help="Z-score threshold")
    p.add_argument("--cooldown",   type=float, default=0.8,   help="Post-burst cooldown s")
    p.add_argument("--bursts",     type=int,   default=0,     help="Stop after N events (0=∞)")
    p.add_argument("--fmt",        default="jpeg", choices=["jpeg", "tiff"],
                   help="Output image format (default: jpeg)")
    p.add_argument("--shm-min-mb", type=float, default=SHM_MIN_MB,
                   help="Minimum free /dev/shm MB before pausing capture (default: 50)")
    p.add_argument("--ev",         type=float, default=-1.0,
                   help="Exposure value bias in stops; negative = darker (default: 0.0)")
    return p.parse_args()


def main():
    print("Burst Capture 2026/5/19 version %s" % VERSION)
    args = parse_args()

    # --- Start background worker thread (lower OS priority) -----------------
    worker_thread = threading.Thread(target=_worker, args=(args.fmt,), daemon=True)
    worker_thread.start()
    # Nice the worker so capture loop gets CPU priority
    try:
        os.setpriority(os.PRIO_PROCESS, worker_thread.ident, 10)
    except Exception:
        pass   # best-effort; not fatal

    # --- Flush any DNGs left over from a previous run -----------------------
    _flush_leftover_frames(args.fmt) 

    # --- Camera init --------------------------------------------------------
    kwargs = {}
    if args.tune:
        kwargs["tuning"] = Picamera2.load_tuning_file(args.tune)

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

    # raw          = {"size": RAW_SIZE,   "format": "SRGGB12_CSI2P"},  # for IMX477 sensor
    config = picam2.create_video_configuration(
        main         = {"size": RAW_SIZE,   "format": "BGR888"},
        lores        = {"size": LORES_SIZE, "format": "YUV420"},
        raw          = {"size": RAW_SIZE, "format": "SRGGB10_CSI2P"},
        controls     = cam_controls,
        buffer_count = 6,   # small fixed pool; frames are saved to shm immediately
    )
    picam2.configure(config)

    cam_config = picam2.camera_configuration()
    lores_valid_w, lores_valid_h = cam_config["lores"]["size"]
    log.info("Lores valid image size: %dx%d", lores_valid_w, lores_valid_h)

    # --- State --------------------------------------------------------------
    tracker        = ZoneTracker(args.threshold, lores_valid_w)
    saver          = BurstSaver()
    # Rolling 3-frame pre-event buffer: each entry is (npy_path, meta) saved to /dev/shm
    prev_prev_frame: tuple | None = None   # (npy_path, meta) two frames ago
    prev_frame     : tuple | None = None   # (npy_path, meta) one frame ago
    pre_event_frame: tuple | None = None   # saved background frame (2 frames before EVENT START)
    # Event tracking
    in_event       = False
    event_frame_count = 0          # frames since first trigger this event
    event_start_time  = 0.0        # monotonic time of first trigger
    prev_centroid       : float | None = None   # centroid from previous frame
    event_start_centroid: float | None = None   # centroid at EVENT START (for travel check)
    event_anchor_zone   : int   | None = None   # first occupied zone at EVENT START
    cooldown_until = 0.0
    event_count    = 0
    frame_count    = 0
    running        = True

    def handle_sigint(sig, frame):
        nonlocal running
        log.info("Interrupted — stopping.")
        running = False
        _stop_event.set()
        with _rsync_lock:
            if _rsync_proc is not None:
                _rsync_proc.terminate()

    signal.signal(signal.SIGINT, handle_sigint)

    # --- Run ----------------------------------------------------------------
    picam2.start()

    _wait_for_time_sync()
    global _wall_anchor, _sensor_anchor_ns
    _anchor_meta      = picam2.capture_metadata()
    _wall_anchor      = datetime.now()
    _sensor_anchor_ns = _anchor_meta.get("SensorTimestamp")
    if _sensor_anchor_ns is not None:
        log.info("[timing] Sensor clock anchored to wall clock")
    else:
        log.warning("[timing] SensorTimestamp unavailable — filenames will use datetime.now()")

    log.info("Camera started.  Warming up background model (%d frames)…", BG_WARMUP)

    t0 = time.monotonic()
    prev_log_time = t0

    while running:
        if args.bursts and event_count >= args.bursts:
            log.info("Reached --bursts limit (%d). Stopping.", args.bursts)
            break

        request = picam2.capture_request()
        try:
            now = time.monotonic()
            frame_count += 1

            lores   = request.make_array("lores")
            y_plane = lores[:lores_valid_h, :lores_valid_w]

            # Always save this frame to shm so it is available as prev_frame next iteration.
            # We write it speculatively and delete it if we don't end up needing it.
            meta         = request.get_metadata()
            sensor_ts_ns = meta.get("SensorTimestamp")
            dt           = sensor_ts_to_datetime(sensor_ts_ns)
            stem         = ts_stem(dt)
            SHM_DIR.mkdir(parents=True, exist_ok=True)
            cur_npy = SHM_DIR / f"{stem}.npy"

            free_mb = shm_free_mb()
            if free_mb >= args.shm_min_mb:
                np.save(str(cur_npy), request.make_array("main"))
                cur_frame = (cur_npy, meta)
            else:
                cur_frame = None   # can't save; treat as dropped frame

        finally:
            # Release the camera request immediately — pixel data is in shm now
            request.release()

        # --- Zone / centroid tracking (no camera request held) ---------------

        if now < cooldown_until:
            # During cooldown: update zone models, discard saved frame
            tracker.feed(y_plane, freeze=False)
            if cur_frame:
                cur_frame[0].unlink(missing_ok=True)
            if pre_event_frame:
                pre_event_frame[0].unlink(missing_ok=True)
                pre_event_frame = None
            prev_frame = None
            prev_prev_frame = None
            prev_centroid = None
            continue

        centroid, occupied, reason = tracker.feed(y_plane, freeze=in_event,
                                                   anchor_zone=event_anchor_zone if in_event else None)

        if not in_event:
            if occupied:
                # First frame of a new event; cur_frame becomes the first event frame.
                # prev_prev_frame (2 frames before event) is the cleanest background —
                # prev_frame may already contain a partially-visible object.
                pre_event_frame   = prev_prev_frame  # 2 frames before event start
                if prev_frame and prev_frame is not prev_prev_frame:
                    prev_frame[0].unlink(missing_ok=True)
                prev_prev_frame   = None
                in_event          = True
                event_frame_count = 1
                event_start_time  = now
                prev_frame        = cur_frame
                prev_centroid     = centroid
                event_start_centroid = centroid
                event_anchor_zone    = occupied[0]
                log.info("EVENT START  frame=%d  %s", frame_count, reason)
            else:
                # Monitoring: roll the 2-frame history forward.
                # prev_prev_frame is discarded; prev_frame becomes prev_prev_frame;
                # cur_frame becomes prev_frame.
                if prev_prev_frame:
                    prev_prev_frame[0].unlink(missing_ok=True)
                prev_prev_frame = prev_frame
                prev_frame    = cur_frame
                prev_centroid = None
                if frame_count % PRINT_INTERVAL == 0: # show status every this many frames
                    now_log = time.monotonic()
                    fps = PRINT_INTERVAL / (now_log - prev_log_time) if frame_count > 10 else 0
                    prev_log_time = now_log
                    log.info("monitoring  frame=%d  fps=%.1f  %s",
                             frame_count, fps, reason)

        else:
            # --- Active event ------------------------------------------------
            event_frame_count += 1
            event_duration = now - event_start_time

            # Timeout failsafe: abandon event and reinitialise all zone models
            if event_duration > EVENT_TIMEOUT_S:
                log.warning("[event] timeout after %.1f s — reinitialising zones", event_duration)
                if cur_frame:
                    cur_frame[0].unlink(missing_ok=True)
                if prev_frame:
                    prev_frame[0].unlink(missing_ok=True)
                if pre_event_frame:
                    pre_event_frame[0].unlink(missing_ok=True)
                    pre_event_frame = None
                tracker.rewarm()
                in_event          = False
                event_frame_count = 0
                prev_frame        = None
                prev_prev_frame   = None
                prev_centroid     = None
                event_start_centroid = None
                event_anchor_zone    = None
                cooldown_until    = now + args.cooldown
                continue

            log.info("  event frame %d  %s", event_frame_count, reason)

            if not occupied:
                # Vehicle has cleared all zones.
                # Fall back to saving if it travelled far enough laterally,
                # even if it never crossed the frame midpoint.
                travel = (abs(prev_centroid - event_start_centroid)
                          if prev_centroid is not None and event_start_centroid is not None
                          else 0.0)
                if (event_frame_count >= 3 and travel >= EVENT_MIN_TRAVEL
                        and prev_frame and cur_frame):
                    had_pre = pre_event_frame is not None
                    frames_to_save = ([pre_event_frame] if pre_event_frame else []) + [prev_frame, cur_frame]
                    n = saver.save(frames_to_save, args.shm_min_mb)
                    pre_event_frame = None
                    event_count += 1
                    log.info("SAVE (travel=%.0fpx)  event=%d  pre_event=%s",
                             travel, saver.event_num, "yes" if had_pre else "no")
                    log.info("Event %d queued for conversion (%d frames)", event_count, n)
                else:
                    log.info("EVENT END (no save — centroid did not cross midpoint, travel=%.0fpx)", travel)
                    if cur_frame:
                        cur_frame[0].unlink(missing_ok=True)
                    if prev_frame:
                        prev_frame[0].unlink(missing_ok=True)
                    if pre_event_frame:
                        pre_event_frame[0].unlink(missing_ok=True)
                        pre_event_frame = None
                tracker.reanchor(4)
                in_event          = False
                event_frame_count = 0
                prev_frame        = None
                prev_prev_frame   = None
                prev_centroid     = None
                event_start_centroid = None
                event_anchor_zone    = None
                cooldown_until    = now + args.cooldown
                continue

            # Check for midpoint crossing:
            #   - must be at least the 3rd event frame (filter pedestrians / short blips)
            #   - centroid must have moved past the frame midpoint this frame
            frame_center = tracker.frame_center
            crossed = (centroid is not None and prev_centroid is not None
                       and event_frame_count >= 3
                       and ((prev_centroid < frame_center <= centroid)   # left-to-right
                            or (prev_centroid > frame_center >= centroid)))  # right-to-left

            if crossed:
                if prev_frame and cur_frame:
                    had_pre = pre_event_frame is not None
                    frames_to_save = ([pre_event_frame] if pre_event_frame else []) + [prev_frame, cur_frame]
                    n = saver.save(frames_to_save, args.shm_min_mb)
                    pre_event_frame = None
                    event_count += 1
                    log.info("SAVE  event=%d  pre_event=%s  prev_offset=%+.0f  cur_offset=%+.0f",
                             saver.event_num,
                             "yes" if had_pre else "no",
                             prev_centroid - frame_center,
                             centroid - frame_center)
                    log.info("Event %d queued for conversion (%d frames)", event_count, n)
                else:
                    log.warning("[event] midpoint crossed but frame(s) missing (shm low?); skipping save")
                    if cur_frame:
                        cur_frame[0].unlink(missing_ok=True)
                    if prev_frame:
                        prev_frame[0].unlink(missing_ok=True)
                    if pre_event_frame:
                        pre_event_frame[0].unlink(missing_ok=True)
                        pre_event_frame = None

                if SAVE_LORES_DEBUG:
                    try:
                        from PIL import Image as _PilImage
                        _lores_path = SHM_DIR / f"{stem}_lores.jpg"
                        _PilImage.fromarray(y_plane).rotate(180).save(str(_lores_path), format="JPEG", quality=85)
                        log.info("[debug] lores snapshot saved to %s", _lores_path.name)
                    except Exception as _e:
                        log.warning("[debug] lores snapshot failed: %s", _e)

                tracker.reanchor(4)
                in_event          = False
                event_frame_count = 0
                prev_frame        = None
                prev_prev_frame   = None
                pre_event_frame   = None
                prev_centroid     = None
                event_start_centroid = None
                event_anchor_zone    = None
                cooldown_until    = now + args.cooldown
                continue

            # Centroid hasn't crossed yet — keep rolling
            # Discard the frame before prev_frame (no longer needed)
            if prev_frame and prev_frame[0] != (cur_frame[0] if cur_frame else None):
                prev_frame[0].unlink(missing_ok=True)
            prev_frame    = cur_frame
            prev_centroid = centroid

    picam2.stop()

    # Signal worker to exit; if interrupted, stop_event is already set and
    # the worker will bail out without draining remaining jobs.
    log.info("Waiting for worker thread to exit…")
    _work_queue.put(None)
    worker_thread.join()
    log.info("Done. %d events captured, %d frames processed.", event_count, frame_count)


if __name__ == "__main__":
    main()
