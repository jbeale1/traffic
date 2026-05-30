#!/usr/bin/env python3
"""
burst_capture.py  —  Zone-tracked vehicle capture for Pi 5 
using IMX296 (Global Shutter) or IMX477 (HQ Camera)

Background mode: monitors NUM_ZONES horizontal zones of the lores YUV stream at 10 fps
using per-zone EMA background models.  When a vehicle enters the frame its centroid is
tracked across zones; as soon as the centroid crosses the horizontal midpoint the two
flanking frames (50 or 100 ms apart) are saved — giving a well-centred pair without manual
selection.  For oversized vehicles that fill ≥ NUM_ZONES-1 zones simultaneously, two pairs
are saved instead: one when the vehicle first fills the frame, and one when it starts to
leave.  Zone models are frozen during an event and reanchored afterward.  An 8-second
timeout abandons stalled events and reinitialises all zone models.

Each event produces three files named:
  YYYYMMDD_HHMMSS_sss_NNN.jpg  — ISP-processed JPEG (HQ cam: 4056x1600 after crop), deleted after transfer
                                 (note: lower res with IMX296 sensor)

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

  note: camera is assumed upside-down. Image is rotated 180 before saving.
"""

import argparse
import collections
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

VERSION = "1.994"  # split max_z into zm/zv in log; add zv_zones to reason string
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
NUM_ZONES    = 20               # number of horizontal detection zones across lores frame
ZONE_W       = int(LORES_SIZE[0] / (NUM_ZONES - 1))  # width in pixels of each zone
EVENT_TIMEOUT_S = 8.0          # abandon event and rewarm if vehicle doesn't clear in this time
EVENT_MIN_TRAVEL = 75          # min centroid travel (lores px) to save even without midpoint crossing

ZV_HIT_THRESHOLD   = 10.0               # per-zone zv score to count a zone as vehicle-occupied (sizing/centroid)
ZV_OVERSIZED_ZONES = int(NUM_ZONES * 0.92)  # zv-hot zone count that classifies a vehicle as oversized

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

def _load_event_num() -> tuple[int, str]:
    """Read the last event number and its date from the state file.

    Returns (event_num, date_str) where date_str is 'YYYYMMDD' of the last saved
    event, or (0, '') if the file is absent or unreadable.  The caller is
    responsible for deciding whether to reset the counter based on the date.
    """
    try:
        text = EVENT_NUM_FILE.read_text().strip().split()
        date_str, num_str = text[0], text[1]
        return int(num_str), date_str
    except Exception:
        pass
    return 0, ""

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
            self.mean_m   = col_means.astype(np.float32)
            self.mad_m    = np.ones_like(self.mean_m) * 8.0
            self.mean_v   = col_cvs.astype(np.float32)
            self.mad_v    = np.ones_like(self.mean_v) * 0.02
        else:
            a = 0.5 if self._reanchor_frames > 0 else self.alpha
            if self._reanchor_frames > 0:
                self._reanchor_frames -= 1
            self.mad_m    = a * np.abs(col_means - self.mean_m) + (1 - a) * self.mad_m
            self.mean_m   = a * col_means + (1 - a) * self.mean_m
            self.mad_v    = a * np.abs(col_cvs - self.mean_v)   + (1 - a) * self.mad_v
            self.mean_v   = a * col_cvs  + (1 - a) * self.mean_v
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
        """Return (col_means, col_cvs) for one zone stripe (upper body band only).

        Stripe covers lores rows 40-130 (upper ~27-43% of frame height), targeting
        vehicle roof/upper body panels.  Keeping the stripe well above the road
        avoids contamination from low-sun road shadows.
        """
        stripe = y_plane[40:130, x_start : x_start + self.zone_w]
        m = stripe.mean(axis=0)
        s = stripe.std(axis=0)
        cv = s / np.maximum(m, 1.0)
        return m, cv

    def feed(self, y_plane, freeze=False, anchor_zone=None, hit_threshold=3,
             exposure_ratio: float = 1.0) -> tuple[float | None, list[int], str, np.ndarray, np.ndarray, float, list[int]]:
        """Process one lores frame.

        Args:
            y_plane:        lores Y plane.
            freeze:         if True, skip background model updates (called during an event).
            anchor_zone:    index of the zone that started the current event, or None
                            during monitoring.
            exposure_ratio: (current_exp * current_ag) / (anchor_exp * anchor_ag).
                            Supplied by the caller from frame metadata; used to correct
                            the frozen background model for AGC shifts during an event.
                            1.0 when not in a frozen event.

        Returns:
            (centroid, occupied_zones, reason, zone_zm, zone_zv, gain_used, zv_occupied)
            zone_zm / zone_zv are float32 arrays of length num_zones containing the
            per-zone peak z-score for the mean and CV signals respectively.
            Zones that are not yet warmed up carry NaN.
            gain_used is the multiplicative exposure-gain correction applied this frame.
            occupied_zones uses zm+zv AND gate — used for event triggering only.
            zv_occupied uses ZV_HIT_THRESHOLD on zv alone — used for centroid and
            oversized detection.
        """
        occupied = []
        max_zm = 0.0
        max_zv = 0.0
        zone_zm = np.full(self.num_zones, np.nan, dtype=np.float32)
        zone_zv = np.full(self.num_zones, np.nan, dtype=np.float32)

        # Collect per-zone stats for all ready zones in one pass.
        zone_stats = []   # list of (i, x0, bg, col_means, col_cvs) for ready zones
        for i, (x0, bg) in enumerate(zip(self.zone_starts, self.bgs)):
            m, cv = self._zone_stats(y_plane, x0)
            if not bg.ready:
                bg.update(m, cv)
                zone_stats.append(None)
                continue
            zone_stats.append((i, x0, bg, m, cv))

        # During a frozen event, scale the frozen background model by the ratio of
        # current exposure to the exposure at event start.  This corrects for AGC
        # shifts without relying on unoccupied reference zones (which may be
        # contaminated by the vehicle halo or absent when the vehicle fills the frame).
        gain = exposure_ratio if freeze else 1.0
        if not freeze:
            self._last_gain = 1.0   # reset between events

        for entry in zone_stats:
            if entry is None:
                continue
            i, x0, bg, m, cv = entry
            if gain != 1.0:
                corrected_mean_m = bg.mean_m * gain
                std_m = np.maximum(bg.mad_m * 1.4826, 1.0)
                std_v = np.maximum(bg.mad_v * 1.4826, 0.005)
                zm = np.abs(m - corrected_mean_m) / std_m
                zv = np.abs(cv - bg.mean_v) / std_v
            else:
                zm, zv = bg.z_scores(m, cv)
            # Monitoring: require BOTH zm and zv to be anomalous (AND) to avoid
            # false triggers from shadows (which shift mean but not texture) or
            # global exposure steps (which shift mean uniformly across all zones).
            # During a confirmed event: use zv alone to sustain tracking.
            # zv (CV z-score) is insensitive to global brightness shifts — a uniform
            # exposure change scales both std and mean together, leaving CV unchanged —
            # so it reliably tracks the vehicle's physical location without being
            # fooled by AGC steps.  zm is still used for initial triggering where the
            # AND gate prevents shadow/exposure false starts.
            if freeze:
                hit = int((zv > self.threshold).sum())
            else:
                hit = int(((zm > self.threshold) & (zv > self.threshold)).sum())
            max_zm = max(max_zm, zm.max())
            max_zv = max(max_zv, zv.max())

            zone_zm[i] = float(zm.max())
            zone_zv[i] = float(zv.max())

            if hit >= hit_threshold:
                occupied.append(i)
            elif not freeze:
                bg.update(m, cv)

        if not self.ready:
            return None, [], "warming up", zone_zm, zone_zv, gain, []

        if not occupied:
            return None, [], f"clear  zm={max_zm:.1f}  zv={max_zv:.1f}", zone_zm, zone_zv, gain, []

        max_z_tag = f"  zm={max_zm:.1f}  zv={max_zv:.1f}"

        # zv-hot zones: used for centroid and oversized detection.
        # zm+zv AND gate (occupied) is kept for initial trigger only.
        zv_occupied = [i for i in range(self.num_zones)
                       if not np.isnan(zone_zv[i]) and zone_zv[i] >= ZV_HIT_THRESHOLD]

        if zv_occupied:
            # Centroid from zv-hot zones only — tracks physical vehicle position
            # without being contaminated by road shadows or AGC-driven zm artefacts.
            centroid = float(np.median([self.zone_centers[i] for i in zv_occupied]))
        else:
            # Fallback to trigger-occupied zones if zv is below threshold
            # (shouldn't normally happen during a confirmed event).
            centroid = float(np.median([self.zone_centers[i] for i in occupied]))

        offset = centroid - self.frame_center
        reason = (f"zones={occupied}  zv_zones={len(zv_occupied)}  centroid={centroid:.0f}"
                  f"  offset={offset:+.0f}{max_z_tag}")
        return centroid, occupied, reason, zone_zm, zone_zv, gain, zv_occupied


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
# CSV debug files queued for rsync transfer: each item is a Path.
_csv_queue: queue.Queue = queue.Queue()
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


def _corner_histogram_match(bg_rgb: np.ndarray, car_rgb: np.ndarray,
                            corner_frac: float = 0.10) -> np.ndarray:
    """Return a copy of bg_rgb exposure-matched to car_rgb using only the four corners.

    For each channel independently, builds a cumulative histogram from the four
    corner patches (top-left, top-right, bottom-left, bottom-right) of both images,
    then derives a 256-entry LUT that maps bg pixel values to the corresponding
    quantile in the car image.  Applying this LUT to the full bg_rgb cancels global
    exposure / gamma shifts before the diff is computed, without being contaminated
    by vehicle pixels (which are concentrated in the centre of the frame).
    """
    h, w = bg_rgb.shape[:2]
    ph, pw = max(1, int(h * corner_frac)), max(1, int(w * corner_frac))

    def _corner_pixels(img: np.ndarray) -> np.ndarray:
        patches = [
            img[:ph,   :pw,   :],   # top-left
            img[:ph,   w-pw:, :],   # top-right
            img[h-ph:, :pw,   :],   # bottom-left
            img[h-ph:, w-pw:, :],   # bottom-right
        ]
        return np.concatenate([p.reshape(-1, 3) for p in patches], axis=0)

    bg_px  = _corner_pixels(bg_rgb)
    car_px = _corner_pixels(car_rgb)

    bg_matched = bg_rgb.copy().astype(np.uint8)
    for c in range(3):
        bg_hist,  _ = np.histogram(bg_px[:,  c], bins=256, range=(0, 256))
        car_hist, _ = np.histogram(car_px[:, c], bins=256, range=(0, 256))
        bg_cdf  = bg_hist.cumsum().astype(np.float32);  bg_cdf  /= bg_cdf[-1]  + 1e-6
        car_cdf = car_hist.cumsum().astype(np.float32); car_cdf /= car_cdf[-1] + 1e-6
        lut = np.searchsorted(car_cdf, bg_cdf).astype(np.uint8)
        bg_matched[:, :, c] = lut[bg_rgb[:, :, c]]

    return bg_matched


def _build_fg_mask(bg_rgb: np.ndarray, car_rgb: np.ndarray) -> np.ndarray:
    """Return a uint8 mask (255=foreground, 0=background) for car_rgb given bg_rgb.

    Pipeline:
      0. Corner histogram match — bg_rgb is exposure-corrected to car_rgb using
         the four 10%-corner patches before any diff is computed.
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
    bg_rgb = _corner_histogram_match(bg_rgb, car_rgb)

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

    final = cv2.bitwise_or(mask, holes)

    # Ground-line clamp + shadow removal — runs AFTER hole-fill so it cannot be
    # undone by the flood-fill step.
    #
    # Strategy:
    #   1. Find the vehicle floor row: the lowest fg row within the central 2/3 of
    #      the horizontal body span (avoids shadow pixels at the edges inflating it).
    #   2. Compute body x-span (5th/95th percentile of per-row extents above the
    #      floor, to exclude low-hanging wheel arches inflating the span).
    #   3. For every row below floor+margin, remove fg pixels that are:
    #      a. darker than bg (shadow regardless of x), OR
    #      b. outside the body x-span (trailing/leading wake).
    #   4. Bottom-edge profile clamp: fit a median-smoothed curve to the lowest fg
    #      row per column (within the body x-span), then zero any fg pixels that
    #      fall more than BG_GROUND_MARGIN_PX below that curve.  This catches shadow
    #      halos on dark vehicles where the shadow is the same luminance as the bg
    #      and the darkness test cannot fire.
    h, w = final.shape
    fg_ys, fg_xs = np.where(final > 127)
    if len(fg_ys) > 0:
        x_min, x_max = int(fg_xs.min()), int(fg_xs.max())
        core_x_lo = x_min + (x_max - x_min) // 6
        core_x_hi = x_max - (x_max - x_min) // 6
        core_mask = (fg_xs >= core_x_lo) & (fg_xs <= core_x_hi)
        if core_mask.any():
            # Use 85th percentile rather than max() so that shadow pixels
            # below the vehicle floor (which merge into the fg mask on dark
            # cars) don't drag ground_row — and therefore floor_limit — down
            # into the shadow region, defeating the clamp that follows.
            ground_row  = int(np.percentile(fg_ys[core_mask], 85))
            floor_limit = ground_row + BG_GROUND_MARGIN_PX

            # Body x-span: 5th/95th percentile of per-row extents above the floor
            above = fg_ys < floor_limit
            if above.any():
                above_ys = fg_ys[above]
                above_xs = fg_xs[above]
                unique_rows = np.unique(above_ys)
                row_x_lo = np.array([above_xs[above_ys == r].min() for r in unique_rows])
                row_x_hi = np.array([above_xs[above_ys == r].max() for r in unique_rows])
                body_x_min = int(np.percentile(row_x_lo, 5))
                body_x_max = int(np.percentile(row_x_hi, 95))
            else:
                body_x_min, body_x_max = x_min, x_max

            # Pass 1: luminance + lateral tests
            for r in range(floor_limit, h):
                fg_in_row = np.where(final[r, :] > 127)[0]
                if len(fg_in_row) == 0:
                    continue
                car_lum = car_rgb[r, fg_in_row].mean(axis=1).astype(np.float32)
                bg_lum  =  bg_rgb[r, fg_in_row].mean(axis=1).astype(np.float32)
                shadow       = car_lum < bg_lum - BG_SHADOW_DARK_MARGIN
                outside_body = (fg_in_row < body_x_min) | (fg_in_row > body_x_max)
                final[r, fg_in_row[shadow | outside_body]] = 0

            # Pass 2: bottom-edge profile clamp — catches dark-car shadows that
            # match bg luminance and survive pass 1, AND lateral streaks that extend
            # beyond body_x_max (which pass 1 misses when body_x_max is inflated by
            # a low wheel arch).
            #
            # Scan the full fg x-range (x_min..x_max).  For each column, find the
            # lowest fg row and build a bottom-edge profile.  Median-smooth only the
            # columns within body_x_min..body_x_max (the vehicle undercarriage); for
            # columns outside that span the smoothed value is clamped to floor_limit
            # so everything below the vehicle floor is zeroed there.
            col_range = np.arange(x_min, x_max + 1)
            bot_edge = np.full(len(col_range), np.nan)
            for i, col in enumerate(col_range):
                # Use only pixels above floor_limit so shadow pixels below the
                # vehicle floor don't drag the bottom-edge profile downward.
                hits = np.where(final[:floor_limit, col] > 127)[0]
                if len(hits):
                    bot_edge[i] = hits.max()
            # Build the smoothed profile over the body span, clamp outside it
            valid_bot = ~np.isnan(bot_edge)
            if valid_bot.sum() > 0:
                # Forward/backward fill NaNs before smoothing
                filled = bot_edge.copy()
                last = np.nan
                for i in range(len(filled)):
                    if not np.isnan(filled[i]): last = filled[i]
                    elif not np.isnan(last):     filled[i] = last
                last = np.nan
                for i in range(len(filled) - 1, -1, -1):
                    if not np.isnan(filled[i]): last = filled[i]
                    elif not np.isnan(last):     filled[i] = last
                half = BG_TOP_MEDIAN_W // 2
                padded = np.pad(filled, half, mode='edge')
                smoothed_bot = np.array([np.median(padded[i:i + BG_TOP_MEDIAN_W])
                                         for i in range(len(filled))])
                for i, col in enumerate(col_range):
                    if np.isnan(bot_edge[i]):
                        continue
                    # Outside the body x-span: zero everything at or below floor_limit
                    if col < body_x_min or col > body_x_max:
                        final[floor_limit:, col] = 0
                    else:
                        floor_row = int(smoothed_bot[i]) + BG_GROUND_MARGIN_PX
                        if floor_row < h - 1:
                            final[floor_row + 1:, col] = 0

    # Bottom-10% lateral clamp: prevent shadow "beaks" that extend beyond the
    # vehicle body at the very bottom of the mask.  For the lowest 10% of the
    # fg bounding box height, clip each row's white pixels to the left/right
    # extent of the fg pixels in the rows above that band.  Purely subtractive.
    fg_ys_c, fg_xs_c = np.where(final > 127)
    if len(fg_ys_c) > 0:
        fg_top_c  = int(fg_ys_c.min())
        fg_bot_c  = int(fg_ys_c.max())
        fg_h_c    = fg_bot_c - fg_top_c
        if fg_h_c > 0:
            beak_row = fg_bot_c - max(1, int(fg_h_c * 0.10))
            above_mask_c = fg_ys_c < beak_row
            if above_mask_c.any():
                upper_x_min = int(fg_xs_c[above_mask_c].min())
                upper_x_max = int(fg_xs_c[above_mask_c].max())
                for r in range(beak_row, fg_bot_c + 1):
                    row_fg = np.where(final[r, :] > 127)[0]
                    if len(row_fg) == 0:
                        continue
                    outside = (row_fg < upper_x_min) | (row_fg > upper_x_max)
                    if outside.any():
                        final[r, row_fg[outside]] = 0

    # Final connected-component filter: discard thin streaks or isolated blobs
    # that survive all prior cleanup.
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(final, connectivity=8)
    if n_labels > 1:
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        final = np.where(labels == largest, np.uint8(255), np.uint8(0))

    return final


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
        back_path_saved: Path | None = None   # guard against duplicate _back.jpg per job

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

                    # Debug: save background reference image when mask is suspiciously large.
                    # Condition: mask has white pixels in both left and right corner columns
                    # AND more than 50% of all pixels are white.
                    # Only saved once per job: all vehicle frames share the same background
                    # and produce the same filename (frames[0][0].stem), so saving it for
                    # each frame would append duplicate entries to `converted` and cause
                    # rsync to fail on the second attempt after the first transfer succeeds.
                    if bg_rgb is not None and back_path_saved is None:
                        mask_w = mask.shape[1]
                        corner_w  = max(2, mask_w // 20)   # ~5% of frame width
                        left_hit  = mask[:, :corner_w].any()
                        right_hit = mask[:, mask_w - corner_w:].any()
                        white_frac = (mask > 0).mean()
                        if left_hit and right_hit and white_frac > 0.5:
                            bg_stem = f"{frames[0][0].stem}_{event_num}_back"
                            back_path = npy_path.with_name(f"{bg_stem}.jpg")
                            PilImage.fromarray(bg_rgb).save(
                                str(back_path), format="JPEG", quality=92)
                            log.info("[worker] debug bg saved %s  (white=%.1f%%)",
                                     back_path.name, white_frac * 100)
                            converted.append((None, back_path))
                            back_path_saved = back_path
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

        # Transfer any pending CSV debug files.
        while True:
            try:
                csv_path = _csv_queue.get_nowait()
            except queue.Empty:
                break
            if csv_path.exists():
                ok = rsync_file(csv_path)
                if ok:
                    log.info("[worker] transferred csv %s", csv_path.name)
                else:
                    log.warning("[worker] csv transfer failed, keeping %s", csv_path.name)
                    _failed_transfers.append((None, csv_path))

        _work_queue.task_done()


# ---------------------------------------------------------------------------
# Zone debug logger — captures per-zone zm/zv for every frame in a window
# around each vehicle event and writes a CSV to SHM_DIR for transfer.
#
# Layout:
#   phase       — "pre", "event", "post", "cooldown"
#   frame_num   — global frame counter from main loop
#   timestamp   — HH:MM:SS.mmm wall-clock string
#   centroid    — lores-pixel centroid (NaN when no vehicle detected)
#   occupied    — comma-separated list of occupied zone indices (empty string when none)
#   zm_00…zm_19 — per-zone peak mean z-score  (NaN = zone not warmed up)
#   zv_00…zv_19 — per-zone peak CV z-score    (NaN = zone not warmed up)
# ---------------------------------------------------------------------------

_DEBUG_PRE_FRAMES  = 5    # rolling pre-event frames to prepend
_DEBUG_POST_FRAMES = 20   # tail frames to append after event ends

class ZoneDebugLogger:
    """Accumulates zone zm/zv data around one vehicle event and writes a CSV."""

    def __init__(self, num_zones: int = NUM_ZONES):
        self._num_zones  = num_zones
        self._pre_buf    : collections.deque = collections.deque(maxlen=_DEBUG_PRE_FRAMES)
        self._rows       : list = []          # rows for the current active event
        self._active     : bool = False
        self._tail_left  : int  = 0           # post-event frames still to collect
        self._first_stem : str  = ""          # timestamp stem of the first pre-event frame

    # ------------------------------------------------------------------
    # Call every frame regardless of event state
    # ------------------------------------------------------------------
    def record(self, stem: str, frame_num: int, dt: datetime,
               centroid, occupied: list,
               zone_zm: np.ndarray, zone_zv: np.ndarray,
               phase: str, meta: dict | None = None,
               gain_used: float = 1.0):
        """Add one frame's data."""
        m = meta or {}
        exp_us  = m.get("ExposureTime")
        ag      = m.get("AnalogueGain")
        dg      = m.get("DigitalGain")
        lux     = m.get("Lux")
        row = {
            "phase":        phase,
            "frame_num":    frame_num,
            "timestamp":    dt.strftime("%H:%M:%S.") + f"{dt.microsecond // 1000:03d}",
            "exp_us":       f"{exp_us:.0f}" if exp_us is not None else "nan",
            "analogue_gain":f"{ag:.3f}"     if ag      is not None else "nan",
            "digital_gain": f"{dg:.3f}"     if dg      is not None else "nan",
            "lux":          f"{lux:.1f}"    if lux     is not None else "nan",
            "gain_used":    f"{gain_used:.4f}",
            "centroid":     f"{centroid:.1f}" if centroid is not None else "nan",
            "occupied":     ";".join(str(z) for z in occupied),
        }
        for z in range(self._num_zones):
            row[f"zm_{z:02d}"] = f"{zone_zm[z]:.3f}" if not np.isnan(zone_zm[z]) else "nan"
            row[f"zv_{z:02d}"] = f"{zone_zv[z]:.3f}" if not np.isnan(zone_zv[z]) else "nan"

        if not self._active:
            # Rolling pre-event buffer — keep a sliding window of recent frames.
            self._pre_buf.append((stem, row))
        else:
            self._rows.append(row)

    # ------------------------------------------------------------------
    def start_event(self):
        """Call at EVENT START — commits the pre-event buffer and opens the event window."""
        if self._active:
            return  # guard against double-start
        self._active = True
        self._tail_left = 0
        # Pull all buffered pre-frames into _rows; remember the earliest stem for the filename.
        pre_rows = list(self._pre_buf)
        if pre_rows:
            self._first_stem = pre_rows[0][0]
            self._rows = [r for _, r in pre_rows]
        else:
            self._first_stem = ""
            self._rows = []
        self._pre_buf.clear()

    # ------------------------------------------------------------------
    def end_event(self):
        """Call when the event ends (save, timeout, or no-save clear).

        Starts the post-event tail counter; the logger remains active
        until the tail is exhausted, then auto-flushes.
        """
        if not self._active:
            return
        self._tail_left = _DEBUG_POST_FRAMES

    # ------------------------------------------------------------------
    def tick_post(self, stem: str, frame_num: int, dt: datetime,
                  centroid, occupied: list,
                  zone_zm: np.ndarray, zone_zv: np.ndarray,
                  phase: str, meta: dict | None = None,
                  gain_used: float = 1.0) -> bool:
        """Record one post-event tail frame. Returns True when tail exhausted and CSV written."""
        if not self._active or self._tail_left <= 0:
            return False
        self.record(stem, frame_num, dt, centroid, occupied, zone_zm, zone_zv, phase, meta, gain_used)
        self._tail_left -= 1
        if self._tail_left == 0:
            self._flush()
            return True
        return False

    # ------------------------------------------------------------------
    def flush_now(self):
        """Force an immediate flush (e.g. on timeout or program exit)."""
        if self._active and self._rows:
            self._flush()

    # ------------------------------------------------------------------
    def reset(self):
        """Return to pre-event state."""
        self._active    = False
        self._tail_left = 0
        self._rows      = []
        self._first_stem = ""
        self._pre_buf.clear()

    @property
    def active(self) -> bool:
        return self._active

    @property
    def in_tail(self) -> bool:
        return self._active and self._tail_left > 0

    # ------------------------------------------------------------------
    def _flush(self):
        """Write the accumulated rows to a CSV in SHM_DIR and queue for transfer."""
        if not self._rows:
            self.reset()
            return
        stem = self._first_stem or ts_stem(datetime.now())
        csv_path = SHM_DIR / f"{stem}_zones.csv"
        try:
            SHM_DIR.mkdir(parents=True, exist_ok=True)
            import csv as _csv
            fieldnames = (["phase", "frame_num", "timestamp",
                           "exp_us", "analogue_gain", "digital_gain", "lux",
                           "gain_used", "centroid", "occupied"]
                          + [f"zm_{z:02d}" for z in range(self._num_zones)]
                          + [f"zv_{z:02d}" for z in range(self._num_zones)])
            with open(csv_path, "w", newline="") as fh:
                writer = _csv.DictWriter(fh, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(self._rows)
            log.info("[debug_csv] wrote %d rows → %s", len(self._rows), csv_path.name)
            _csv_queue.put(csv_path)
        except Exception as e:
            log.warning("[debug_csv] failed to write %s: %s", csv_path.name, e)
        self.reset()


# ---------------------------------------------------------------------------
# Burst saver — enqueues conversion job
# ---------------------------------------------------------------------------

class BurstSaver:
    def __init__(self):
        self.event_num, self._last_event_date = _load_event_num()
        today = datetime.now().strftime("%Y%m%d")
        if self.event_num and self._last_event_date == today:
            log.info("[event_num] resuming from event %d (today)", self.event_num)
        else:
            if self.event_num and self._last_event_date != today:
                log.info("[event_num] new day — resetting event counter")
            else:
                log.info("[event_num] starting fresh event numbering for today")
            self.event_num = 0
            self._last_event_date = today

    def save(self, frames: list[tuple], shm_min_mb: float) -> int:
        """Enqueue pre-saved (npy_path, meta) tuples for conversion.
        Frame arrays are written at capture time (while the request is live);
        this method handles queuing only. Space was already checked at write time."""
        today = datetime.now().strftime("%Y%m%d")
        if today != self._last_event_date:
            log.info("[event_num] date rolled over (%s -> %s) — resetting event counter",
                     self._last_event_date, today)
            self.event_num = 0
            self._last_event_date = today
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
                   default=str(Path(__file__).parent / "custom_pisp.json"),
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
    print("Burst Capture 2026/5/30 version %s" % VERSION)
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
        log.info("[tuning] loaded tuning file: %s", args.tune)

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
    zone_logger    = ZoneDebugLogger()
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
    event_anchor_exposure: float       = 1.0    # exp_us * analogue_gain at EVENT START (for ratio correction)
    oversized_frame_full_saved = False  # True after frame-full save for an oversized vehicle
    oversized_stash: list | None = None  # [prev_frame, cur_frame] captured at frame-full moment
    oversized_midpoint_queued: set = set()  # npy paths already queued by midpoint save; must not be unlinked
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

    def _exposure_ev(m: dict) -> float:
        """Return ExposureTime_us * AnalogueGain from frame metadata, or 1.0 if unavailable."""
        exp = m.get("ExposureTime")
        ag  = m.get("AnalogueGain", 1.0)
        return float(exp * ag) if exp is not None else 1.0

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
            centroid, occupied, reason, zone_zm, zone_zv, gain_used, zv_occupied = tracker.feed(
                                               y_plane, freeze=False, exposure_ratio=1.0)
            if zone_logger.in_tail:
                done = zone_logger.tick_post(stem, frame_count, dt, centroid, occupied,
                                             zone_zm, zone_zv, "cooldown", meta, gain_used)
                if done:
                    zone_logger.reset()
            if cur_frame:
                cur_frame[0].unlink(missing_ok=True)
            if pre_event_frame:
                pre_event_frame[0].unlink(missing_ok=True)
                pre_event_frame = None
            prev_frame = None
            prev_prev_frame = None
            prev_centroid = None
            continue

        cur_exposure_ev = _exposure_ev(meta)
        exposure_ratio  = (cur_exposure_ev / event_anchor_exposure
                           if in_event and event_anchor_exposure > 0 else 1.0)

        centroid, occupied, reason, zone_zm, zone_zv, gain_used, zv_occupied = tracker.feed(
                                           y_plane, freeze=in_event,
                                           anchor_zone=event_anchor_zone if in_event else None,
                                           hit_threshold=1 if in_event else 3,
                                           exposure_ratio=exposure_ratio)

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
                event_anchor_exposure = cur_exposure_ev
                oversized_frame_full_saved = False
                oversized_stash        = None
                oversized_midpoint_queued = set()
                zone_logger.start_event()
                zone_logger.record(stem, frame_count, dt, centroid, occupied,
                                   zone_zm, zone_zv, "event", meta, gain_used)
                log.info("EVENT START  frame=%d  %s", frame_count, reason)
            else:
                # Monitoring: roll the 2-frame history forward.
                # pre-event rolling buffer gets every monitoring frame.
                if zone_logger.in_tail:
                    done = zone_logger.tick_post(stem, frame_count, dt, centroid, occupied,
                                                 zone_zm, zone_zv, "post", meta, gain_used)
                    if done:
                        zone_logger.reset()
                else:
                    zone_logger.record(stem, frame_count, dt, centroid, occupied,
                                       zone_zm, zone_zv, "pre", meta, gain_used)
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
                zone_logger.flush_now()
                tracker.rewarm()
                in_event          = False
                event_frame_count = 0
                prev_frame        = None
                prev_prev_frame   = None
                prev_centroid     = None
                event_start_centroid = None
                event_anchor_zone    = None
                event_anchor_exposure = 1.0
                oversized_frame_full_saved = False
                oversized_stash        = None
                oversized_midpoint_queued = set()
                cooldown_until    = now + args.cooldown
                continue

            log.info("  event frame %d  %s", event_frame_count, reason)
            zone_logger.record(stem, frame_count, dt, centroid, occupied,
                               zone_zm, zone_zv, "event", meta, gain_used)

            if not occupied:
                # Vehicle has cleared all zones.
                # Fall back to saving if it travelled far enough laterally,
                # even if it never crossed the frame midpoint.
                travel = (abs(prev_centroid - event_start_centroid)
                          if prev_centroid is not None and event_start_centroid is not None
                          else 0.0)
                if (event_frame_count >= 3 and travel >= EVENT_MIN_TRAVEL
                        and prev_frame and cur_frame and pre_event_frame is not None):
                    had_pre = pre_event_frame is not None
                    frames_to_save = ([pre_event_frame] if pre_event_frame else []) + [prev_frame, cur_frame]
                    n = saver.save(frames_to_save, args.shm_min_mb)
                    pre_event_frame = None
                    event_count += 1
                    log.info("SAVE (travel=%.0fpx)  event=%d  pre_event=%s",
                             travel, saver.event_num, "yes" if had_pre else "no")
                    log.info("Event %d queued for conversion (%d frames)", event_count, n)
                else:
                    log.info("EVENT END (no save, travel=%.0fpx) pre_event=%s", travel, "yes" if pre_event_frame else "no")
                    if cur_frame:
                        cur_frame[0].unlink(missing_ok=True)
                    if prev_frame:
                        prev_frame[0].unlink(missing_ok=True)
                    if pre_event_frame:
                        pre_event_frame[0].unlink(missing_ok=True)
                        pre_event_frame = None
                zone_logger.end_event()
                # tracker.reanchor(4)
                in_event          = False
                event_frame_count = 0
                prev_frame        = None
                prev_prev_frame   = None
                prev_centroid     = None
                event_start_centroid = None
                event_anchor_zone    = None
                event_anchor_exposure = 1.0
                oversized_frame_full_saved = False
                oversized_stash        = None
                oversized_midpoint_queued = set()
                cooldown_until    = now + (args.cooldown if event_frame_count > 2 else 0.0)
                continue

            # Check for oversized vehicle: uses zv-hot zone count rather than the
            # zm+zv trigger-occupied count, so road shadows don't inflate the size estimate.
            frame_full = len(zv_occupied) >= ZV_OVERSIZED_ZONES

            if frame_full and not oversized_frame_full_saved:
                # Vehicle has just filled the frame — stash this pair; do not save yet.
                # We'll combine it with the exit pair into a single 4-frame event later.
                if prev_frame and cur_frame:
                    had_pre = pre_event_frame is not None
                    oversized_stash = (([pre_event_frame] if pre_event_frame else [])
                                       + [prev_frame, cur_frame])
                    pre_event_frame = None
                    oversized_frame_full_saved = True
                    log.info("[oversized] frame-full stashed  zv_zones=%d  pre_event=%s",
                             len(zv_occupied), "yes" if had_pre else "no")
                else:
                    log.warning("[event] frame-full but frame(s) missing (shm low?); skipping stash")
                    oversized_frame_full_saved = True  # arm exit detection anyway
                # Keep rolling — wait for frame-exit trigger.
                # Do NOT unlink prev_frame if it was just stashed — it will be
                # consumed by the worker when the exit pair is saved.
                stashed_paths = {f[0] for f in oversized_stash} if oversized_stash else set()
                if (prev_frame and prev_frame[0] != (cur_frame[0] if cur_frame else None)
                        and prev_frame[0] not in stashed_paths):
                    prev_frame[0].unlink(missing_ok=True)
                prev_frame    = cur_frame
                prev_centroid = centroid
                continue

            if oversized_frame_full_saved and not frame_full:
                # Vehicle is starting to leave — combine stash + exit frame into one event.
                # prev_frame here is always the same file as the last frame of the stash
                # (set by `prev_frame = cur_frame` in the frame-full branch), so only
                # append cur_frame to avoid a duplicate entry in the job.
                if oversized_stash and prev_frame and cur_frame:
                    frames_to_save = oversized_stash + [cur_frame]
                    # prev_frame is only equal to stash[-1] when the exit fires on the
                    # very first post-stash frame.  If rolling frames intervened it has
                    # advanced and holds an npy that is not queued — unlink it now.
                    stashed_paths = {f[0] for f in oversized_stash}
                    if prev_frame[0] not in stashed_paths and prev_frame[0] != cur_frame[0]:
                        prev_frame[0].unlink(missing_ok=True)
                    n = saver.save(frames_to_save, args.shm_min_mb)
                    event_count += 1
                    log.info("SAVE (oversized 4-frame)  event=%d  total_frames=%d  exit_zv_zones=%d",
                             saver.event_num, n, len(zv_occupied))
                    log.info("Event %d queued for conversion (%d frames)", event_count, n)
                else:
                    log.warning("[event] frame-exit but stash or frame(s) missing; skipping save")
                    if cur_frame and cur_frame[0] not in oversized_midpoint_queued:
                        cur_frame[0].unlink(missing_ok=True)
                    if prev_frame and prev_frame[0] not in oversized_midpoint_queued:
                        prev_frame[0].unlink(missing_ok=True)
                    if oversized_stash:
                        for f in oversized_stash:
                            if f[0] not in oversized_midpoint_queued:
                                f[0].unlink(missing_ok=True)
                in_event               = False
                event_frame_count      = 0
                prev_frame             = None
                prev_prev_frame        = None
                pre_event_frame        = None
                prev_centroid          = None
                event_start_centroid   = None
                event_anchor_zone      = None
                event_anchor_exposure  = 1.0
                oversized_frame_full_saved = False
                oversized_stash        = None
                oversized_midpoint_queued = set()
                cooldown_until         = now + args.cooldown
                zone_logger.end_event()
                continue

            # Check for midpoint crossing:
            #   - must be at least the 3rd event frame (filter pedestrians / short blips)
            #   - centroid must have moved past the frame midpoint this frame
            #   - skipped for oversized vehicles (handled above)
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
                    log.info("SAVE%s  event=%d  pre_event=%s  prev_offset=%+.0f  cur_offset=%+.0f",
                             " (oversized midpoint)" if frame_full else "",
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

                # For oversized vehicles keep the event running so the frame-exit save
                # can still fire.  Just discard pre_event_frame (already saved) and
                # arm the oversized stash machinery.
                if frame_full:
                    pre_event_frame = None
                    oversized_frame_full_saved = True   # skip the stash; go straight to exit watch
                    oversized_stash = None
                    # Track npy paths queued by the midpoint save so the frame-exit
                    # else-branch cannot unlink them and cause a worker FileNotFoundError.
                    # frames_to_save is defined above in the `if prev_frame and cur_frame`
                    # branch; if that branch was skipped (shm low) there is nothing queued.
                    oversized_midpoint_queued = ({f[0] for f in frames_to_save}
                                                 if prev_frame and cur_frame else set())
                    prev_frame    = cur_frame
                    prev_centroid = centroid
                    continue

                # tracker.reanchor(4)
                in_event          = False
                event_frame_count = 0
                prev_frame        = None
                prev_prev_frame   = None
                pre_event_frame   = None
                prev_centroid     = None
                event_start_centroid = None
                event_anchor_zone    = None
                event_anchor_exposure = 1.0
                oversized_frame_full_saved = False
                oversized_stash        = None
                oversized_midpoint_queued = set()
                cooldown_until    = now + args.cooldown
                zone_logger.end_event()
                continue

            # Centroid hasn't crossed yet — keep rolling
            # Discard the frame before prev_frame (no longer needed),
            # but only if it hasn't been stashed for an oversized event.
            stashed_paths = {f[0] for f in oversized_stash} if oversized_stash else set()
            if (prev_frame and prev_frame[0] != (cur_frame[0] if cur_frame else None)
                    and prev_frame[0] not in stashed_paths):
                prev_frame[0].unlink(missing_ok=True)
            prev_frame    = cur_frame
            prev_centroid = centroid

    picam2.stop()

    zone_logger.flush_now()   # write any partial event CSV before exiting

    # Signal worker to exit; if interrupted, stop_event is already set and
    # the worker will bail out without draining remaining jobs.
    log.info("Waiting for worker thread to exit…")
    _work_queue.put(None)
    worker_thread.join()
    log.info("Done. %d events captured, %d frames processed.", event_count, frame_count)


if __name__ == "__main__":
    main()
