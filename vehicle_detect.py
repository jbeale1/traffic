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
import json
import math
import argparse
import collections
import csv
import logging
import signal
import socket
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

VERSION        = "1.113" # add focus mode button to web page
FRAME_RATE     = 20.0
LORES_SIZE     = (320, 240)
HIRES_SIZE     = (1456, 1088)
JPEG_QUALITY   = 95

# Set True to save a lores MOG2 mask PNG alongside every captured JPEG.
# Useful for algorithm development; disable for normal operation.
#SAVE_MASK      = True
SAVE_MASK      = False

# Crop applied to saved full-res JPEGs (in 1456x1088 frame coordinates)
HIRES_CROP_TOP    = 76+91  # add 91 pixels to make active region centered on sensor
HIRES_CROP_BOTTOM = 830+91

# Background blur applied outside the vehicle bounding box before JPEG save.
# Reduces file size by softening high-frequency background (trees, grass, etc.).
BLUR_KERNEL_SIZE  = 7     # must be odd; 7x7 Gaussian
BLUR_MARGIN_HIRES = 60    # unblurred padding around vehicle bbox (hires pixels)
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
ROI_TOP    = 60+20
ROI_BOTTOM = 130+20+6  # any lower includes shadows on road

MOG2_HISTORY        = 1000
MOG2_DETECT_SHADOWS = False

# Edge detection parameters (Scharr operator on grayscale lores)
# Scharr gives better gradient isotropy than Sobel at 3x3.
# EDGE_BLUR_KSIZE: pre-blur kernel size (must be odd) to suppress noise before
# gradient computation; 0 to disable.
EDGE_BLUR_KSIZE = 3

# Fixed clip range for Scharr → uint8 scaling (applied to each signed channel).
# Each channel (sx, sy) is clipped to [-EDGE_CLIP_MAX, +EDGE_CLIP_MAX] then
# shifted to [0, 255] with 128 as zero.  This preserves edge direction and sign,
# giving MOG2 a richer 2-channel joint distribution to model:
#   - Background foliage: high variance in both channels, mean near 128
#   - Smooth vehicle panel: both channels near 128 (near-zero gradient) —
#     different mean from foliage → detected as foreground
#   - Vehicle boundary: large excursion in sx or sy → outlier from background
# Per-frame normalisation is NOT used: it destroys absolute magnitude
# relationships, making a smooth-surfaced vehicle invisible against textured
# background.
#
# Scharr on uint8 input: max possible value ≈ 3600 (hard black→white step).
# Typical background texture (road, grass) peaks in the 200–600 range.
# EDGE_CLIP_MAX sets the signed magnitude that maps to 0 or 255.
EDGE_CLIP_MAX       = 500.0   # tune empirically; raise if background is noisy
MOG2_VAR_THRESHOLD  = 10      # tune after setting EDGE_CLIP_MAX; 2-ch input needs lower value

MORPH_KERNEL_SIZE = 5

# Horizontal open kernel applied after the standard open/close to suppress the
# ghost trail left at the vehicle's roofline and road-edge boundaries after it
# passes.  The trail appears as two thin horizontal strips (top and bottom of
# the vehicle path) caused by strong boundary edges that MOG2 cannot relearn
# while LR_VEHICLE=0.0.  A wide, short kernel erodes these strips while
# preserving the taller filled blob of the vehicle body.
# Width should be at least as wide as the trail gap between the two strips.
# Height of 1 makes it maximally selective for thin horizontal artefacts.
MORPH_HOPEN_W = 15   # pixels; raise if trail persists, lower if vehicle body is eroded
MORPH_HOPEN_H = 3    # pixels; 1 = only exact single-row strips suppressed

MIN_BLOB_AREA       = 100 # catch smaller objects like bicyclists
MIN_ASPECT_RATIO    = 0.34 # capture pedestrians
MIN_BLOB_WIDTH      = 20
MIN_BLOB_HEIGHT     = 20   # lowered from 25 to avoid intermittent streak resets on dark cars
MIN_HULL_FILL_RATIO = 0.44 # lowered from 0.45 to avoid boundary ties

# Minimum ROI-relative row that the blob's bounding box must reach.
# Rejects background grass / illumination blobs that never touch the road zone.
# Calibrated from observed data: grass-only false triggers bottom out ~row 35,
# real vehicles reach ~row 75+.
MIN_BLOB_BOTTOM_ROW = 60  # was 50
MIN_CENTROID_AREA   = 400 # was 3000, to see motorcycles and bicyclists

# Minimum foreground pixel count to log when --verbose-fg is active.
VERBOSE_FG_MIN_PIXELS = 150

MIN_CONSECUTIVE_FRAMES = 3  # minimum consecutive frames that a blob must persist to start an event
MIN_EVENT_FRAMES       = 4   # minimum confirmed frames to accept an event as valid
MIN_INTERIOR_VEL_FRAMES = 3  # minimum interior (edges=..) frames to use for velocity fit
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

# Fractional EV deviation from the pre-event baseline above which a centroid
# history entry is considered EV-suspect (AGC shifted significantly from the
# background the MOG2 model was built on).  Suspect frames are trimmed from the
# tail of centroid_history before motion quality evaluation.
# 0.15 = 15% EV change, roughly half a stop.  Set to 0 to disable trimming.
# NOTE: with edge-based input this is less critical but retained as a safety net.
EV_SUSPECT_THRESHOLD = 0.15

# Number of pre-event frames used to compute the EV baseline for suspect detection.
EV_PRE_EVENT_WINDOW = 30   # ~1.5 s at 20 fps

# Number of frames after event end to log EV values, to observe AGC recovery.
EV_POST_EVENT_LOG_FRAMES = 0  # log EV for this many frames after event end, even if no foreground is detected

EDGE_MARGIN  = 3
FRAME_WIDTH  = LORES_SIZE[0]

# Maximum event duration before forced close and background re-adaptation
MAX_EVENT_FRAMES = int(FRAME_RATE * 2.5)  # 2.5 seconds — coordinated with HIRES_BUFFER_SIZE

# Pedestrian detection: events whose median blob width stays below this threshold
# and whose cx centroid progresses monotonically across the frame are classified
# as pedestrians.  Their timeout is extended by this multiplier so a slow-walking
# person is not force-closed before fully crossing.
PEDESTRIAN_MAX_W            = 80    # lores px; pedestrians stay narrow across the full event
PEDESTRIAN_MIN_MONO_FRAC    = 0.75  # fraction of per-frame cx steps that must share the dominant sign
PEDESTRIAN_TIMEOUT_MULT     = 2     # multiply MAX_EVENT_FRAMES by this for pedestrians

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
# LiDAR event receiver  (UDP, port 5006)
# Buffers incoming LiDAR events for correlation with camera events.
# ---------------------------------------------------------------------------

LIDAR_UDP_PORT       = 5006  # must match CAMERA_PORT in lidar_events.py
# Asymmetric match window: LiDAR often fires before the camera epoch (which comes
# from the best-centered frame, near the middle/end of a slow pedestrian crossing).
# EARLY covers the case where LiDAR precedes camera; LATE covers late UDP delivery.
LIDAR_MATCH_EARLY  = 3.0   # seconds: LiDAR may precede camera epoch by up to this
LIDAR_MATCH_LATE   = 2.0   # seconds: LiDAR may follow  camera epoch by up to this
LIDAR_BUFFER_TTL   = 10.0  # seconds: discard stale entries from both pools

# _lidar_pending: camera event info dicts not yet matched.
# Keyed by id(info); UDP thread updates them in-place.
_lidar_pending: dict = {}   # id(info) -> info dict

# _lidar_pkt_buf: received LiDAR packets not yet matched to a camera event.
# Each entry: {"t", "dur", "d", "wall_t", "fi"}.
# wall_t = time.time() at receipt; fi = main-loop frame index at receipt.
# After LIDAR_SYNTH_WAIT seconds without a native match, the main loop
# promotes the entry to a synthetic event.
_lidar_pkt_buf: list = []

_lidar_lock = threading.Lock()

# Deduplication: track recently seen LiDAR packet timestamps to discard retransmits.
# Dict of {t_rounded: wall_time_seen}; pruned in the UDP listener timeout handler.
_lidar_seen: dict = {}
LIDAR_DEDUP_TTL = 5.0   # seconds to remember a seen packet timestamp

# Frame index shared between main loop and UDP listener so packets can record
# which frame was current on arrival.  Written only by main loop (CPython GIL safe).
_current_fi: int = 0

LIDAR_SYNTH_WAIT = 2.0   # seconds: if no native event matches within this time,
                          # promote the LiDAR packet to a synthetic camera event


def _lidar_in_window(camera_epoch: float, lt: float) -> bool:
    """Return True if lt falls within the asymmetric match window around camera_epoch."""
    dt = camera_epoch - lt   # positive = LiDAR earlier than camera
    return -LIDAR_MATCH_LATE <= dt <= LIDAR_MATCH_EARLY


def _try_match_pkt(lt: float, dur: int, dist: float) -> bool:
    """Try to match a LiDAR packet against _lidar_pending (caller holds no lock).
    Returns True if matched."""
    with _lidar_lock:
        candidates = [(abs(info["epoch"] - lt), k, info)
                      for k, info in _lidar_pending.items()
                      if info.get("lidar_d") is None
                      and _lidar_in_window(info["epoch"], lt)]
    if not candidates:
        return False
    best_dt, best_k, best_info = min(candidates, key=lambda x: x[0])
    if not _lidar_in_window(best_info["epoch"], lt):
        return False
    best_info["lidar_t"]   = lt
    best_info["lidar_dur"] = dur
    best_info["lidar_d"]   = dist
    best_info["lidar_dt"]  = round(best_info["epoch"] - lt, 3)
    log.info("[lidar_match] event %s  dt=%.3f s  dur=%d ms  d=%.3f m",
             best_info.get("event_count"), best_info["lidar_dt"], dur, dist)
    with _lidar_lock:
        _lidar_pending.pop(best_k, None)
    _write_csv_for_event(best_info)
    return True


def _lidar_register(info: dict):
    """Register a newly completed camera event as awaiting a LiDAR match.

    Three paths:
    1. Synthetic event: event_meta contains '_lidar_prefill' — lidar fields already
       known; write CSV immediately, no match search needed.
    2. Buffered packet: a LiDAR packet arrived before _lidar_register was called;
       match it from _lidar_pkt_buf and write CSV immediately.
    3. No match yet: add to _lidar_pending; UDP listener will fill fields in-place
       when packet arrives.  A fallback timer writes CSV without lidar data after
       CSV_LIDAR_WAIT seconds if still unmatched.
    """
    # Path 1: synthetic event — lidar data pre-filled by _save_synthetic_event
    prefill = info.pop("_lidar_prefill", None)
    if prefill is not None:
        epoch = info["epoch"]
        info["lidar_t"]   = prefill["t"]
        info["lidar_dur"] = prefill["dur"]
        info["lidar_d"]   = prefill["d"]
        info["lidar_dt"]  = round(epoch - prefill["t"], 3)
        _write_csv_for_event(info)
        return

    epoch = info["epoch"]
    # Path 2: check for a buffered packet that arrived before this call
    with _lidar_lock:
        candidates = [(abs(p["t"] - epoch), i)
                      for i, p in enumerate(_lidar_pkt_buf)
                      if _lidar_in_window(epoch, p["t"])]
    if candidates:
        best_dt, best_i = min(candidates, key=lambda x: x[0])
        if _lidar_in_window(epoch, _lidar_pkt_buf[best_i]["t"]):
            with _lidar_lock:
                pkt = _lidar_pkt_buf.pop(best_i)
            info["lidar_t"]   = pkt["t"]
            info["lidar_dur"] = pkt["dur"]
            info["lidar_d"]   = pkt["d"]
            info["lidar_dt"]  = round(epoch - pkt["t"], 3)
            log.info("[lidar_match] event %s  dt=%.3f s  dur=%d ms  d=%.3f m (buffered pkt)",
                     info.get("event_count"), info["lidar_dt"], pkt["dur"], pkt["d"])
            _write_csv_for_event(info)
            return

    # Path 3: no immediate match — register and set a fallback CSV timer
    with _lidar_lock:
        _lidar_pending[id(info)] = info
    def _csv_fallback():
        still_pending = False
        with _lidar_lock:
            still_pending = id(info) in _lidar_pending
            if still_pending:
                _lidar_pending.pop(id(info), None)
        if still_pending:
            log.info("[csv] no lidar match for event %s after %.1f s — writing without",
                     info.get("event_count"), CSV_LIDAR_WAIT)
            _write_csv_for_event(info)
    threading.Timer(CSV_LIDAR_WAIT, _csv_fallback).start()


def _lidar_udp_listener():
    """Background thread: receive LiDAR UDP packets.
    On each packet: try to match against _lidar_pending (camera event already
    registered).  If no match, buffer the packet in _lidar_pkt_buf so that a
    camera event registering slightly later can still find it.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.bind(('', LIDAR_UDP_PORT))
    except OSError as e:
        log.error("[lidar_udp] bind failed on port %d: %s", LIDAR_UDP_PORT, e)
        return
    log.info("[lidar_udp] listening on UDP port %d", LIDAR_UDP_PORT)
    sock.settimeout(2.0)
    while True:
        try:
            data, addr = sock.recvfrom(256)
            pkt  = json.loads(data.decode())
            lt   = float(pkt["t"])
            dur  = int(pkt["dur"])
            dist = float(pkt["d"])
            log.info("[lidar_udp] rx from %s: t=%.3f dur=%d ms d=%.3f m",
                     addr[0], lt, dur, dist)
            # Discard retransmit duplicates
            with _lidar_lock:
                if lt in _lidar_seen:
                    log.info("[lidar_udp] duplicate (retransmit) — discarded")
                    continue
                _lidar_seen[lt] = time.time()
            if not _try_match_pkt(lt, dur, dist):
                # Camera event not yet registered — buffer packet for deferred match
                # or synthetic event promotion after LIDAR_SYNTH_WAIT seconds.
                with _lidar_lock:
                    _lidar_pkt_buf.append({
                        "t": lt, "dur": dur, "d": dist,
                        "wall_t": time.time(), "fi": _current_fi,
                    })
                log.info("[lidar_udp] buffered for deferred match (fi=%d)", _current_fi)
        except socket.timeout:
            # Prune stale entries from both pools
            now = time.time()
            with _lidar_lock:
                stale_k = [k for k, info in _lidar_pending.items()
                           if now - info.get("epoch", now) > LIDAR_BUFFER_TTL]
                for k in stale_k:
                    _lidar_pending.pop(k, None)
                _lidar_pkt_buf[:] = [p for p in _lidar_pkt_buf
                                     if now - p["t"] <= LIDAR_BUFFER_TTL]
                stale_seen = [t for t, seen_at in _lidar_seen.items()
                              if now - seen_at > LIDAR_DEDUP_TTL]
                for t in stale_seen:
                    _lidar_seen.pop(t, None)
        except Exception as e:
            log.warning("[lidar_udp] rx error: %s", e)


threading.Thread(target=_lidar_udp_listener, daemon=True,
                 name="lidar_udp").start()


def _save_synthetic_event(pkt: dict, hires_buf, event_count: int,
                          preview_state) -> None:
    """Save a synthetic (LiDAR-only) event: two frames bracketing the UDP packet
    arrival frame.  event_type='lidar', no background blur (bbox=None).
    LiDAR fields are pre-filled in event_meta so _lidar_register skips the
    match search and writes the CSV row immediately."""
    target_fi = pkt["fi"]
    to_save   = _select_frames(hires_buf, target_fi)
    if to_save is None:
        log.warning("[lidar_synth] event %d: fi=%d not in hires_buf — skipping",
                    event_count, target_fi)
        return

    frames_to_save = [(bgr, stem, None, meta) for bgr, stem, meta, entry_fi in to_save]

    # Pre-fill lidar fields; _lidar_register detects lidar_d already set and
    # skips the match search, writing the CSV row directly.
    event_meta = {
        'blob_width':   0,
        'event_frames': 0,
        'event_type':   'lidar',
        'velocity':     None,
        '_lidar_prefill': {
            "t": pkt["t"], "dur": pkt["dur"], "d": pkt["d"],
        },
    }
    log.info("[lidar_synth] event %d  fi=%s  d=%.3f m  dur=%d ms",
             event_count, [e[3] for e in to_save], pkt["d"], pkt["dur"])
    start_save(frames_to_save, event_count, rightward=True,
               event_meta=event_meta, preview_state=preview_state)

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

def _build_exif(meta: dict, bbox_lores=None):
    """Build a Pillow ExifData object from libcamera metadata.

    Tags that are SRATIONAL (signed rational) -- ShutterSpeedValue, BrightnessValue --
    must go into the Exif sub-IFD (0x8769) via get_ifd(), not the top-level dict,
    because Pillow encodes top-level rationals as unsigned RATIONAL.

    bbox_lores: optional (bx, by, bw, bh) in lores coordinates (ROI-relative y).
    Converted to hires-cropped pixel coordinates and stored in ImageDescription as
    BBox=x1,y1,x2,y2 where (x1,y1) is top-left and (x2,y2) is bottom-right of the
    vehicle in the saved JPEG's coordinate space.
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

    # ImageDescription: human-readable lux and bounding box for viewers that show IFD0 only.
    # BBox is in the saved JPEG's coordinate space (hires, vertically cropped by HIRES_CROP_TOP).
    desc_parts = []
    if lux is not None:
        desc_parts.append(f"Lux={lux:.1f}")
    if bbox_lores is not None:
        bx, by, bw, bh = bbox_lores
        hx1 = int(bx * LORES_TO_HIRES_X)
        hx2 = int((bx + bw) * LORES_TO_HIRES_X)
        hy1 = int((by + ROI_TOP) * LORES_TO_HIRES_Y) - HIRES_CROP_TOP
        hy2 = int((by + bh + ROI_TOP) * LORES_TO_HIRES_Y) - HIRES_CROP_TOP
        hx1 = max(0, hx1);  hx2 = min(HIRES_SIZE[0], hx2)
        hy1 = max(0, hy1);  hy2 = min(HIRES_CROP_BOTTOM - HIRES_CROP_TOP, hy2)
        desc_parts.append(f"BBox={hx1},{hy1},{hx2},{hy2}")
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


# ---------------------------------------------------------------------------
# Optics: pixel-width → real-world length and speed
# ---------------------------------------------------------------------------
# IMX296 sensor active width: 1456 px * 0.00345 mm/px = 5.02 mm
# Lores width: 320 px spans the full sensor width
# Lens focal length: 12.0 mm
# Camera-to-road offset: lidar reports distance to vehicle; add ROAD_OFFSET_M
# to get the true object distance u (in mm for thin-lens formula).
SENSOR_WIDTH_MM  = 1456 * 0.00345   # 5.0232 mm
LORES_WIDTH_PX   = 320
FOCAL_MM         = 12.0
ROAD_OFFSET_M    = 11.0             # metres from camera to lidar beam path

def _compute_length_and_speed(blob_width_px: float, lidar_d_m: float,
                               lidar_dur_ms: int, velocity_px_fr):
    """Return (length_m, speed_mph) using thin-lens optics, or (None, None)
    if any required input is missing/zero."""
    if not blob_width_px or not lidar_d_m or not lidar_dur_ms:
        return None, None
    u_mm = (lidar_d_m + ROAD_OFFSET_M) * 1000.0
    # Thin-lens: 1/v = 1/f - 1/u  →  v = f*u / (u - f)
    v_mm = (FOCAL_MM * u_mm) / (u_mm - FOCAL_MM)
    mag  = v_mm / u_mm
    img_mm  = (blob_width_px / LORES_WIDTH_PX) * SENSOR_WIDTH_MM
    length_m = (img_mm / mag) / 1000.0
    t_s      = lidar_dur_ms / 1000.0
    speed_ms = length_m / t_s
    speed_mph = speed_ms * 2.23694
    # Apply direction sign from camera velocity (positive = rightward)
    if velocity_px_fr is not None and velocity_px_fr < 0:
        speed_mph = -speed_mph
    return round(length_m, 2), round(speed_mph, 1)


CSV_HEADER = ("event,time,epoch,width(px),frames,type,velocity(px/fr),"
              "lux,shutter(us),lidar_d(m),lidar_dur(ms),lidar_dt(ms),"
              "length(m),speed_L(mph),speed_C(mph)\n")

# Seconds to wait for a LiDAR match before writing the CSV row without lidar data
CSV_LIDAR_WAIT = 4.0   # seconds to wait for lidar match before writing CSV without it
                       # must exceed LIDAR_MATCH_LATE plus worst-case UDP transit delay


def _compute_vel_mph(velocity_px_fr, lidar_d_m):
    """Convert camera velocity (lores px/frame) to mph using optics.
    Returns None if either input is missing."""
    if velocity_px_fr is None or lidar_d_m is None:
        return None
    u_mm    = (lidar_d_m + ROAD_OFFSET_M) * 1000.0
    v_mm    = (FOCAL_MM * u_mm) / (u_mm - FOCAL_MM)
    mag     = v_mm / u_mm
    mm_per_px = (SENSOR_WIDTH_MM / LORES_WIDTH_PX) / mag
    vel_ms  = velocity_px_fr * FRAME_RATE * mm_per_px / 1000.0
    return round(vel_ms * 2.23694, 1)


def _append_csv_log(event_count, time_str, epoch, blob_width,
                    event_frames, event_type, velocity, date_str,
                    lux=None, shutter_us=None,
                    lidar_d=None, lidar_dur=None, lidar_dt=None,
                    length_m=None, speed_mph=None, vel_mph=None):
    """
    Append one row to /home/pi/CAMA/YYYYMMDD_log.csv.
    Creates the file (with header) if it does not yet exist.
    date_str is 'YYYYMMDD', derived from the capture timestamp.
    lidar_dt is in seconds internally; stored in CSV as milliseconds.
    speed_mph = speed derived from lidar duration (L); vel_mph from camera velocity (C).
    """
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"{date_str}_log.csv"
    write_header = not log_path.exists()
    vel_str       = f"{velocity:.4f}"        if velocity   is not None else ""
    lux_str       = f"{lux:.1f}"             if lux        is not None else ""
    shutter_str   = f"{shutter_us:.0f}"      if shutter_us is not None else ""
    lidar_d_str   = f"{lidar_d:.3f}"         if lidar_d    is not None else ""
    lidar_dur_str = f"{lidar_dur}"            if lidar_dur  is not None else ""
    lidar_dt_str  = f"{lidar_dt * 1000:.0f}" if lidar_dt   is not None else ""
    length_str    = f"{length_m:.2f}"        if length_m   is not None else ""
    speed_str     = f"{speed_mph:.1f}"       if speed_mph  is not None else ""
    vel_mph_str   = f"{vel_mph:.1f}"         if vel_mph    is not None else ""
    row = (f"{event_count},{time_str},{epoch:.3f},"
           f"{blob_width},{event_frames},{event_type},{vel_str},"
           f"{lux_str},{shutter_str},"
           f"{lidar_d_str},{lidar_dur_str},{lidar_dt_str},"
           f"{length_str},{speed_str},{vel_mph_str}\n")
    try:
        with log_path.open('a') as f:
            if write_header:
                f.write(CSV_HEADER)
            f.write(row)
    except Exception as e:
        log.warning("[csv] failed to write log: %s", e)


def _write_csv_for_event(info: dict):
    """Write the CSV row for a completed event, including any lidar data now in info."""
    length_m, speed_mph = _compute_length_and_speed(
        info.get('blob_width'), info.get('lidar_d'),
        info.get('lidar_dur'), info.get('velocity'))
    vel_mph = _compute_vel_mph(info.get('velocity'), info.get('lidar_d'))
    _append_csv_log(
        info['event_count'], info['time_str'], info['epoch'],
        info['blob_width'], info['event_frames'], info['event_type'],
        info.get('velocity'), info['date_str'],
        lux=info.get('lux'), shutter_us=info.get('shutter_us'),
        lidar_d=info.get('lidar_d'), lidar_dur=info.get('lidar_dur'),
        lidar_dt=info.get('lidar_dt'),
        length_m=length_m, speed_mph=speed_mph, vel_mph=vel_mph)


def _fit_velocity(history):
    """Compute signed velocity (lores px/frame) by linear regression.
    Prefers fully-interior (edges=..) frames when at least 4 are available,
    falling back to the full history otherwise.  Returns None if too few points."""
    interior = [(fi, cx) for fi, cx, full, *_ in history if full]
    pts = interior if len(interior) >= MIN_INTERIOR_VEL_FRAMES else [(fi, cx) for fi, cx, *_ in history]
    if len(pts) < 4:
        return None
    _cx_val = np.array([cx for _, cx in pts], dtype=np.float32)
    _cx_idx = np.arange(len(_cx_val), dtype=np.float32)
    return float(np.polyfit(_cx_idx, _cx_val, 1)[0])


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
    for hires_bgr, stem, bbox_lores, meta in frames:
        fname = f"{stem}_{event_count}.jpg"
        path  = SHM_BASE / fname
        processed = blur_background(hires_bgr, union_bbox, rightward)
        cropped   = processed[HIRES_CROP_TOP:HIRES_CROP_BOTTOM, :]
        # Convert BGR→RGB for PIL, build EXIF, save as JPEG.
        # Blur uses union_bbox (covers both frames); EXIF embeds per-frame bbox_lores.
        rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        pil_img.save(str(path), format="JPEG", quality=JPEG_QUALITY,
                     exif=_build_exif(meta, bbox_lores) if meta else None)
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
                # Build info dict with all data needed for CSV and web display.
                # CSV write is deferred to _lidar_register / _write_csv_for_event
                # so lidar fields can be included if a match arrives within CSV_LIDAR_WAIT.
                info = dict(event_meta)
                info['time_str']    = time_str
                info['event_count'] = event_count
                info['epoch']       = epoch
                info['date_str']    = date_str
                info['lux']         = meta.get('Lux') if meta else None
                info['shutter_us']  = meta.get('ExposureTime') if meta else None
                info['lidar_t']     = None
                info['lidar_dur']   = None
                info['lidar_d']     = None
                info['lidar_dt']    = None
                info.setdefault('blob_x', None)
                if _preview_state is not None:
                    with _preview_state.lock:
                        _preview_state.event_history = (
                            [info] + _preview_state.event_history)[:5]
                _lidar_register(info)
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
    focus_jpeg:    bytes | None   = None   # latest 320px center crop of hires frame
    focus_mode:    bool           = False  # True → /stream serves focus_jpeg
    blob:          dict | None    = None   # latest blob dict (or None between events)
    fg_mask:       object         = None   # latest fg_mask ndarray (ROI coords)
    event_history: list           = dataclasses.field(default_factory=list)  # up to 5 recent events, newest first

# Horizontal offset for 320 px center crop of the 1456 px wide hires frame
_FOCUS_CROP_X = (HIRES_SIZE[0] - LORES_SIZE[0]) // 2   # = 568


_OVERLAY_COLOR = (0, 200, 0)    # green tint for motion mask pixels  (BGR)
_OVERLAY_ALPHA = 0.45
_BBOX_COLOR    = (0, 220, 255)  # cyan-yellow box                     (BGR)
_BBOX_THICK    = 2
_PREVIEW_SCALE = 1              # upscale factor (1 = native lores resolution)
_PREVIEW_PORT  = 8080


def _compose_preview(color_frame, fg_mask, blob) -> bytes:
    """
    Build the annotated preview JPEG:
      - lores color frame cropped to LORES_CROP_TOP..LORES_CROP_BOTTOM rows
      - green tint where fg_mask is nonzero (ROI band only)
      - bounding box if blob is not None
    Returns JPEG bytes.
    """
    vis = color_frame.copy()

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

    # Crop to the rows actually used by the motion detection algorithm
    vis = vis[LORES_CROP_TOP:LORES_CROP_BOTTOM, :]
    h, w = vis.shape[:2]

    # 2x nearest-neighbour upscale
    vis = cv2.resize(vis, (w * _PREVIEW_SCALE, h * _PREVIEW_SCALE),
                     interpolation=cv2.INTER_NEAREST)

    ok, buf = cv2.imencode('.jpg', vis, [cv2.IMWRITE_JPEG_QUALITY, 75])
    return buf.tobytes() if ok else b''


def _preview_writer(state: PreviewState, color_frame, fg_mask, hires_bgr=None):
    """Called from the main loop at ~4 fps. Reads current blob from state."""
    with state.lock:
        blob = state.blob
    jpeg = _compose_preview(color_frame, fg_mask, blob)
    with state.lock:
        state.jpeg = jpeg
        if hires_bgr is not None:
            # 320 px center crop of full hires frame, vertically cropped same as saves
            crop = hires_bgr[HIRES_CROP_TOP:HIRES_CROP_BOTTOM,
                             _FOCUS_CROP_X:_FOCUS_CROP_X + LORES_SIZE[0]]
            ok, buf = cv2.imencode('.jpg', crop, [cv2.IMWRITE_JPEG_QUALITY, 80])
            state.focus_jpeg = buf.tobytes() if ok else state.focus_jpeg


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
            '#thumb-wrap{position:relative;display:inline-block}'
            '#thumb{opacity:0.85;display:block;max-width:100%}'
            '#overlay{position:absolute;top:0;left:0;width:100%;height:100%;'
            'pointer-events:none}'
            '#info{color:#ccc;font-family:monospace;font-size:14px;'
            'background:#1e1e1e;padding:8px 16px;border-radius:4px;'
            'white-space:pre;letter-spacing:0.03em}'
            '#bbox-btn{font-family:monospace;font-size:13px;padding:4px 12px;'
            'background:#2a2a2a;color:#ccc;border:1px solid #555;'
            'border-radius:4px;cursor:pointer}'
            '#bbox-btn.on{background:#1a4a1a;color:#6f6;border-color:#4a4}'
            '#focus-btn{font-family:monospace;font-size:13px;padding:4px 12px;'
            'background:#2a2a2a;color:#ccc;border:1px solid #555;'
            'border-radius:4px;cursor:pointer}'
            '#focus-btn.on{background:#1a3a4a;color:#6cf;border-color:#4ac}'
            '</style></head>'
            '<body>'
            '<img src="/stream">'
            '<div style="display:flex;gap:8px">'
            '<button id="focus-btn" onclick="toggleFocus()">focus: OFF</button>'
            '<button id="bbox-btn" onclick="toggleBbox()">bbox overlay: OFF</button>'
            '</div>'
            '<div id="info">—</div>'
            '<div id="thumb-wrap">'
            '  <img id="thumb" src="/thumb">'
            '  <canvas id="overlay"></canvas>'
            '</div>'
            '<script>'
            f'var FRAME_RATE={FRAME_RATE};'
            f'var FOCAL_MM={FOCAL_MM};'
            f'var SENSOR_WIDTH_MM={SENSOR_WIDTH_MM};'
            f'var LORES_WIDTH_PX={LORES_WIDTH_PX};'
            f'var ROAD_OFFSET_M={ROAD_OFFSET_M};'
            f'var LORES_TO_HIRES_X={LORES_TO_HIRES_X};'
            'var bboxOn=false;'
            'var lastEvent=null;'
            'function toggleFocus(){'
            '  fetch("/focus",{method:"POST"}).then(r=>r.json()).then(function(d){'
            '    var btn=document.getElementById("focus-btn");'
            '    btn.textContent="focus: "+(d.focus?"ON":"OFF");'
            '    if(d.focus) btn.classList.add("on"); else btn.classList.remove("on");'
            '  }).catch(()=>{});'
            '}'
            'function toggleBbox(){'
            '  bboxOn=!bboxOn;'
            '  var btn=document.getElementById("bbox-btn");'
            '  btn.textContent="bbox overlay: "+(bboxOn?"ON":"OFF");'
            '  if(bboxOn) btn.classList.add("on"); else btn.classList.remove("on");'
            '  drawOverlay();'
            '}'
            'function drawOverlay(){'
            '  var thumb=document.getElementById("thumb");'
            '  var canvas=document.getElementById("overlay");'
            '  canvas.width=thumb.naturalWidth||thumb.offsetWidth;'
            '  canvas.height=thumb.naturalHeight||thumb.offsetHeight;'
            '  var ctx=canvas.getContext("2d");'
            '  ctx.clearRect(0,0,canvas.width,canvas.height);'
            '  if(!bboxOn||!lastEvent||lastEvent.blob_x===null||lastEvent.blob_x===undefined){return;}'
            '  var d=lastEvent;'
            '  var scale=LORES_TO_HIRES_X/2;'
            '  var x1=Math.round(d.blob_x*scale);'
            '  var x2=Math.round((d.blob_x+d.blob_width)*scale);'
            '  ctx.strokeStyle="rgba(255,80,80,0.9)";'
            '  ctx.lineWidth=2;'
            '  ctx.beginPath();ctx.moveTo(x1,0);ctx.lineTo(x1,canvas.height);ctx.stroke();'
            '  ctx.beginPath();ctx.moveTo(x2,0);ctx.lineTo(x2,canvas.height);ctx.stroke();'
            '}'
            'function refreshInfo(){'
            '  fetch("/info").then(r=>r.json()).then(arr=>{'
            '    if(!arr.length){return;}'
            '    lastEvent=arr[0];'
            '    var lines=arr.slice().reverse().map(function(d){'
            '      var lidar = "";'
            '      var wStr = d.blob_width+"px";'
            '      var velStr = (d.velocity!==null?(d.velocity>=0?"+":"")+d.velocity.toFixed(2):"?")+"px/fr";'
            '      var spdStr = "";'
            '      if(d.lidar_d !== null && d.lidar_d !== undefined){'
            '        var dtSign = d.lidar_dt >= 0 ? "+" : "";'
            '        lidar = "  lidar:" + d.lidar_d.toFixed(2) + "m"'
            '               + " " + d.lidar_dur + "ms"'
            '               + " dt=" + dtSign + d.lidar_dt.toFixed(3) + "s";'
            '        var u_mm = (d.lidar_d + ROAD_OFFSET_M) * 1000.0;'
            '        var v_mm = (FOCAL_MM * u_mm) / (u_mm - FOCAL_MM);'
            '        var mag  = v_mm / u_mm;'
            '        var img_mm = (d.blob_width / LORES_WIDTH_PX) * SENSOR_WIDTH_MM;'
            '        var len_m = (img_mm / mag) / 1000.0;'
            '        wStr = len_m.toFixed(2) + "m";'
            '        var spd_ms_lidar = len_m / (d.lidar_dur / 1000.0);'
            '        var spd_mph = spd_ms_lidar * 2.23694;'
            '        if(d.velocity !== null && d.velocity < 0) spd_mph = -spd_mph;'
            '        var spdSign = spd_mph >= 0 ? "+" : "";'
            '        spdStr = "  " + spdSign + spd_mph.toFixed(1) + "mph(L)";'
            '        if(d.velocity !== null){'
            '          var mm_per_px = (SENSOR_WIDTH_MM / LORES_WIDTH_PX) / mag;'
            '          var vel_ms = d.velocity * FRAME_RATE * mm_per_px / 1000.0;'
            '          var vel_mph = vel_ms * 2.23694;'
            '          var velSign = vel_mph >= 0 ? "+" : "";'
            '          velStr = velSign + vel_mph.toFixed(1) + "mph(C)";'
            '        }'
            '      } else {'
            '        lidar = "  lidar:--";'
            '      }'
            '      return "#"+d.event_count'
            '        +"  "+d.time_str'
            '        +"  w="+wStr'
            '        +"  type="+d.event_type'
            '        +"  vel="+velStr'
            '        +lidar'
            '        +spdStr;'
            '    });'
            '    document.getElementById("info").textContent=lines.join("\\n");'
            '    drawOverlay();'
            '  }).catch(()=>{});'
            '}'
            'setInterval(function(){'
            '  var t=document.getElementById("thumb");'
            '  t.src="/thumb?t="+Date.now();'
            '  refreshInfo();'
            '},5000);'
            'document.getElementById("thumb").onload=function(){drawOverlay();};'
            'refreshInfo();'
            '</script>'
            '</body></html>',
            mimetype='text/html')

    def _gen():
        while True:
            with state.lock:
                frame = state.focus_jpeg if state.focus_mode else state.jpeg
            if frame:
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'
                       + frame + b'\r\n')
            time.sleep(0.25)   # poll at 20 Hz; image itself updates at ~4 fps

    @app.route('/focus', methods=['POST'])
    def focus():
        import json as _json
        with state.lock:
            state.focus_mode = not state.focus_mode
            mode = state.focus_mode
        return Response(_json.dumps({'focus': mode}), mimetype='application/json')

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


def is_pedestrian_event(vehicle_frame_indices, fi_bbox):
    """Return True if the current event looks like a pedestrian rather than a vehicle.

    Criteria (both must hold):
      1. Median blob width over all event frames is below PEDESTRIAN_MAX_W.
      2. The cx centroid progression is predominantly monotonic: at least
         PEDESTRIAN_MIN_MONO_FRAC of consecutive-frame cx steps share the same sign
         (i.e. the object moves steadily in one direction across the frame).

    fi_bbox maps fi → (bx, by, bw, bh) in lores coordinates; cx is approximated
    as bx + bw/2 from the stored bounding box.
    """
    if len(vehicle_frame_indices) < 4:
        return False

    sorted_fi = sorted(vehicle_frame_indices)
    widths = []
    cxs    = []
    for fi in sorted_fi:
        bbox = fi_bbox.get(fi)
        if bbox is not None:
            bx, by, bw, bh = bbox
            widths.append(bw)
            cxs.append(bx + bw / 2.0)

    if not widths:
        return False

    # Criterion 1: narrow blob throughout the event
    if float(np.median(widths)) >= PEDESTRIAN_MAX_W:
        return False

    # Criterion 2: monotonic cx progression
    if len(cxs) < 2:
        return False
    steps = [cxs[i] - cxs[i - 1] for i in range(1, len(cxs))]
    pos = sum(1 for s in steps if s > 0)
    neg = sum(1 for s in steps if s < 0)
    dominant = max(pos, neg)
    if dominant / len(steps) < PEDESTRIAN_MIN_MONO_FRAC:
        return False

    return True


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
        return None, None, None

    component_mask = (labels == best_label).astype(np.uint8) * 255
    pts = cv2.findNonZero(component_mask)

    if pts is None or len(pts) < 3:
        return None, None, None

    hull   = cv2.convexHull(pts)
    hx, hy, hw, hh = cv2.boundingRect(hull)
    area   = cv2.contourArea(hull)
    aspect = hw / max(hh, 1)
    fill   = area / max(hw * hh, 1)

    m = cv2.moments(hull)
    cx = (m['m10'] / m['m00']) if m['m00'] != 0 else hx + hw / 2.0
    cy = (m['m01'] / m['m00']) if m['m00'] != 0 else hy + hh / 2.0

    if area < MIN_BLOB_AREA:   return None, None, None
    if hw   < MIN_BLOB_WIDTH:  return None, None, None

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
        return None, reason, None

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
    }, None, component_mask


def estimate_center_frame(centroid_history):
    """
    Linear fit of centroid_x vs frame_index.
    Returns (predicted_fi: float, used_interior: bool) or (None, False).
    centroid_history entries are (fi, cx, fully_interior, ev) 4-tuples.
    """
    interior = [(fi, cx) for fi, cx, full, *_ in centroid_history if full]
    subset   = interior if len(interior) >= 4 else \
               [(fi, cx) for fi, cx, *_ in centroid_history] \
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
    Direction of travel is inferred from the median of per-frame cx differences,
    which is robust against a single AGC artifact at the tail that would
    otherwise flip the first-vs-last direction estimate.
    """
    if len(centroid_history) < 4:
        return centroid_history, []

    cx_vals = [cx for _, cx, *_ in centroid_history]
    # Use median of per-frame steps to determine direction; a single outlier
    # frame (e.g. AGC transient at the tail) cannot flip the sign this way.
    steps_all = [cx_vals[i] - cx_vals[i - 1] for i in range(1, len(cx_vals))]
    direction = 1 if float(np.median(steps_all)) >= 0 else -1  # +1 rightward

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

    phase_fis = [fi for fi, *_ in phase_history]
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
    _chosen_bbox = fi_bbox.get(chosen_fi)
    _chosen_w = _chosen_bbox[2] if _chosen_bbox is not None else None
    log.info("  [%s] saving fi=%s  (chosen_fi=%d  pred=%s  by=%s  lag=%d  chosen_w=%s)",
             label, [e[3] for e in to_save], chosen_fi,
             f"{pred:.1f}" if pred is not None else "–", chosen_by, lag,
             str(_chosen_w) if _chosen_w is not None else "?")
    # Use bbox of the first saved frame (the thumbnail frame) for the web overlay
    # so blob_x/blob_width match what is actually displayed.
    _thumb_bbox = fi_bbox.get(to_save[0][3])
    _overlay_bbox = _thumb_bbox if _thumb_bbox is not None else _chosen_bbox
    if event_meta is not None and _overlay_bbox is not None:
        event_meta = dict(event_meta)   # don't mutate caller's dict
        event_meta['blob_width'] = _overlay_bbox[2]
        event_meta['blob_x']     = _overlay_bbox[0]
    elif event_meta is not None and _chosen_w is not None:
        event_meta = dict(event_meta)
        event_meta['blob_width'] = _chosen_w
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
        _thumb_bbox = fi_bbox.get(to_save[0][3])
        _em = dict(event_meta) if event_meta is not None else {}
        if _thumb_bbox is not None:
            _em['blob_width'] = _thumb_bbox[2]
            _em['blob_x']     = _thumb_bbox[0]
        frames_to_save = [(bgr, stem, fi_bbox.get(entry_fi), meta)
                          for bgr, stem, meta, entry_fi in to_save]
        start_save(frames_to_save, event_count, rightward,
                   event_meta=_em, preview_state=preview_state,
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
    h_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (MORPH_HOPEN_W, MORPH_HOPEN_H))

    # --- State ---------------------------------------------------------------
    # Rolling buffer of (hires_bgr, stem, meta, fi) — most recent HIRES_BUFFER_SIZE frames
    hires_buf: collections.deque = collections.deque(maxlen=HIRES_BUFFER_SIZE)

    # Metadata ring buffer kept for EXIF writing only (no delay compensation needed)
    consecutive           = 0
    vehicle_active        = False
    lockout_remaining     = 0
    event_frames          = 0      # frames since vehicle_active became True
    centroid_history      = []     # (fi, cx, fully_interior, ev)
    vehicle_frame_indices = []     # fi of every VEHICLE-tagged frame
    vehicle_edge_frames   = []     # (fi, touches_left, touches_right) for same frames
    fi_bbox               = {}     # fi → (bx, by, bw, bh) lores bbox for event frames
    fi_mask               = {}     # stem → fg_mask ndarray (ROI-relative) for event frames
    best_center_idx       = None   # global frame index
    best_center_dist      = float('inf')
    best_center_cx        = None
    max_blob_width        = 0      # max hull width seen during this event
    interior_blob_widths  = []     # widths from fully-interior (edges=..) confirmed frames
    frame_count           = 0
    fi                    = 0      # detection frame index (increments every frame)
    # Rolling EV history from non-vehicle frames; used as pre-event baseline
    # for EV-suspect frame detection.
    ev_pre_event: collections.deque = collections.deque(maxlen=EV_PRE_EVENT_WINDOW)
    # Per-event AGC tracking
    ev_event_baseline = 1.0    # median EV from ev_pre_event at event start
    ev_event_peak     = 1.0    # max EV seen during event
    ev_post_frames    = 0      # frames logged after event end for recovery tracking

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

        # Read EV from metadata — used for centroid history and EV-suspect trimming.
        # No longer used for pixel correction; edge input is illumination-invariant.
        exp = meta.get("ExposureTime") or 1.0
        ag  = meta.get("AnalogueGain")  or 1.0
        dg  = meta.get("DigitalGain")   or 1.0
        ev  = float(exp) * float(ag) * float(dg)

        # Update shared fi so UDP listener can record which frame is current
        global _current_fi
        _current_fi = fi

        # Check for LiDAR packets that timed out without a native event match
        # and promote them to synthetic events.
        now = time.time()
        synth_pkts = []
        with _lidar_lock:
            remaining = []
            for p in _lidar_pkt_buf:
                if now - p["wall_t"] >= LIDAR_SYNTH_WAIT:
                    synth_pkts.append(p)
                else:
                    remaining.append(p)
            _lidar_pkt_buf[:] = remaining
        for pkt in synth_pkts:
            _, _saved_date = _load_event_num()
            if _saved_date != datetime.now().strftime('%Y%m%d'):
                event_count = 0
            event_count += 1
            _save_event_num(event_count)
            log.info("[lidar_synth] no native event — promoting to synthetic event %d",
                     event_count)
            _save_synthetic_event(pkt, hires_buf, event_count, preview_state)

        # Store high-res frame in rolling buffer (includes fi for later selection)
        hires_buf.append((hires_bgr.copy(), stem, meta, fi))

        # Convert lores YUV420 to grayscale for edge detection.
        # Scharr gradient magnitude is illumination-normalized: a global brightness
        # scale factor (AGC step, cloud shadow) cancels out in the gradient, so
        # MOG2 sees a stable background without any EV correction machinery.
        buf_w  = lores_arr.shape[1]
        yuv    = lores_arr[:lores_valid_h * 3 // 2, :buf_w]
        lores_bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)
        lores_bgr = lores_bgr[:lores_valid_h, :lores_valid_w]

        gray = cv2.cvtColor(lores_bgr, cv2.COLOR_BGR2GRAY)
        if EDGE_BLUR_KSIZE > 0:
            gray = cv2.GaussianBlur(gray, (EDGE_BLUR_KSIZE, EDGE_BLUR_KSIZE), 0)
        sx = cv2.Scharr(gray, cv2.CV_32F, 1, 0)
        sy = cv2.Scharr(gray, cv2.CV_32F, 0, 1)
        # Encode signed gradient as uint8 centered on 128.
        # Preserves edge direction so MOG2 models the full joint (sx, sy)
        # distribution rather than just magnitude — smooth vehicle panels
        # (both channels near 128) are distinguishable from textured background
        # (high variance around 128) even when their mean magnitudes overlap.
        scale = 127.0 / EDGE_CLIP_MAX
        sx_u8 = np.clip(sx * scale + 128, 0, 255).astype(np.uint8)
        sy_u8 = np.clip(sy * scale + 128, 0, 255).astype(np.uint8)
        edge_2ch = np.stack([sx_u8, sy_u8], axis=2)  # shape (H, W, 2)

        # MOG2 input: 2-channel signed-gradient crop of ROI
        roi = edge_2ch[ROI_TOP:ROI_BOTTOM, :]
        # Preview overlay uses sx channel (horizontal edges most visible on vehicles)
        color = sx_u8

        # MOG2 learning rate — freeze during vehicle events only.
        # No EV correction or boost logic needed: edges are illumination-invariant.
        if vehicle_active:
            lr = LR_VEHICLE
        else:
            lr = LR_NORMAL

        # Track pre-event EV for EV-suspect tail trimming (retained as safety net)
        if not vehicle_active:
            ev_pre_event.append(ev)

        fg_mask = fgbg.apply(roi, learningRate=lr)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN,  kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        # Suppress horizontal ghost strips at vehicle roofline/road-edge boundaries.
        # These appear as thin horizontal bands in the already-vacated region behind
        # the vehicle; the wide short kernel removes them while preserving the taller
        # filled vehicle blob.
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN,  h_kernel)

        if args.verbose_fg:
            fg_count = cv2.countNonZero(fg_mask)
            if fg_count >= VERBOSE_FG_MIN_PIXELS:
                log.info("  fi=%d  fg_pixels=%d  ev=%.1f", fi, fg_count, ev)

        # Blob detection
        if lockout_remaining > 0:
            lockout_remaining -= 1
            blob          = None
            reject_reason = None
            with preview_state.lock:
                preview_state.blob = None
        else:
            blob, reject_reason, blob_mask = find_vehicle_blob(fg_mask)
            # if reject_reason:
            #    log.info("  fi=%d  REJECTED  %s", fi, reject_reason)
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
            fi_mask[stem] = blob_mask.copy() if (SAVE_MASK and blob_mask is not None) else None
            edges = ('L' if blob['touches_left'] else '.') + \
                    ('R' if blob['touches_right'] else '.')

            if blob['area'] >= MIN_CENTROID_AREA:
                centroid_history.append((fi, cx, blob['fully_interior'], ev))

            if consecutive >= MIN_CONSECUTIVE_FRAMES:
                if not vehicle_active:
                    log.info("VEHICLE START  frame=%d", frame_count)
                    ev_event_baseline = float(np.median(ev_pre_event)) if ev_pre_event else ev
                    ev_event_peak     = ev
                    ev_post_frames    = 0
                    log.info("  [agc] event start  exp=%.0f  ag=%.2f  dg=%.2f  ev=%.1f",
                             exp, ag, dg, ev)
                vehicle_active  = True
                event_frames   += 1
                vehicle_frame_indices.append(fi)
                vehicle_edge_frames.append((fi, blob['touches_left'], blob['touches_right']))
                max_blob_width  = max(max_blob_width, bw)
                if blob['fully_interior']:
                    interior_blob_widths.append(bw)
                log.info("  fi=%d  cx=%.1f  w=%d  h=%d  ar=%.2f  edges=%s"
                         "  exp=%.0f  ag=%.2f  ev=%.1f  ev_ratio=%.3f",
                         fi, cx, bw, bh, blob['aspect_ratio'], edges,
                         exp, ag, ev,
                         ev / ev_event_baseline if ev_event_baseline else 1.0)
                ev_event_peak = max(ev_event_peak, ev)

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

            # Force-close event if it runs too long.
            # Pedestrians (narrow blob, monotonic cx) get a doubled timeout so a
            # slow-walking person crossing the full frame is not prematurely closed.
            _ped = (vehicle_active
                    and is_pedestrian_event(vehicle_frame_indices, fi_bbox))
            _max_frames = MAX_EVENT_FRAMES * (PEDESTRIAN_TIMEOUT_MULT if _ped else 1)
            if vehicle_active and event_frames >= _max_frames:
                log.warning("VEHICLE TIMEOUT  frame=%d — forcing close%s",
                            frame_count, "  [pedestrian]" if _ped else "")
                blob = None   # fall through to close logic below
                # Re-adapt MOG2 quickly to current background
                for _ in range(10):
                    fgbg.apply(roi, learningRate=0.5)

        if not blob and vehicle_active:
            # --- Vehicle event ended -----------------------------------------

            # --- EV-suspect tail trimming ------------------------------------
            # While a vehicle is in frame, AGC shifts gradually in one direction
            # (down for bright vehicles, up for dark ones relative to background).
            # Those gradual frames are still valid detections.  What we want to
            # remove is the sudden EV *reversal* at the tail when the vehicle
            # exits and the camera snaps back toward the background exposure.
            #
            # Method: use signed EV ratio (ev/ev_ref) to track direction.
            # Compute the prevailing per-frame trend over the first half of the
            # event, then scan the second half for the first step whose sign
            # opposes that trend AND whose magnitude exceeds EV_SUSPECT_THRESHOLD.
            # Everything from that step onward is trimmed.  This handles both
            # bright vehicles (EV falling then snapping up on exit) and dark
            # vehicles (EV rising then snapping down on exit) symmetrically.
            if EV_SUSPECT_THRESHOLD > 0 and ev_pre_event and len(centroid_history) >= 4:
                ev_ref = float(np.median(ev_pre_event))
                if ev_ref > 0:
                    ratios = [_ev / ev_ref for _, _, _, _ev in centroid_history]
                    n_ch   = len(ratios)
                    steps  = [ratios[i] - ratios[i - 1] for i in range(1, n_ch)]
                    # Prevailing trend: median step over the first half
                    half       = max(1, n_ch // 2)
                    trend_sign = np.sign(float(np.median(steps[:half])))
                    # Scan second half for a step that reverses the trend sharply
                    trim_from = None
                    for _i in range(half, n_ch - 1):
                        step = steps[_i]           # step entering ratios[_i + 1]
                        if (trend_sign != 0
                                and np.sign(step) != trend_sign
                                and abs(step) > EV_SUSPECT_THRESHOLD):
                            trim_from = _i + 1
                            break
                    if trim_from is not None:
                        n_trimmed = n_ch - trim_from
                        jump_pct  = abs(steps[trim_from - 1]) * 100
                        trim_fi   = centroid_history[trim_from][0]
                        del centroid_history[trim_from:]
                        log.info("  [ev_trim] removed %d EV-suspect tail frame(s) "
                                 "from fi=%d (ev_ref=%.0f  reversal=%.0f%%)",
                                 n_trimmed, trim_fi, ev_ref, jump_pct)

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
                cx_p = [cx for _, cx, *_ in phase]
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
            cx_vals = [cx for _, cx, *_ in eval_history]
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
            confirmed_cx = [cx for fi, cx, *_ in eval_history if fi in confirmed_fi_set]
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

            log.info("  [agc] event end    exp=%.0f  ag=%.2f  dg=%.2f  ev=%.1f"
                     "  baseline=%.1f  peak=%.1f  peak_ratio=%.3f",
                     exp, ag, dg, ev,
                     ev_event_baseline, ev_event_peak,
                     ev_event_peak / ev_event_baseline if ev_event_baseline else 1.0)
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

            elif event_frames < MIN_EVENT_FRAMES:
                log.warning("VEHICLE DISCARDED  event_frames=%d < MIN_EVENT_FRAMES=%d — "
                            "too short; resetting background model",
                            event_frames, MIN_EVENT_FRAMES)
                for _ in range(20):
                    fgbg.apply(roi, learningRate=0.5)

            elif chosen_fi is not None and hires_buf:
                # Use median of fully-interior frame widths to classify vehicle length.
                # A single AGC-blown edge-spanning frame cannot inflate this metric.
                # Fall back to max_blob_width only when no interior frames were seen.
                if interior_blob_widths:
                    robust_width = float(np.median(interior_blob_widths))
                else:
                    robust_width = float(max_blob_width)
                is_long = robust_width >= HIRES_LONG_VEHICLE_WIDTH
                if is_long != (max_blob_width >= HIRES_LONG_VEHICLE_WIDTH):
                    log.info("  [robust_width] median_interior=%.0f  max=%d  "
                             "overrides long classification → is_long=%s",
                             robust_width, max_blob_width, is_long)

                _, _last_date = _load_event_num()
                if _last_date != datetime.now().strftime('%Y%m%d'):
                    event_count = 0
                event_count += 1
                _save_event_num(event_count)

                # Compute signed velocity (lores px/frame) from eval_history.
                # Prefers interior-only frames; falls back to full history.
                # (phase1 only when phase1_only, else full centroid history.)
                _velocity = _fit_velocity(eval_history)

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
                        _chosen_bbox = fi_bbox.get(chosen_fi)
                        _thumb_bbox  = fi_bbox.get(to_save[0][3])
                        _overlay_bbox = _thumb_bbox if _thumb_bbox is not None else _chosen_bbox
                        if _overlay_bbox is not None:
                            _event_meta['blob_width'] = _overlay_bbox[2]
                            _event_meta['blob_x']     = _overlay_bbox[0]
                        elif _chosen_bbox is not None:
                            _event_meta['blob_width'] = _chosen_bbox[2]
                        log.info("  saving fi=%s  (chosen_fi=%d  lag=%d  chosen_w=%d  max_w=%d)",
                                 [e[3] for e in to_save], chosen_fi, lag,
                                 _event_meta['blob_width'], max_blob_width)
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
                else:
                    if interior_blob_widths:
                        robust_width = float(np.median(interior_blob_widths))
                    else:
                        robust_width = float(max_blob_width)
                    is_long = robust_width >= HIRES_LONG_VEHICLE_WIDTH
                    if is_long != (max_blob_width >= HIRES_LONG_VEHICLE_WIDTH):
                        log.info("  [robust_width] median_interior=%.0f  max=%d  "
                                 "overrides long classification → is_long=%s",
                                 robust_width, max_blob_width, is_long)
                    _, _last_date = _load_event_num()
                    if _last_date != datetime.now().strftime('%Y%m%d'):
                        event_count = 0
                    event_count += 1
                    _save_event_num(event_count)

                    # Compute signed velocity using interior-only frames when available.
                    _velocity = _fit_velocity(eval_history)

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
                            # Use the first saved frame's bbox width as the event width
                            _mid_bbox = fi_bbox.get(to_save[0][3])
                            if _mid_bbox is not None:
                                _event_meta['blob_width'] = _mid_bbox[2]
                                _event_meta['blob_x']     = _mid_bbox[0]
                            log.info("  saving fi=%s  (mid_fi=%.1f  no interior frames  chosen_w=%d)",
                                     [e[3] for e in to_save], mid_fi, _event_meta['blob_width'])
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
            interior_blob_widths  = []
            lockout_remaining     = LOCKOUT_FRAMES
            ev_post_frames        = EV_POST_EVENT_LOG_FRAMES

        elif not blob:
            # No event, no vehicle: reset consecutive counter
            consecutive  = 0
            event_frames = 0
            # Log EV recovery for POST_EVENT_LOG_FRAMES frames after an event ends
            if ev_post_frames > 0:
                log.info("  [agc] post-event +%d  exp=%.0f  ag=%.2f  dg=%.2f  ev=%.1f"
                         "  ev_ratio=%.3f",
                         EV_POST_EVENT_LOG_FRAMES - ev_post_frames + 1,
                         exp, ag, dg, ev,
                         ev / ev_event_baseline if ev_event_baseline else 1.0)
                ev_post_frames -= 1

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
            _preview_writer(preview_state, lores_bgr, fg_mask, hires_bgr)

        fi += 1

    picam2.stop()
    log.info("Done. %d events captured, %d frames processed.",
             event_count, frame_count)


if __name__ == "__main__":
    main()
