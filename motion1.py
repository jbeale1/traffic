#!/home/pi/cam-env/bin/python

# capture motion-triggered frames with PiCamera2
# using a low-res stream for motion detection
# sampled ~10 fps, saved ~3 fps using 2nd thread for JPEG encoding and disk I/O
# writes log entries for each motion event with timestamps, brightness, motion size, etc.
# pushes saved JPEGs to remote host via rsync as soon as they are written
# serves a decimated preview stream on port 8080 for live preview in a browser
# filters out noise/tree events by requiring minimum CoM displacement or velocity
# custom AeExposureMode limits longest shutter speed to 2 msec

# J.Beale 2026-04-26

import os
import glob
import json
import piexif
from PIL import Image
import cv2
import numpy as np
from picamera2 import Picamera2
from libcamera import ColorSpace, controls
import time
from datetime import datetime, timedelta
import threading
from collections import deque
import csv
import subprocess
import io
import socket
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import math

# get site-specific parameters
from config import (REMOTE_HOST, REMOTE_DIR, LOG_DIR, TUNING_FILE,
                    ROI_X1, ROI_Y1, ROI_X2, ROI_Y2,
                    THRESHOLD_DAY, THRESHOLD_NIGHT,
                    LATENCY_FIRST_MS, LATENCY_STEADY_MS,
                    MOTION_EXCLUDE_TOP, MOTION_EXCLUDE_BOTTOM, MIN_SAVE_INTERVAL_S)

TRANSFER_BUFFER_SIZE   = 64     # max queued filenames (each ~1.3 MB in /dev/shm)
TRANSFER_MAX_RETRIES   = 3      # give up after this many consecutive rsync failures
SAVE_BUFFER_SIZE       = 12
MAX_SHM_BACKLOG        = 40     # pause full-res saves if transfer queue exceeds this

EVENT_MAX_DURATION_S   = 8.0    # force-close an event after this many seconds
EVENT_MIN_DISPLACEMENT = 262    # lores px: min CoM range to keep event (half ROI width)
EVENT_MIN_VELOCITY     = 5.0    # lores px/frame: min |velocity| to keep event


SHM_DIR                = "/dev/shm/cam2"  # working directory for full-res JPEGs
SHM_STALE_HOURS        = 1               # delete leftover JPEGs older than this on startup
SD_SAVE_DIR            = "/home/pi/CAM2/saved"  # fallback destination when ethernet absent

def _eth_has_ip(iface="eth0"):
    """Return True if the given interface has an active IPv4 address."""
    try:
        result = subprocess.run(
            ["ip", "-4", "addr", "show", iface],
            capture_output=True, text=True, timeout=5
        )
        return "inet " in result.stdout
    except Exception:
        return False

LOCAL_SAVE_MODE = not _eth_has_ip("eth0")
if LOCAL_SAVE_MODE:
    print("[startup] No ethernet IP detected — saving to SD card, rsync disabled")
    os.makedirs(SD_SAVE_DIR, exist_ok=True)
else:
    print("[startup] Ethernet active — saving to /dev/shm and rsyncing as normal")


# Corresponding ROI in lores (640x480) coordinates
LORES_X1 = round(ROI_X1 * 640 / 4056)
LORES_Y1 = round(ROI_Y1 * 480 / 3040)
LORES_X2 = round(ROI_X2 * 640 / 4056)
LORES_Y2 = round(ROI_Y2 * 480 / 3040)

# Vertical exclusion zone within the lores ROI (leaf/sky noise at edges)
_roi_lores_h       = LORES_Y2 - LORES_Y1
EXCLUDE_TOP_ROWS   = round(_roi_lores_h * MOTION_EXCLUDE_TOP)
EXCLUDE_BOT_ROWS   = round(_roi_lores_h * MOTION_EXCLUDE_BOTTOM)

# Minimum connected blob size to count as real motion (filters leaf shimmer)
MIN_BLOB_PIXELS    = 25

# Focus-mode hardware ROI: central 1/3 of the normal ROI in sensor coordinates
_roi_w = ROI_X2 - ROI_X1
_roi_h = ROI_Y2 - ROI_Y1
FOCUS_ROI_X1 = ROI_X1 + _roi_w // 3
FOCUS_ROI_Y1 = ROI_Y1 + _roi_h // 3
FOCUS_ROI_X2 = ROI_X1 + _roi_w * 2 // 3
FOCUS_ROI_Y2 = ROI_Y1 + _roi_h * 2 // 3

# Day/night switching thresholds and parameters
MAX_SHUTTER_US     = 2000   # night mode fixed shutter (usec)
DAY_THRESHOLD      = 25     # bg brightness above this -> day candidate
NIGHT_THRESHOLD    = 20     # bg brightness below this -> night candidate
SWITCH_INTERVALS   = 3      # consecutive summary intervals to confirm mode switch
SUMMARY_INTERVAL   = 200    # frames per summary (~1 min at 10 fps)

SAVE_TIMING_LOG = False  # save timing logs per saved frame

# Pipeline latency corrections (sensor timestamp -> actual exposure start).
# The sensor timestamp marks end-of-pipeline delivery, not exposure start.
# First frame of each event uses a smaller correction (~40 ms less) due to
# pipeline buffering behaviour observed empirically.
# (LATENCY_FIRST_MS and LATENCY_STEADY_MS come from config)

# --- Ensure SHM_DIR exists and clean up stale files ---
os.makedirs(SHM_DIR, exist_ok=True)
_stale_cutoff = time.time() - SHM_STALE_HOURS * 3600
for _f in glob.glob(os.path.join(SHM_DIR, "*.jpg")):
    try:
        if os.path.getmtime(_f) < _stale_cutoff:
            os.remove(_f)
            print(f"[startup] removed stale file: {_f}")
    except OSError:
        pass

# Global state variables
verbose       = False
is_night_mode = False  # start in day mode
threshold     = THRESHOLD_DAY
prev_gray     = None
focus_mode      = False  # when True, preview shows central 1/3 of ROI
change_mode     = False  # when True, preview highlights motion pixels
motion_enabled  = True   # when False, motion events are detected but not saved
use_shm_mode    = False  # when True, override LOCAL_SAVE_MODE and push via rsync
_manual_capture_fname = ""   # basename of last manual capture, shown in status
_manual_capture_time  = 0.0  # wall time of last manual capture

bg_accum              = 0.0
bg_count              = 0
summary_frame         = 0
day_candidate_count   = 0
night_candidate_count = 0

# Event logging variables
in_motion_event     = False
event_t_start       = 0.0
event_frame_count   = 0
event_max_px        = 0
event_peak_fname    = ""
event_pre_px        = 0
event_last_px       = 0
_last_save_time     = 0.0    # wall time of last full-res capture
event_shutter_sum   = 0.0
event_gain_sum      = 0.0
event_bg_brightness = 0.0
event_com_x_list    = []
event_fnames        = []   # filenames saved during this event

_preview_last_fname = ""   # last motion-triggered filename
_preview_changed    = 0    # most recent motion pixel count
_preview_timestamp  = ""   # timestamp of last preview frame
_preview_seq        = 0    # increments each time a new JPEG is encoded
_preview_event_log  = deque()  # (event_end_time, frame_count) tuples
_preview_last_update_t = 0.0   # wall time when preview JPEG was last updated
_cpu_temp_filtered = 0.0   # exponentially smoothed CPU temperature (°C)


# ---------------------------------------------------------------------------
# Event filter and open/close helpers
# ---------------------------------------------------------------------------

def _event_passes_filter(com_x_list):
    """Return True if the event's CoM motion justifies keeping saved frames."""
    if len(com_x_list) < 2:
        return False
    displacement = max(com_x_list) - min(com_x_list)
    if displacement >= EVENT_MIN_DISPLACEMENT:
        return True
    velocity = abs(np.polyfit(range(len(com_x_list)), com_x_list, 1)[0])
    return velocity >= EVENT_MIN_VELOCITY


def _close_event(changed, is_cap=False):
    """Evaluate and close the current motion event.

    is_cap=True when force-closing due to the 5-second duration cap.
    """
    global in_motion_event, event_last_px

    in_motion_event = False
    duration        = time.time() - event_t_start
    keep            = _event_passes_filter(event_com_x_list)

    if keep:
        if not LOCAL_SAVE_MODE or use_shm_mode:
            with transfer_lock:
                for fn in event_fnames:
                    transfer_queue.append(fn)
            if event_fnames:
                transfer_event.set()
        with _preview_lock:
            _preview_event_log.append((time.time(), event_frame_count))
    else:
        for fn in event_fnames:
            try:
                os.remove(fn)
            except FileNotFoundError:
                # File not written yet by saver thread; mark for deferred deletion
                with _pending_delete_lock:
                    _pending_delete.add(fn)
            except OSError:
                pass

    write_event_log({
        "t_start":       event_t_start,
        "duration":      duration,
        "peak_fname":    event_peak_fname,
        "frame_count":   event_frame_count,
        "bg_brightness": event_bg_brightness,
        "max_motion_px": event_max_px,
        "pre_event_px":  event_pre_px,
        "post_event_px": changed,
        "mean_shutter":  event_shutter_sum / event_frame_count if event_frame_count else 0,
        "mean_gain":     event_gain_sum    / event_frame_count if event_frame_count else 0,
        "com_x_list":    list(event_com_x_list),
        "kept":          keep,
    })

    if not is_cap:
        event_last_px = changed


def _start_event():
    """Initialise event tracking state for a new event."""
    global in_motion_event, event_t_start, event_frame_count, event_max_px
    global event_peak_fname, event_pre_px, event_shutter_sum, event_gain_sum
    global event_bg_brightness, event_com_x_list, event_fnames

    in_motion_event     = True
    event_t_start       = time.time()
    event_frame_count   = 0
    event_max_px        = 0
    event_peak_fname    = ""
    event_pre_px        = event_last_px
    event_shutter_sum   = 0.0
    event_gain_sum      = 0.0
    event_bg_brightness = bg_accum / bg_count if bg_count > 0 else 0.0
    event_com_x_list    = []
    event_fnames        = []


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def write_event_log(evt):
    log_path = os.path.join(LOG_DIR, f"log_motion_{datetime.fromtimestamp(evt['t_start']).strftime('%Y%m%d')}.csv")
    is_new = not os.path.exists(log_path)

    com_list = evt['com_x_list']
    if len(com_list) >= 2:
        velocity = np.polyfit(range(len(com_list)), com_list, 1)[0]
    elif len(com_list) == 1:
        velocity = 0.0
    else:
        velocity = float('nan')

    displacement = (max(com_list) - min(com_list)) if com_list else 0.0

    with open(log_path, "a", newline="") as f:
        w = csv.writer(f)
        if is_new:
            w.writerow(["epoch_start", "time_hms", "peak_fname", "duration_s", "frame_count",
                        "bg_brightness", "max_motion_px", "pre_event_px", "post_event_px",
                        "mean_shutter_us", "mean_gain", "velocity_px_per_frame",
                        "displacement_px", "kept"])
        w.writerow([
            f"{evt['t_start']:.1f}",
            datetime.fromtimestamp(evt['t_start']).strftime('%H:%M:%S'),
            evt['peak_fname'],
            f"{evt['duration']:.1f}",
            evt['frame_count'],
            f"{evt['bg_brightness']:.1f}",
            evt['max_motion_px'],
            evt['pre_event_px'],
            evt['post_event_px'],
            f"{evt['mean_shutter']:.0f}",
            f"{evt['mean_gain']:.2f}",
            f"{velocity:.1f}",
            f"{displacement:.0f}",
            int(evt.get('kept', True)),
        ])

def apply_night_mode(cam):
    global threshold, is_night_mode
    threshold     = THRESHOLD_NIGHT
    is_night_mode = True
    cam.set_controls({
        "ExposureValue":    -2.0,
        "AeExposureMode":   controls.AeExposureModeEnum.Custom,
        "AnalogueGainMode": 0,
        "AwbMode":          controls.AwbModeEnum.Daylight,
    })

def apply_day_mode(cam):
    global threshold, is_night_mode
    threshold     = THRESHOLD_DAY
    is_night_mode = False
    cam.set_controls({
        "ExposureValue":    -2.0,
        "AeExposureMode":   controls.AeExposureModeEnum.Custom,
        "AnalogueGainMode": 0,
        "AwbMode":          controls.AwbModeEnum.Daylight,
    })

# --- Saver thread ---
save_queue = deque(maxlen=SAVE_BUFFER_SIZE)
save_event = threading.Event()
save_lock  = threading.Lock()

def saver_thread():
    while True:
        save_event.wait()
        save_event.clear()
        while True:
            with save_lock:
                if not save_queue:
                    break
                frame, fname, exif_bytes = save_queue.popleft()
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            img.save(fname, "JPEG", quality=85, exif=exif_bytes)
            with _pending_delete_lock:
                if fname in _pending_delete:
                    _pending_delete.discard(fname)
                    try:
                        os.remove(fname)
                    except OSError:
                        pass

threading.Thread(target=saver_thread, daemon=True).start()

# --- Transfer thread ---
transfer_queue = deque(maxlen=TRANSFER_BUFFER_SIZE)
transfer_event = threading.Event()
transfer_lock  = threading.Lock()
_pending_delete = set()
_pending_delete_lock = threading.Lock()

def transfer_thread():
    retry_delay = 5   # seconds; doubles on each failure, caps at 60
    while True:
        transfer_event.wait()
        transfer_event.clear()
        while True:
            with transfer_lock:
                if not transfer_queue:
                    break
                fpath = transfer_queue.popleft()

            if not os.path.exists(fpath):
                retry_delay = 5
                continue

            result = subprocess.run(
                ["rsync", "--remove-source-files", "-q",
                 fpath, f"{REMOTE_HOST}:{REMOTE_DIR}"],
                capture_output=True
            )

            if result.returncode == 0:
                retry_delay = 5  # reset backoff on success
            else:
                err = result.stderr.decode().strip()
                print(f"[transfer] rsync failed, retrying in {retry_delay}s: {err}")
                with transfer_lock:
                    transfer_queue.appendleft(fpath)
                time.sleep(retry_delay)
                retry_delay = min(retry_delay * 2, 60)
                transfer_event.set()
                break  # re-enter wait loop

threading.Thread(target=transfer_thread, daemon=True).start()

# --- CPU temperature monitoring ---
def _read_cpu_temp():
    """Read CPU temp via vcgencmd; return float °C or None on error."""
    try:
        result = subprocess.run(
            ["vcgencmd", "measure_temp"],
            capture_output=True, text=True, timeout=3
        )
        # output is e.g. "temp=47.2'C\n"
        val = result.stdout.strip()
        return float(val.split("=")[1].split("'")[0])
    except Exception:
        return None

def _temp_monitor_thread():
    global _cpu_temp_filtered
    t = _read_cpu_temp()
    if t is not None:
        _cpu_temp_filtered = t
    while True:
        time.sleep(10)
        t = _read_cpu_temp()
        if t is not None:
            _cpu_temp_filtered = 0.9 * _cpu_temp_filtered + 0.1 * t

threading.Thread(target=_temp_monitor_thread, daemon=True).start()

# --- Preview server ---

PREVIEW_PORT          = 8080
TIMELINE_REGEN_FRAMES = 56    # regenerate timeline bar every Nth lores frame
TIMELINE_WIDTH        = 640
TIMELINE_HEIGHT       = 10
TIMELINE_WINDOW_S     = 3600.0

_preview_lock          = threading.Lock()
_preview_jpeg          = b""   # latest snapshot JPEG bytes
_preview_timeline_png  = b""   # latest timeline PNG bytes
_preview_timeline_counter = 0

def _trigger_timeline_rebuild():
    t = threading.Thread(target=_build_timeline, daemon=True)
    t.start()

def _build_timeline():
    """Render a 640x10 PNG timeline of events in the past hour."""
    global _preview_timeline_png
    now    = time.time()
    cutoff = now - TIMELINE_WINDOW_S
    img    = np.full((TIMELINE_HEIGHT, TIMELINE_WIDTH, 3), 40, dtype=np.uint8)  # dark grey

    with _preview_lock:
        events = [(t, n) for t, n in _preview_event_log if t > cutoff]

    # Count event endings per pixel column
    hits = np.zeros(TIMELINE_WIDTH, dtype=np.int32)
    for t, _ in events:
        col = int((t - cutoff) / TIMELINE_WINDOW_S * TIMELINE_WIDTH)
        col = max(0, min(TIMELINE_WIDTH - 1, col))
        hits[col] += 1

    # Blue lines; brighter for more hits in same column (base 120, +40 per extra, cap 255)
    for col, count in enumerate(hits):
        if count > 0:
            green = min(255, 120 + (count - 1) * 40)
            img[:, col] = (0, green, 0)  # BGR

    ok, png_buf = cv2.imencode(".png", img)
    if ok:
        with _preview_lock:
            _preview_timeline_png = png_buf.tobytes()


class _PreviewHandler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass  # suppress access log noise

    def do_POST(self):
        global focus_mode, change_mode, _manual_capture_fname, _manual_capture_time, motion_enabled, use_shm_mode
        path = self.path.split("?")[0]

        def _state_json():
            return json.dumps({
                "motion": motion_enabled,
                "focus":  focus_mode,
                "change": change_mode,
                "shm":    use_shm_mode,
                "local":  LOCAL_SAVE_MODE,
            }).encode()

        def _send_json(body):
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        if path == "/shmmode":
            if LOCAL_SAVE_MODE:
                use_shm_mode = not use_shm_mode
                if use_shm_mode:
                    os.makedirs(SHM_DIR, exist_ok=True)
            _send_json(_state_json())

        elif path == "/enable":
            motion_enabled = not motion_enabled
            _send_json(_state_json())

        elif path == "/focus":
            focus_mode = not focus_mode
            if focus_mode:
                picam2.set_controls({"ScalerCrop": (
                    FOCUS_ROI_X1, FOCUS_ROI_Y1,
                    FOCUS_ROI_X2 - FOCUS_ROI_X1,
                    FOCUS_ROI_Y2 - FOCUS_ROI_Y1,
                )})
            else:
                picam2.set_controls({"ScalerCrop": (
                    ROI_X1, ROI_Y1,
                    ROI_X2 - ROI_X1,
                    ROI_Y2 - ROI_Y1,
                )})
            global prev_gray
            prev_gray = None  # discard stale frame from previous crop geometry
            _send_json(_state_json())

        elif path == "/change":
            change_mode = not change_mode
            _send_json(_state_json())

        elif path == "/capture":
            now = datetime.now()
            ms  = now.microsecond // 1000
            save_dir = current_save_dir()
            fname = os.path.join(save_dir,
                                 f"{now.strftime('%Y%m%d_%H%M%S')}_{ms:03d}_C.jpg")
            try:
                frame     = picam2.capture_array("main")
                if focus_mode:
                    roi_frame = frame[FOCUS_ROI_Y1:FOCUS_ROI_Y2,
                                      FOCUS_ROI_X1:FOCUS_ROI_X2].copy()
                else:
                    roi_frame = frame[ROI_Y1:ROI_Y2, ROI_X1:ROI_X2].copy()
                img_pil = Image.fromarray(cv2.cvtColor(roi_frame, cv2.COLOR_BGR2RGB))
                img_pil.save(fname, "JPEG", quality=92)
                bname = os.path.basename(fname)
                with _preview_lock:
                    _manual_capture_fname = bname
                    _manual_capture_time  = time.time()
                if not LOCAL_SAVE_MODE or use_shm_mode:
                    with transfer_lock:
                        transfer_queue.append(fname)
                    transfer_event.set()
                body = bname.encode()
                self.send_response(200)
            except Exception as e:
                body = str(e).encode()
                self.send_response(500)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        else:
            self.send_error(404)

    def do_GET(self):
        path = self.path.split("?")[0]  # strip query string

        if path == "/snapshot":
            with _preview_lock:
                data = _preview_jpeg
            if not data:
                self.send_error(503, "No frame yet")
                return
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
            self.end_headers()
            self.wfile.write(data)

        elif path == "/timeline":
            with _preview_lock:
                data = _preview_timeline_png
            if not data:
                self.send_error(503, "No timeline yet")
                return
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
            self.end_headers()
            self.wfile.write(data)

        elif path == "/state":
            state = {
                "motion":  motion_enabled,
                "focus":   focus_mode,
                "change":  change_mode,
                "shm":     use_shm_mode,
                "local":   LOCAL_SAVE_MODE,
            }
            body = json.dumps(state).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(body)

        elif path == "/status":
            now = time.time()
            with _preview_lock:
                ts        = _preview_timestamp
                fname     = _preview_last_fname
                cap_fname = _manual_capture_fname
                cap_time  = _manual_capture_time
                px        = _preview_changed
                seq       = _preview_seq
                cutoff = now - 3600
                recent = [(t, n) for t, n in _preview_event_log if t > cutoff]
                while _preview_event_log and _preview_event_log[0][0] <= cutoff:
                    _preview_event_log.popleft()
            if cap_fname and (now - cap_time) > 10:
                cap_fname = ""
            with transfer_lock:
                backlog = len(transfer_queue)
            n_events = len(recent)
            n_frames = sum(n for _, n in recent)
            backlog_str = f"  BACKLOG:{backlog}" if backlog >= MAX_SHM_BACKLOG else ""
            cap_str     = f"  CAP:{cap_fname}" if cap_fname else ""
            body = (f"{ts} | px:{px} | seq:{seq} | last: {fname or '(none yet)'}"
                    f" | {n_events}/{n_frames}{backlog_str}{cap_str}"
                    f" | {_cpu_temp_filtered:.1f}°C").encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(body)

        elif path in ("/", "/stream"):
            shm_btn = '<button id="shmbtn" class="btn" onclick="toggleShm()">SHM OFF</button>' if LOCAL_SAVE_MODE else ''
            html = """<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<title>CAM2 Preview</title>
<style>
  body { background:#111; color:#ccc; font-family:monospace;
         display:flex; flex-direction:column; align-items:center;
         margin:0; padding:8px; }
  img  { max-width:100%; border:1px solid #444; display:block; }
  #timeline { width:640px; height:10px; display:block; margin-top:2px; border:1px solid #333; }
  #status { margin-top:6px; font-size:14px; letter-spacing:0.03em; }
  .btn { margin-top:8px; padding:6px 18px; font-family:monospace; font-size:14px;
         background:#333; color:#ccc; border:1px solid #666; cursor:pointer; margin-right:6px; }
  #focusbtn.active   { background:#554400; color:#ffcc00; border-color:#aa8800; }
  #enablebtn.inactive { background:#550000; color:#ff6666; border-color:#aa0000; }
  #shmbtn.active     { background:#003355; color:#66ccff; border-color:#0077aa; }
  #changebtn.active  { background:#003300; color:#00ff88; border-color:#00aa44; }
</style>
</head><body>
<img src="/snapshot" id="feed">
<img src="/timeline" id="timeline">
<div id="status">connecting...</div>
<div>
<button id="enablebtn" class="btn" onclick="toggleEnable()">Motion ON</button>
<button id="focusbtn"  class="btn" onclick="toggleFocus()">Focus OFF</button>
<button id="capturebtn" class="btn" onclick="doCapture()">Capture</button>
<button id="changebtn" class="btn" onclick="toggleChange()">Change OFF</button>
SHMBTN_PLACEHOLDER
</div>
<script>
  var lastSeq = -1;

  function applyState(s) {
    var eb = document.getElementById("enablebtn");
    if (s.motion) { eb.textContent = "Motion ON";  eb.classList.remove("inactive"); }
    else          { eb.textContent = "Motion OFF"; eb.classList.add("inactive"); }

    var fb = document.getElementById("focusbtn");
    if (s.focus) { fb.textContent = "Focus ON";  fb.classList.add("active"); }
    else         { fb.textContent = "Focus OFF"; fb.classList.remove("active"); }

    var sb = document.getElementById("shmbtn");
    if (sb) {
      if (s.shm) { sb.textContent = "SHM ON";  sb.classList.add("active"); }
      else       { sb.textContent = "SHM OFF"; sb.classList.remove("active"); }
    }

    var cb = document.getElementById("changebtn");
    if (s.change) { cb.textContent = "Change ON";  cb.classList.add("active"); }
    else          { cb.textContent = "Change OFF"; cb.classList.remove("active"); }
  }

  // Fetch server state once on load to sync button visuals
  fetch("/state")
    .then(r => r.json())
    .then(applyState)
    .catch(() => {});

  function toggleEnable() {
    fetch("/enable", {method:"POST"})
      .then(r => r.json())
      .then(applyState)
      .catch(() => {});
  }

  function toggleFocus() {
    fetch("/focus", {method:"POST"})
      .then(r => r.json())
      .then(applyState)
      .catch(() => {});
  }

  function toggleShm() {
    fetch("/shmmode", {method:"POST"})
      .then(r => r.json())
      .then(applyState)
      .catch(() => {});
  }

  function toggleChange() {
    fetch("/change", {method:"POST"})
      .then(r => r.json())
      .then(applyState)
      .catch(() => {});
  }

  function doCapture() {
    fetch("/capture", {method:"POST"})
      .then(r => r.text())
      .then(function(s) {
        document.getElementById("status").textContent = "Captured: " + s;
      })
      .catch(() => {});
  }

  setInterval(function(){
    fetch("/status")
      .then(r => r.text())
      .then(function(t) {
        var m = t.match(/seq:(\d+)/);
        if (m) {
          var seq = parseInt(m[1]);
          if (seq !== lastSeq) {
            lastSeq = seq;
            var ts = Date.now();
            document.getElementById("feed").src     = "/snapshot?_=" + ts;
            document.getElementById("timeline").src = "/timeline?_=" + ts;
          }
        }
        document.getElementById("status").textContent = t.replace(/\s*\|\s*seq:\d+/, "");
      })

      .catch(() => {});
  }, 1000);

  document.addEventListener("visibilitychange", function() {
    if (!document.hidden) {
      window.location.reload();
    }
  });
</script>
</body></html>"""
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            html_bytes = html.replace("SHMBTN_PLACEHOLDER", shm_btn).encode()
            self.send_header("Content-Length", str(len(html_bytes)))
            self.end_headers()
            self.wfile.write(html_bytes)

        else:
            self.send_error(404)


def _preview_server_thread():
    srv = ThreadingHTTPServer(("", PREVIEW_PORT), _PreviewHandler)
    srv.serve_forever()

threading.Thread(target=_preview_server_thread, daemon=True).start()
print(f"Preview server on port {PREVIEW_PORT}  /snapshot or /stream")


# --- Camera setup ---
# On Pi 5 with dual camera ports the camera may not be index 0.
# Find the camera attached to "cam0" by checking global_camera_info(); fall back to 0.
def _find_camera_num(port="cam0"):
    for info in Picamera2.global_camera_info():
        if port in str(info.get("Id", "")).lower():
            return info["Num"]
    return 0

_cam_num = _find_camera_num("cam0")
picam2 = Picamera2(_cam_num, tuning=Picamera2.load_tuning_file(TUNING_FILE))

# Note: picam2 misleadingly names RGB888 as BGR888 and vice-versa
config = picam2.create_still_configuration(
    main={"size": (4056, 3040), "format": "RGB888"},
    lores={"size": (640, 480), "format": "YUV420"},
    buffer_count=2
)
config["colour_space"] = ColorSpace.Srgb()
picam2.configure(config)
picam2.start()
apply_day_mode(picam2)
picam2.set_controls({"ScalerCrop": (ROI_X1, ROI_Y1, ROI_X2 - ROI_X1, ROI_Y2 - ROI_Y1)})
time.sleep(5)

metadata = picam2.capture_metadata()
print(f"AWB gains: {metadata.get('ColourGains')}")
print(f"Shutter: {metadata.get('ExposureTime')}us  Gain: {metadata.get('AnalogueGain'):.2f}")
print(f"Starting in {'NIGHT' if is_night_mode else 'DAY'} mode")

# --- Wait for NTP time sync before anchoring ---
def _wait_for_time_sync(timeout_s=120, poll_interval_s=5):
    """Block until systemd reports the clock is NTP-synchronized, or timeout."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            result = subprocess.run(
                ["timedatectl", "show", "--property=NTPSynchronized", "--value"],
                capture_output=True, text=True, timeout=5
            )
            if result.stdout.strip() == "yes":
                print(f"[timing] NTP synchronized, anchoring wall clock")
                return True
        except Exception:
            pass
        print(f"[timing] Waiting for NTP sync...")
        time.sleep(poll_interval_s)
    print(f"[timing] WARNING: NTP sync timeout after {timeout_s}s, anchoring with unsynchronized clock")
    return False

_wait_for_time_sync()

# --- Timing anchor: correlate sensor monotonic clock to wall clock ---
_wall_anchor      = datetime.now()
_sensor_anchor_ns = metadata.get("SensorTimestamp", None)

timing_log_lock = threading.Lock()

def sensor_to_exposure_start(sensor_ts_ns, is_first_frame):
    """Convert a raw SensorTimestamp (ns) to an estimated exposure-start datetime.

    The sensor timestamp marks end-of-pipeline delivery, not exposure start.
    We subtract the empirically measured pipeline latency:
      - LATENCY_FIRST_MS  for the first frame of a motion event
      - LATENCY_STEADY_MS for all subsequent frames
    Returns None if the anchor or timestamp is unavailable.
    """
    if sensor_ts_ns is None or _sensor_anchor_ns is None:
        return None
    latency_ms = LATENCY_FIRST_MS if is_first_frame else LATENCY_STEADY_MS
    sensor_dt  = _wall_anchor + timedelta(seconds=(sensor_ts_ns - _sensor_anchor_ns) / 1e9)
    return sensor_dt - timedelta(milliseconds=latency_ms)

def current_save_dir():
    """Return the active save directory based on mode flags."""
    return SD_SAVE_DIR if (LOCAL_SAVE_MODE and not use_shm_mode) else SHM_DIR

def make_fname(exposure_dt, wall_dt):
    """Return filename based on corrected exposure-start time,
    falling back to wall-clock time if sensor data is unavailable."""
    dt = exposure_dt if exposure_dt is not None else wall_dt
    ms = dt.microsecond // 1000
    return os.path.join(current_save_dir(), f"{dt.strftime('%Y%m%d_%H%M%S')}_{ms:03d}.jpg")

def write_timing_log(fname, wall_dt, sensor_dt, exposure_dt):
    """Append one row per saved frame with wall, raw-sensor, and corrected exposure times."""
    log_path = os.path.join(LOG_DIR, f"log_timing_{wall_dt.strftime('%Y%m%d')}.csv")
    is_new   = not os.path.exists(log_path)

    def fmt(dt):
        if dt is None:
            return "", ""
        return (f"{dt.timestamp():.4f}",
                dt.strftime('%H:%M:%S.') + f"{dt.microsecond // 1000:03d}")

    wall_s,     wall_hms     = fmt(wall_dt)
    sensor_s,   sensor_hms   = fmt(sensor_dt)
    exposure_s, exposure_hms = fmt(exposure_dt)
    diff_ms = (f"{(float(wall_s) - float(sensor_s)) * 1000:.1f}"
               if wall_s and sensor_s else "")

    with timing_log_lock:
        with open(log_path, "a", newline="") as f:
            w = csv.writer(f)
            if is_new:
                w.writerow(["filename", "wall_epoch_s", "wall_hms_ms",
                            "sensor_epoch_s", "sensor_hms_ms",
                            "exposure_epoch_s", "exposure_hms_ms",
                            "wall_minus_sensor_ms"])
            w.writerow([os.path.basename(fname),
                        wall_s, wall_hms,
                        sensor_s, sensor_hms,
                        exposure_s, exposure_hms,
                        diff_ms])

def _update_preview(lores, diff_mask_lores=None):
    """Update the snapshot JPEG and timeline PNG at their respective rates.

    diff_mask_lores: boolean array covering the lores ROI, or None.
    When change_mode is True and a mask is provided, motion pixels are shown
    at full color and non-motion pixels are dimmed to half-intensity grayscale.
    """
    global _preview_timeline_counter
    global _preview_jpeg, _preview_timestamp, _preview_seq
    global _preview_last_update_t

    now_t = time.time()
    if now_t - _preview_last_update_t >= 1.0:
        _preview_last_update_t = math.floor(now_t)
        bgr_full = cv2.cvtColor(lores, cv2.COLOR_YUV2BGR_I420)
        roi_img  = bgr_full[:480, :640]   # full lores = ScalerCrop region = ROI

        if focus_mode:
            rh, rw   = roi_img.shape[:2]
            fx1, fy1 = rw // 3, rh // 3
            fx2, fy2 = fx1 * 2, fy1 * 2
            crop        = roi_img[fy1:fy2, fx1:fx2]
            bgr_preview = cv2.resize(crop, (rw, rh), interpolation=cv2.INTER_LINEAR)
        else:
            bgr_preview = roi_img.copy()

        if change_mode and diff_mask_lores is not None and not focus_mode:
            # Resize mask to preview dimensions if they differ
            ph, pw = bgr_preview.shape[:2]
            mh, mw = diff_mask_lores.shape[:2]
            if (mh, mw) != (ph, pw):
                mask_u8   = diff_mask_lores.astype(np.uint8) * 255
                mask_u8   = cv2.resize(mask_u8, (pw, ph), interpolation=cv2.INTER_NEAREST)
                mask_bool = mask_u8 > 0
            else:
                mask_bool = diff_mask_lores
            # Non-motion pixels: half-intensity grayscale; motion pixels: full color
            gray      = cv2.cvtColor(bgr_preview, cv2.COLOR_BGR2GRAY)
            gray_half = (gray.astype(np.uint16) // 2).astype(np.uint8)
            gray_bgr  = cv2.cvtColor(gray_half, cv2.COLOR_GRAY2BGR)
            bgr_preview = np.where(mask_bool[:, :, np.newaxis], bgr_preview, gray_bgr)

        ok, jpg_buf = cv2.imencode(".jpg", bgr_preview, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if ok:
            with _preview_lock:
                _preview_jpeg      = jpg_buf.tobytes()
                _preview_timestamp = datetime.now().strftime("%H:%M:%S")
                _preview_seq      += 1

    _preview_timeline_counter += 1
    if _preview_timeline_counter >= TIMELINE_REGEN_FRAMES:
        _preview_timeline_counter = 0
        _trigger_timeline_rebuild()

def _handle_motion_frame(changed, gray_roi, diff_mask):
    """Process one lores frame that exceeded the motion threshold."""
    global event_max_px, event_frame_count, event_shutter_sum
    global event_gain_sum, event_last_px, event_peak_fname
    global _preview_last_fname, _last_save_time

    # Force-close event if it has exceeded the duration cap
    if in_motion_event and (time.time() - event_t_start) > EVENT_MAX_DURATION_S:
        _close_event(changed, is_cap=True)

    # Start new event if needed (must come before event_fnames.append)
    if not in_motion_event:
        _start_event()

    # ==================================================================

    # Check transfer backlog and save-rate limit before full-res capture
    with transfer_lock:
        backlog = len(transfer_queue)
    backlog_blocked  = backlog >= MAX_SHM_BACKLOG
    now_t            = time.time()
    interval_ok      = (now_t - _last_save_time) >= MIN_SAVE_INTERVAL_S

    if not backlog_blocked and interval_ok and not focus_mode and motion_enabled:
        frame     = picam2.capture_array("main")
        roi_frame = frame[ROI_Y1:ROI_Y2, ROI_X1:ROI_X2].copy()
        _last_save_time = now_t
    else:
        roi_frame = None

    # Always capture metadata (no /dev/shm allocation)
    meta         = picam2.capture_metadata()
    now          = datetime.now()
    shutter_us   = meta.get("ExposureTime", 0)
    gain         = meta.get("AnalogueGain", 1.0)
    sensor_ts_ns = meta.get("SensorTimestamp")

    is_first    = not in_motion_event
    sensor_dt   = (_wall_anchor + timedelta(seconds=(sensor_ts_ns - _sensor_anchor_ns) / 1e9)
                   if sensor_ts_ns is not None and _sensor_anchor_ns is not None else None)
    exposure_dt = sensor_to_exposure_start(sensor_ts_ns, is_first)
    fname       = make_fname(exposure_dt, now)

    if roi_frame is not None:
        with _preview_lock:
            _preview_last_fname = os.path.basename(fname)
        if SAVE_TIMING_LOG:
            write_timing_log(fname, now, sensor_dt, exposure_dt)
        exif_bytes = piexif.dump({"Exif": {
            piexif.ExifIFD.ExposureTime:    (shutter_us, 1_000_000),
            piexif.ExifIFD.ISOSpeedRatings: round(gain * 100),
            piexif.ExifIFD.Temperature:     (round(_cpu_temp_filtered * 10), 10),  # e.g. 47.3°C -> (473, 10)
        }})            
        with save_lock:
            dropped = len(save_queue) == save_queue.maxlen
            save_queue.append((roi_frame, fname, exif_bytes))
        save_event.set()
        if verbose:
            if dropped:
                print("WARNING: save buffer full, oldest frame dropped")
            print(fname)
        event_fnames.append(fname)

    # CoM tracking (always, regardless of backlog)
    col_sums = diff_mask.sum(axis=0).astype(float)
    if col_sums.sum() > 0:
        com_x = np.average(np.arange(diff_mask.shape[1]), weights=col_sums)
        event_com_x_list.append(LORES_X1 + com_x)

    # Update event accumulators
    event_frame_count += 1
    event_shutter_sum += shutter_us
    event_gain_sum    += gain
    if changed > event_max_px:
        event_max_px = changed
        if roi_frame is not None:
            event_peak_fname = os.path.basename(fname)
    event_last_px = changed


def _handle_summary():
    """Print periodic brightness summary and switch day/night mode if warranted."""
    global bg_accum, bg_count, summary_frame
    global day_candidate_count, night_candidate_count

    summary_frame += 1
    if summary_frame < SUMMARY_INTERVAL:
        return

    if bg_count > 0:
        bg_avg   = bg_accum / bg_count
        mode_str = "NIGHT" if is_night_mode else "DAY"
        if verbose:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] [{mode_str}] bg brightness avg: {bg_avg:.1f}"
                  f"  ({bg_count}/{SUMMARY_INTERVAL} non-motion frames)  thr={threshold}")

        going_day   = is_night_mode     and bg_avg >= DAY_THRESHOLD
        going_night = not is_night_mode and bg_avg <= NIGHT_THRESHOLD

        if going_day:
            day_candidate_count += 1
            night_candidate_count = 0
        elif going_night:
            night_candidate_count += 1
            day_candidate_count = 0
        else:
            day_candidate_count = night_candidate_count = 0

        if verbose and going_day:
            print(f"  Day candidate {day_candidate_count}/{SWITCH_INTERVALS}")
        if verbose and going_night:
            print(f"  Night candidate {night_candidate_count}/{SWITCH_INTERVALS}")

        if going_day and day_candidate_count >= SWITCH_INTERVALS:
            apply_day_mode(picam2)
            day_candidate_count = 0
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Switched to DAY mode")
        elif going_night and night_candidate_count >= SWITCH_INTERVALS:
            apply_night_mode(picam2)
            night_candidate_count = 0
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Switched to NIGHT mode")
    else:
        if verbose:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] bg brightness avg: n/a (all frames were motion)")

    bg_accum = bg_count = summary_frame = 0


# --- Main loop ---
changed    = 0

while True:
    # Capture the lores frame for motion detection and preview updates.
    # This is done on every loop iteration regardless of transfer backlog, since
    # the lores stream is lightweight; used for CoM tracking and day/night switching.
    lores = picam2.capture_array("lores")

    gray_roi = lores[:480, :640][LORES_Y1:LORES_Y2, LORES_X1:LORES_X2]

    if prev_gray is not None:
        diff_mask = cv2.absdiff(gray_roi, prev_gray) > 25
        # Zero out excluded top and bottom strips
        if EXCLUDE_TOP_ROWS > 0:
            diff_mask[:EXCLUDE_TOP_ROWS, :] = False
        if EXCLUDE_BOT_ROWS > 0:
            diff_mask[-EXCLUDE_BOT_ROWS:, :] = False
        # Remove blobs smaller than MIN_BLOB_PIXELS
        n_labels, _, stats, _ = cv2.connectedComponentsWithStats(
            diff_mask.astype(np.uint8), connectivity=8)
        for label in range(1, n_labels):  # skip background label 0
            if stats[label, cv2.CC_STAT_AREA] < MIN_BLOB_PIXELS:
                diff_mask[stats[label, cv2.CC_STAT_TOP]:
                          stats[label, cv2.CC_STAT_TOP] + stats[label, cv2.CC_STAT_HEIGHT],
                          stats[label, cv2.CC_STAT_LEFT]:
                          stats[label, cv2.CC_STAT_LEFT] + stats[label, cv2.CC_STAT_WIDTH]] = False
        changed   = np.sum(diff_mask)
        _preview_changed = changed
        if verbose and changed > 5:
            print(changed)

        if changed > threshold:
            _handle_motion_frame(changed, gray_roi, diff_mask)
        else:
            if in_motion_event:
                _close_event(changed)
            bg_accum += float(np.mean(gray_roi))
            bg_count += 1
            event_last_px = changed

        _handle_summary()
        _update_preview(lores, diff_mask_lores=diff_mask)
    else:
        _update_preview(lores)

    prev_gray = gray_roi
    time.sleep(0.05)
