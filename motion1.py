#!/home/pi/cam-env/bin/python

# capture motion-triggered frames with PiCamera2
# using a low-res stream for motion detection
# sampled ~10 fps, saved ~3 fps using 2nd thread for JPEG encoding and disk I/O
# writes log entries for each motion event with timestamps, brightness, motion size, etc.
# pushes saved JPEGs to remote host via rsync as soon as they are written
# serves a decimated preview stream on port 8080 for live preview in a browser
# filters out noise/tree events by requiring minimum CoM displacement or velocity
# custom AeExposureMode limits longest shutter speed to 2 msec

# J.Beale 2026-04-12

import os
import glob
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

# get site-specific parameters
from config import (REMOTE_HOST, REMOTE_DIR, LOG_DIR, TUNING_FILE,
                    ROI_X1, ROI_Y1, ROI_X2, ROI_Y2,
                    THRESHOLD_DAY, THRESHOLD_NIGHT,
                    LATENCY_FIRST_MS, LATENCY_STEADY_MS)

TRANSFER_BUFFER_SIZE   = 64     # max queued filenames (each ~1.3 MB in /dev/shm)
TRANSFER_MAX_RETRIES   = 3      # give up after this many consecutive rsync failures
SAVE_BUFFER_SIZE       = 12
MAX_SHM_BACKLOG        = 20     # pause full-res saves if transfer queue exceeds this

SHM_DIR                = "/dev/shm/cam9"  # working directory for full-res JPEGs
SHM_STALE_HOURS        = 1               # delete leftover JPEGs older than this on startup

EVENT_MAX_DURATION_S   = 5.0    # force-close an event after this many seconds
EVENT_MIN_DISPLACEMENT = 262    # lores px: min CoM range to keep event (half ROI width)
EVENT_MIN_VELOCITY     = 5.0    # lores px/frame: min |velocity| to keep event

# Corresponding ROI in lores (640x480) coordinates
LORES_X1 = round(ROI_X1 * 640 / 4056)
LORES_Y1 = round(ROI_Y1 * 480 / 3040)
LORES_X2 = round(ROI_X2 * 640 / 4056)
LORES_Y2 = round(ROI_Y2 * 480 / 3040)

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
event_shutter_sum   = 0.0
event_gain_sum      = 0.0
event_bg_brightness = 0.0
event_com_x_list    = []
event_fnames        = []   # filenames saved during this event

_preview_last_fname = ""   # last motion-triggered filename
_preview_timestamp  = ""   # timestamp of last preview frame
_preview_event_log  = deque()  # (event_end_time, frame_count) tuples


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

# --- Preview server ---

PREVIEW_PORT          = 8080
PREVIEW_EVERY         = 10    # encode snapshot every Nth lores frame
TIMELINE_REGEN_FRAMES = 56    # regenerate timeline bar every Nth lores frame
TIMELINE_WIDTH        = 640
TIMELINE_HEIGHT       = 10
TIMELINE_WINDOW_S     = 3600.0

_preview_lock          = threading.Lock()
_preview_jpeg          = b""   # latest snapshot JPEG bytes
_preview_timeline_png  = b""   # latest timeline PNG bytes
_preview_frame_count   = 0
_preview_timeline_counter = 0


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
            blue = min(255, 120 + (count - 1) * 40)
            img[:, col] = (blue, 0, 0)  # BGR

    ok, png_buf = cv2.imencode(".png", img)
    if ok:
        with _preview_lock:
            _preview_timeline_png = png_buf.tobytes()


class _PreviewHandler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass  # suppress access log noise

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

        elif path == "/status":
            with _preview_lock:
                ts     = _preview_timestamp
                fname  = _preview_last_fname
                now    = time.time()
                cutoff = now - 3600
                recent = [(t, n) for t, n in _preview_event_log if t > cutoff]
                # prune old entries while we have the lock
                while _preview_event_log and _preview_event_log[0][0] <= cutoff:
                    _preview_event_log.popleft()
            with transfer_lock:
                backlog = len(transfer_queue)
            n_events = len(recent)
            n_frames = sum(n for _, n in recent)
            backlog_str = f"  BACKLOG:{backlog}" if backlog >= MAX_SHM_BACKLOG else ""
            body = (f"{ts} | {changed} | last: {fname or '(none yet)'}"
                    f" | {n_events}/{n_frames}{backlog_str}").encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(body)

        elif path in ("/", "/stream"):
            html = b"""<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<title>CAM9 Preview</title>
<style>
  body { background:#111; color:#ccc; font-family:monospace;
         display:flex; flex-direction:column; align-items:center;
         margin:0; padding:8px; }
  img  { max-width:100%; border:1px solid #444; display:block; }
  #timeline { width:640px; height:10px; display:block; margin-top:2px; border:1px solid #333; }
  #status { margin-top:6px; font-size:14px; letter-spacing:0.03em; }
</style>
</head><body>
<img src="/snapshot" id="feed">
<img src="/timeline" id="timeline">
<div id="status">connecting...</div>
<script>
  setInterval(function(){
    var ts = Date.now();
    document.getElementById("feed").src     = "/snapshot?_=" + ts;
    document.getElementById("timeline").src = "/timeline?_=" + ts;
    fetch("/status?_=" + ts)
      .then(r => r.text())
      .then(t => { document.getElementById("status").textContent = t; })
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
            self.send_header("Content-Length", str(len(html)))
            self.end_headers()
            self.wfile.write(html)

        else:
            self.send_error(404)


def _preview_server_thread():
    srv = ThreadingHTTPServer(("", PREVIEW_PORT), _PreviewHandler)
    srv.serve_forever()

threading.Thread(target=_preview_server_thread, daemon=True).start()
print(f"Preview server on port {PREVIEW_PORT}  /snapshot or /stream")


# --- Camera setup ---
picam2 = Picamera2(tuning=Picamera2.load_tuning_file(TUNING_FILE))

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
time.sleep(5)

metadata = picam2.capture_metadata()
print(f"AWB gains: {metadata.get('ColourGains')}")
print(f"Shutter: {metadata.get('ExposureTime')}us  Gain: {metadata.get('AnalogueGain'):.2f}")
print(f"Starting in {'NIGHT' if is_night_mode else 'DAY'} mode")

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

def make_fname(exposure_dt, wall_dt):
    """Return SHM_DIR filename based on corrected exposure-start time,
    falling back to wall-clock time if sensor data is unavailable."""
    dt = exposure_dt if exposure_dt is not None else wall_dt
    ms = dt.microsecond // 1000
    return os.path.join(SHM_DIR, f"{dt.strftime('%Y%m%d_%H%M%S')}_{ms:03d}.jpg")

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

# --- Main loop ---
changed = 0  # initialise so /status handler never references undefined name

while True:
    lores = picam2.capture_array("lores")

    # Snapshot preview (every PREVIEW_EVERY frames)
    _preview_frame_count += 1
    if _preview_frame_count >= PREVIEW_EVERY:
        _preview_frame_count = 0
        # Shape (720, 640) is packed planar YUV420 (I420): 480 Y rows + 240 chroma rows
        bgr_full    = cv2.cvtColor(lores, cv2.COLOR_YUV2BGR_I420)
        bgr_preview = bgr_full[:480, :640]
        ok, jpg_buf = cv2.imencode(".jpg", bgr_preview, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if ok:
            with _preview_lock:
                _preview_jpeg      = jpg_buf.tobytes()
                _preview_timestamp = datetime.now().strftime("%H:%M:%S")

    # Timeline bar (every TIMELINE_REGEN_FRAMES frames)
    _preview_timeline_counter += 1
    if _preview_timeline_counter >= TIMELINE_REGEN_FRAMES:
        _preview_timeline_counter = 0
        _build_timeline()

    gray_roi = lores[:480, :640][LORES_Y1:LORES_Y2, LORES_X1:LORES_X2]

    if prev_gray is not None:
        diff_mask = cv2.absdiff(gray_roi, prev_gray) > 25
        changed   = np.sum(diff_mask)
        if verbose and changed > 5:
            print(changed)

        if changed > threshold:
            # Force-close event if it has exceeded the duration cap
            if in_motion_event and (time.time() - event_t_start) > EVENT_MAX_DURATION_S:
                _close_event(changed, is_cap=True)

            # Start new event if needed (must come before event_fnames.append)
            if not in_motion_event:
                _start_event()

            # Check transfer backlog before attempting full-res capture
            with transfer_lock:
                backlog = len(transfer_queue)
            backlog_blocked = backlog >= MAX_SHM_BACKLOG

            if not backlog_blocked:
                frame     = picam2.capture_array("main")
                roi_frame = frame[ROI_Y1:ROI_Y2, ROI_X1:ROI_X2].copy()
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

            # X center of mass of changed pixels in lores ROI (always, regardless of backlog)
            col_sums = diff_mask.sum(axis=0).astype(float)
            if col_sums.sum() > 0:
                com_x = np.average(np.arange(diff_mask.shape[1]), weights=col_sums)
                event_com_x_list.append(LORES_X1 + com_x)  # absolute lores x

            # Update event accumulators
            event_frame_count += 1
            event_shutter_sum += shutter_us
            event_gain_sum    += gain
            if changed > event_max_px:
                event_max_px = changed
                if roi_frame is not None:
                    event_peak_fname = os.path.basename(fname)
            event_last_px = changed

        else:
            # Non-motion frame
            if in_motion_event:
                _close_event(changed)
            bg_accum += float(np.mean(gray_roi))
            bg_count += 1
            event_last_px = changed

        summary_frame += 1
        if summary_frame >= SUMMARY_INTERVAL:
            if bg_count > 0:
                bg_avg   = bg_accum / bg_count
                mode_str = "NIGHT" if is_night_mode else "DAY"
                if verbose:
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] [{mode_str}] bg brightness avg: {bg_avg:.1f}"
                          f"  ({bg_count}/{SUMMARY_INTERVAL} non-motion frames)  thr={threshold}")

                # Unified day/night switching logic
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

    prev_gray = gray_roi
    time.sleep(0.05)
