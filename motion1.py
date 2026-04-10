#!/home/pi/my-env/bin/python

# capture motion-triggered frames with PiCamera2
# uses a low-res stream for motion detection
# sampled ~10 fps, saved ~3 fps using 2nd thread for JPEG encoding and disk I/O
# writes log entries for each motion event with timestamps, brightness, motion size, etc.
# pushes saved JPEGs to remote host via rsync as soon as they are written
# serves a decimated MJPEG stream on port 8080 for live preview in a browser
# J.Beale 2026-04-10
# custom AeExposureMode limits longest shutter speed to 2 msec

import os
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

import io       # for mjpeg server
import socket
from http.server import HTTPServer, BaseHTTPRequestHandler, ThreadingHTTPServer

LOG_DIR = "/home/pi/CAM9"  # save motion log here

# Remote destination for pushed JPEGs
REMOTE_HOST = "me@myserver.local"
REMOTE_DIR  = "/mnt/data/CAM9/"
TRANSFER_BUFFER_SIZE = 64   # max queued filenames (each ~1.3 MB in /dev/shm)
TRANSFER_MAX_RETRIES = 3    # give up after this many consecutive rsync failures

# ROI in main (full-res) coordinates
ROI_X1, ROI_Y1 = 720, 0
ROI_X2, ROI_Y2 = 4055, 2252

# Corresponding ROI in lores (640x480) coordinates
LORES_X1 = round(ROI_X1 * 640 / 4056)  # 114
LORES_Y1 = round(ROI_Y1 * 480 / 3040)  # 0
LORES_X2 = round(ROI_X2 * 640 / 4056)  # 638
LORES_Y2 = round(ROI_Y2 * 480 / 3040)  # 355

# Day/night switching thresholds and parameters
MAX_SHUTTER_US     = 2000   # night mode fixed shutter (usec)
THRESHOLD_NIGHT    = 40     # changed pixels to trigger at night (small lights)
THRESHOLD_DAY      = 4000   # changed pixels to trigger in daytime
DAY_THRESHOLD      = 25     # bg brightness above this -> day candidate
NIGHT_THRESHOLD    = 20     # bg brightness below this -> night candidate
SWITCH_INTERVALS   = 3      # consecutive summary intervals to confirm mode switch
SUMMARY_INTERVAL   = 200    # frames per summary (~1 min at 10 fps)
SAVE_BUFFER_SIZE   = 12

# Pipeline latency corrections (sensor timestamp -> actual exposure start).
# The sensor timestamp marks end-of-pipeline delivery, not exposure start.
# First frame of each event uses a smaller correction (~40 ms less) due to
# pipeline buffering behaviour observed empirically.
LATENCY_FIRST_MS  = 405    # ms to subtract for first frame of a motion event
LATENCY_STEADY_MS = 441    # ms to subtract for subsequent frames (cal ref: GPS UTC pps)

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
in_motion_event   = False
event_t_start     = 0.0
event_frame_count = 0
event_max_px      = 0
event_peak_fname  = ""
event_pre_px      = 0
event_last_px     = 0
event_shutter_sum = 0.0
event_gain_sum    = 0.0
event_bg_brightness = 0.0
event_com_x_list  = []   # lores-space x CoM per motion frame

_preview_last_fname = ""   # last motion-triggered filename
_preview_timestamp  = ""   # timestamp of last preview frame

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

    with open(log_path, "a", newline="") as f:
        w = csv.writer(f)
        if is_new:
            w.writerow(["epoch_start", "time_hms", "peak_fname", "duration_s", "frame_count",
                        "bg_brightness", "max_motion_px", "pre_event_px", "post_event_px",
                        "mean_shutter_us", "mean_gain", "velocity_px_per_frame"])
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
            # Enqueue for transfer only after the file is fully written
            with transfer_lock:
                transfer_queue.append(fname)
            transfer_event.set()

threading.Thread(target=saver_thread, daemon=True).start()

# --- Transfer thread ---
transfer_queue = deque(maxlen=TRANSFER_BUFFER_SIZE)
transfer_event = threading.Event()
transfer_lock  = threading.Lock()

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

# --- MJPEG preview server ---

PREVIEW_PORT   = 8080
PREVIEW_EVERY  = 10        # serve every Nth lores frame
_preview_lock  = threading.Lock()
_preview_jpeg  = b""       # latest encoded JPEG bytes
_preview_frame_count = 0   # counts lores frames for decimation

class _PreviewHandler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

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

        elif path == "/status":
            with _preview_lock:
                ts    = _preview_timestamp
                fname = _preview_last_fname
            body = f"{ts}  |  last saved: {fname or '(none yet)'}".encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(body)

        elif path == "/mjpeg":
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            try:
                while True:
                    with _preview_lock:
                        data = _preview_jpeg
                    if data:
                        self.wfile.write(
                            b"--frame\r\nContent-Type: image/jpeg\r\n"
                            + f"Content-Length: {len(data)}\r\n\r\n".encode()
                            + data + b"\r\n"
                        )
                        self.wfile.flush()
                    time.sleep(1.0)
            except (BrokenPipeError, ConnectionResetError):
                pass

        elif path in ("/", "/stream"):
            html = b"""<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<title>CAM9 Preview</title>
<style>
  body { background:#111; color:#ccc; font-family:monospace;
         display:flex; flex-direction:column; align-items:center;
         margin:0; padding:8px; }
  img  { max-width:100%; border:1px solid #444; }
  #status { margin-top:6px; font-size:14px; letter-spacing:0.03em; }
</style>
</head><body>
<img src="/snapshot" id="feed">
<div id="status">connecting...</div>
<script>
  setInterval(function(){
    var ts = Date.now();
    document.getElementById("feed").src = "/snapshot?_=" + ts;
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
picam2 = Picamera2(tuning=Picamera2.load_tuning_file("/home/pi/CAM9/imx477_custom.json"))

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
    """Return /dev/shm filename based on corrected exposure-start time,
    falling back to wall-clock time if sensor data is unavailable."""
    dt = exposure_dt if exposure_dt is not None else wall_dt
    ms = dt.microsecond // 1000
    return f"/dev/shm/{dt.strftime('%Y%m%d_%H%M%S')}_{ms:03d}.jpg"

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
while True:
    lores    = picam2.capture_array("lores")

    _preview_frame_count += 1
    if _preview_frame_count >= PREVIEW_EVERY:
        _preview_frame_count = 0

        # Shape (720, 640) is packed planar YUV420 (I420): 480 Y rows + 240 chroma rows, 1 channel


        bgr_full = cv2.cvtColor(lores, cv2.COLOR_YUV2BGR_I420)
        bgr_preview = bgr_full[:480, :640]
        ok, jpg_buf = cv2.imencode(".jpg", bgr_preview, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if ok:
            with _preview_lock:
                _preview_jpeg     = jpg_buf.tobytes()
                _preview_timestamp = datetime.now().strftime("%H:%M:%S")

    gray_roi = lores[:480, :640][LORES_Y1:LORES_Y2, LORES_X1:LORES_X2]

    if prev_gray is not None:
        diff_mask = cv2.absdiff(gray_roi, prev_gray) > 25
        changed   = np.sum(diff_mask)
        if verbose and changed > 5:
            print(changed)

        if changed > threshold:
            frame     = picam2.capture_array("main")
            meta      = picam2.capture_metadata()
            roi_frame = frame[ROI_Y1:ROI_Y2, ROI_X1:ROI_X2].copy()
            now       = datetime.now()

            shutter_us   = meta.get("ExposureTime", 0)
            gain         = meta.get("AnalogueGain", 1.0)
            sensor_ts_ns = meta.get("SensorTimestamp")

            # Determine whether this is the first frame of a new motion event
            # before updating in_motion_event, so the flag is correct for
            # both the latency correction and the event tracking block below.
            is_first    = not in_motion_event
            sensor_dt   = (_wall_anchor + timedelta(seconds=(sensor_ts_ns - _sensor_anchor_ns) / 1e9)
                           if sensor_ts_ns is not None and _sensor_anchor_ns is not None else None)
            exposure_dt = sensor_to_exposure_start(sensor_ts_ns, is_first)
            fname       = make_fname(exposure_dt, now)
            with _preview_lock:
                _preview_last_fname = os.path.basename(fname)

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

            # X center of mass of changed pixels in lores ROI
            col_sums = diff_mask.sum(axis=0).astype(float)
            if col_sums.sum() > 0:
                com_x = np.average(np.arange(diff_mask.shape[1]), weights=col_sums)
                event_com_x_list.append(LORES_X1 + com_x)  # absolute lores x

            # Event tracking
            if not in_motion_event:
                in_motion_event     = True
                event_t_start       = time.time()
                event_frame_count   = 0
                event_max_px        = 0
                event_peak_fname    = os.path.basename(fname)
                event_pre_px        = event_last_px
                event_shutter_sum   = 0.0
                event_gain_sum      = 0.0
                event_bg_brightness = bg_accum / bg_count if bg_count > 0 else 0.0
                event_com_x_list.clear()
            event_frame_count += 1
            event_shutter_sum += shutter_us
            event_gain_sum    += gain
            if changed > event_max_px:
                event_max_px     = changed
                event_peak_fname = os.path.basename(fname)
            event_last_px = changed

        else:
            # Non-motion frame
            if in_motion_event:
                in_motion_event = False
                write_event_log({
                    "t_start":       event_t_start,
                    "duration":      time.time() - event_t_start,
                    "peak_fname":    event_peak_fname,
                    "frame_count":   event_frame_count,
                    "bg_brightness": event_bg_brightness,
                    "max_motion_px": event_max_px,
                    "pre_event_px":  event_pre_px,
                    "post_event_px": changed,
                    "mean_shutter":  event_shutter_sum / event_frame_count,
                    "mean_gain":     event_gain_sum    / event_frame_count,
                    "com_x_list":    list(event_com_x_list),
                })
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
