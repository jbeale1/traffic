#!/home/pi/my-env/bin/python

# capture motion-triggered frames with PiCamera2
# using a low-res stream for motion detection
# sampled ~10 fps, saved ~3 fps using 2nd thread for JPEG encoding and disk I/O
# writes log entries for each motion event with timestamps, brightness, motion size, etc.
# pushes saved JPEGs to remote host via rsync as soon as they are written
# J.Beale 2026-03-31

import piexif
from PIL import Image
import cv2
import numpy as np
from picamera2 import Picamera2
from libcamera import ColorSpace, controls
import time
from datetime import datetime
import threading
from collections import deque
import os
import csv
import subprocess

LOG_DIR = "/home/pi/CAM9"  # save motion log here

# Remote destination for storing JPEGs
REMOTE_HOST = "user@host.local"  # update to actual remote host
REMOTE_DIR  = "/mnt/data/CAM9/"  # update to actual directory on the remote host for images

TRANSFER_BUFFER_SIZE = 64   # max queued filenames (each ~1.3 MB in /dev/shm)
TRANSFER_MAX_RETRIES = 3    # give up after this many consecutive rsync failures

# ROI in main (full-res) coordinates : useful portion of frame
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
THRESHOLD_DAY      = 5000   # changed pixels to trigger in daytime
DAY_THRESHOLD      = 25     # bg brightness above this -> day candidate
NIGHT_THRESHOLD    = 20     # bg brightness below this -> night candidate
SWITCH_INTERVALS   = 3      # consecutive summary intervals to confirm mode switch
SUMMARY_INTERVAL   = 200    # frames per summary (~1 min at 10 fps)
SAVE_BUFFER_SIZE   = 12

# Global state variables
verbose       = False
is_night_mode = False  # start in day mode
threshold     = THRESHOLD_DAY
prev_gray     = None
last_timestamp        = ""
frame_in_second       = 0
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
        "ExposureTimeMode": 1,
        "ExposureTime":     MAX_SHUTTER_US,
        "AnalogueGainMode": 0,
        "AwbMode":          controls.AwbModeEnum.Daylight,
    })

def apply_day_mode(cam):
    global threshold, is_night_mode
    threshold     = THRESHOLD_DAY
    is_night_mode = False
    cam.set_controls({
        "ExposureValue":    -2.0,
        "ExposureTimeMode": 0,
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
    consecutive_failures = 0
    while True:
        transfer_event.wait()
        transfer_event.clear()
        while True:
            with transfer_lock:
                if not transfer_queue:
                    break
                fpath = transfer_queue.popleft()

            if not os.path.exists(fpath):
                # Already gone (shouldn't happen, but be safe)
                continue

            result = subprocess.run(
                ["rsync", "--remove-source-files", "-q",
                 fpath, f"{REMOTE_HOST}:{REMOTE_DIR}"],
                capture_output=True
            )

            if result.returncode == 0:
                consecutive_failures = 0
            else:
                consecutive_failures += 1
                err = result.stderr.decode().strip()
                print(f"[transfer] rsync failed ({consecutive_failures}/{TRANSFER_MAX_RETRIES}): {err}")
                if consecutive_failures < TRANSFER_MAX_RETRIES:
                    # Put it back at the front and pause before retrying
                    with transfer_lock:
                        transfer_queue.appendleft(fpath)
                    time.sleep(5)
                    transfer_event.set()
                else:
                    print(f"[transfer] giving up on {fpath} after {TRANSFER_MAX_RETRIES} failures")
                    consecutive_failures = 0
                break  # re-enter wait loop; event will be set again if queue non-empty

threading.Thread(target=transfer_thread, daemon=True).start()

# --- Camera setup ---
picam2 = Picamera2()
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

# --- Main loop ---
while True:
    lores    = picam2.capture_array("lores")
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
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if timestamp != last_timestamp:
                frame_in_second = 0
                last_timestamp  = timestamp
            fname           = f"/dev/shm/{timestamp}_{frame_in_second:02d}.jpg"
            frame_in_second += 1

            shutter_us = meta.get("ExposureTime", 0)
            gain       = meta.get("AnalogueGain", 1.0)
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
