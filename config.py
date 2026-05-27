# Site-specific configuration - do not commit
REMOTE_HOST       = "jbeale@jbeale-mini"
REMOTE_DIR        = "/mnt/bluecherry/CAMA/"
LOG_DIR           = "/home/pi/CAMA"
VENV_PYTHON       = "/home/pi/cam-env/bin/python"
TUNING_FILE       = "/home/pi/CAMA/imx477_custom.json"
ROTATION          = True
FORCE_REMOTE_SAVE = True

# Camera ROI (main frame coordinates)
ROI_X1, ROI_Y1 = 0, 0
ROI_X2, ROI_Y2 = 1455, 1087 # for Pi GlobalShutter camera (IMX296)
# ROI_X2, ROI_Y2 = 4055, 3039  # for Pi HQ camera (IMX477)


# Motion thresholds (tune to your scene)
THRESHOLD_DAY     = 1600
THRESHOLD_NIGHT   = 40

MOTION_EXCLUDE_TOP    = 1/5   # fraction of ROI height to ignore at top
MOTION_EXCLUDE_BOTTOM = 1/6   # fraction of ROI height to ignore at bottom

# Pipeline latency calibration (ms)
LATENCY_FIRST_MS  = 405
LATENCY_STEADY_MS = 441

# minimum seconds between full-res saves during an event (actual min: 0.342 s)
MIN_SAVE_INTERVAL_S    = 0.6

DETECT_OUT_DIR = "/home/pi/CAMA/results"


