"""
Configuration and thresholds for the Driver Attention Monitoring System.

Detection uses a zone-based system with duration thresholds:
  ROAD     — forward gaze within tolerances, driver is attentive
  GLANCE   — brief off-road look (mirror check, speedometer), never flagged
  DISTRACTED — sustained off-road gaze exceeding a duration threshold
"""

import os

# --- Paths ---
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "sessions")
YOLO_MODEL_PATH = "yolov8n.pt"  # Will auto-download on first run
FACE_LANDMARKER_MODEL_PATH = "face_landmarker.task"  # Download from MediaPipe
FACE_LANDMARKER_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
)

# --- Camera ---
CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FPS = 20

# --- MediaPipe FaceLandmarker (Tasks API) ---
FACE_LANDMARKER_MAX_FACES = 1
FACE_LANDMARKER_MIN_DETECTION_CONFIDENCE = 0.5
FACE_LANDMARKER_MIN_TRACKING_CONFIDENCE = 0.5

# ─── Calibration ───────────────────────────────────────────────────────────
# Seconds of "look at the road" capture used to establish neutral baseline
CALIBRATION_DURATION = 3.0

# ─── Head Pose Zones ────────────────────────────────────────────────────────
# ROAD zone: driver is looking forward (attentive)
#   yaw within ±ROAD_YAW_TOLERANCE, pitch within ROAD_PITCH range
ROAD_YAW_TOLERANCE = 20.0          # degrees — symmetric left/right of centre
ROAD_PITCH_UP_LIMIT = 15.0         # degrees — max upward tilt still "on road"
ROAD_PITCH_DOWN_LIMIT = -15.0      # degrees — max downward tilt still "on road"

# Anything outside the ROAD zone starts as a GLANCE. A GLANCE becomes
# DISTRACTED only when it exceeds a duration threshold (see below).

# ─── Zone Duration Thresholds (seconds) ─────────────────────────────────────
# Brief glances shorter than this are NEVER flagged, regardless of direction.
GLANCE_GRACE_PERIOD = 1.5

# Sustained off-road gaze beyond this → "distracted_offroad" event
OFFROAD_DISTRACTION_THRESHOLD = 2.5

# Head pitched down beyond ROAD zone for longer than this → "head_down" event
HEAD_DOWN_DISTRACTION_THRESHOLD = 1.5

# Head turned toward passenger side + talking for longer than this →
# "passenger_conversation" event
PASSENGER_TALK_DISTRACTION_THRESHOLD = 3.0

# ─── Passenger Side ─────────────────────────────────────────────────────────
# Which direction is the passenger seat? Positive yaw = right in most setups.
# Set to "right" for left-hand-drive (US/EU) or "left" for right-hand-drive (UK/JP).
PASSENGER_SIDE = "right"
# Minimum |yaw| to count as "turned toward passenger" (must also be outside ROAD zone)
PASSENGER_YAW_MIN = 25.0

# ─── Head Down Detection ────────────────────────────────────────────────────
# Pitch below this (more negative = looking further down) qualifies as "head down"
HEAD_DOWN_PITCH_THRESHOLD = -20.0

# ─── Mouth / Talking Detection ──────────────────────────────────────────────
# Mouth Aspect Ratio (MAR) above this = mouth is open (talking / yawning)
MAR_OPEN_THRESHOLD = 0.45
# Number of consecutive "mouth open" frames to confirm talking
MAR_TALK_MIN_FRAMES = 3

# ─── Eye Aspect Ratio (EAR) for drowsiness / eye closure ───────────────────
EAR_THRESHOLD = 0.21
DROWSINESS_TIME_THRESHOLD = 1.5    # seconds of sustained eye closure

# ─── Gaze Aversion (iris tracking, supplements head pose) ──────────────────
GAZE_HORIZONTAL_THRESHOLD = 0.35
GAZE_VERTICAL_THRESHOLD = 0.30

# ─── Phone Detection (YOLOv8) ──────────────────────────────────────────────
PHONE_CLASS_ID = 67
PHONE_CONFIDENCE_THRESHOLD = 0.40

# ─── Event Logging ──────────────────────────────────────────────────────────
# Minimum duration (seconds) for ANY event to be written to CSV
MIN_EVENT_DURATION = 0.3
# Duration (seconds) beyond which an event is flagged "high risk" in the report
HIGH_RISK_DURATION = 2.0

# ─── Severity Levels ───────────────────────────────────────────────────────
SEVERITY_LOW = "low"
SEVERITY_MEDIUM = "medium"
SEVERITY_HIGH = "high"

# ─── Analysis Sampling ─────────────────────────────────────────────────────
YOLO_PROCESS_INTERVAL = 3      # run YOLO every Nth frame
FACE_MESH_PROCESS_INTERVAL = 1 # run FaceLandmarker every Nth frame

# ─── Report ────────────────────────────────────────────────────────────────
TIMELINE_RESOLUTION_SECONDS = 1
TOP_WORST_MOMENTS = 5
REPORT_TEMPLATE_NAME = "report.html"
