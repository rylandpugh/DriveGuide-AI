"""
record.py — Silent drive session recorder with background distraction analysis.

Usage:
    python record.py              # Start recording; press Ctrl+C to stop
    python record.py --duration 60  # Record for 60 seconds then stop automatically

No visual output is shown while recording. All distraction events are logged
to a CSV file alongside the video for post-drive analysis.

Detection uses a **zone-based** system:
  ROAD       — forward gaze (yaw ±20°), driver is attentive
  GLANCE     — off-road look < 1.5 s (mirror/speedometer), never flagged
  DISTRACTED — sustained off-road look that exceeds a duration threshold

Distraction rules (only events exceeding thresholds are logged):
  • head_down          — pitch below −20° for > 1.5 s
  • passenger_convo    — turned toward passenger + talking for > 3 s
  • distracted_offroad — any off-road gaze sustained > 2.5 s
  • eye_closure        — EAR below threshold for > 1.5 s (drowsiness)
  • phone_detected     — cell phone visible in frame (YOLO)

Uses the MediaPipe Tasks API (FaceLandmarker) for facial landmark detection.
"""

import argparse
import csv
import json
import os
import signal
import sys
import time
import urllib.request
from datetime import datetime
from enum import Enum, auto

import cv2
import mediapipe as mp
import numpy as np
from ultralytics import YOLO

import config


# ── Model Download ───────────────────────────────────────────────────────────

def ensure_face_landmarker_model():
    """Download the FaceLandmarker model if not already present."""
    model_path = config.FACE_LANDMARKER_MODEL_PATH
    if os.path.exists(model_path):
        return model_path
    print("[Driver Monitor] Downloading FaceLandmarker model...")
    urllib.request.urlretrieve(config.FACE_LANDMARKER_MODEL_URL, model_path)
    print(f"[Driver Monitor] Model saved to {model_path}")
    return model_path


# ── Landmark Indices ─────────────────────────────────────────────────────────

# Eye landmarks for EAR
LEFT_EYE_IDX = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_IDX = [33, 160, 158, 133, 153, 144]

# Iris landmarks for gaze supplementation
LEFT_IRIS_IDX = [468, 469, 470, 471, 472]
RIGHT_IRIS_IDX = [473, 474, 475, 476, 477]

# Head pose estimation via solvePnP (nose, chin, eye corners, mouth corners)
HEAD_POSE_IDX = [1, 152, 33, 263, 61, 291]

# Inner lip landmarks for Mouth Aspect Ratio (MAR)
# Upper inner lip: 13, Lower inner lip: 14, Left corner: 78, Right corner: 308
UPPER_LIP_IDX = 13
LOWER_LIP_IDX = 14
LIP_LEFT_IDX = 78
LIP_RIGHT_IDX = 308


# ── Geometry Helpers ─────────────────────────────────────────────────────────

def _lm_np(landmark, w, h):
    """Convert a normalised landmark to pixel coordinates."""
    return np.array([landmark.x * w, landmark.y * h])


def compute_ear(landmarks, eye_indices, w, h):
    """Eye Aspect Ratio for one eye."""
    pts = [_lm_np(landmarks[i], w, h) for i in eye_indices]
    v1 = np.linalg.norm(pts[1] - pts[5])
    v2 = np.linalg.norm(pts[2] - pts[4])
    horiz = np.linalg.norm(pts[0] - pts[3])
    if horiz == 0:
        return 0.3
    return (v1 + v2) / (2.0 * horiz)


def compute_mar(landmarks, w, h):
    """Mouth Aspect Ratio — vertical lip opening / horizontal lip width."""
    upper = _lm_np(landmarks[UPPER_LIP_IDX], w, h)
    lower = _lm_np(landmarks[LOWER_LIP_IDX], w, h)
    left = _lm_np(landmarks[LIP_LEFT_IDX], w, h)
    right = _lm_np(landmarks[LIP_RIGHT_IDX], w, h)
    vert = np.linalg.norm(upper - lower)
    horiz = np.linalg.norm(left - right)
    if horiz == 0:
        return 0.0
    return vert / horiz


def estimate_head_pose(landmarks, w, h):
    """Return (yaw, pitch) in degrees via solvePnP."""
    model_points = np.array([
        [0.0, 0.0, 0.0],           # Nose tip
        [0.0, -330.0, -65.0],      # Chin
        [-225.0, 170.0, -135.0],   # Left eye corner
        [225.0, 170.0, -135.0],    # Right eye corner
        [-150.0, -150.0, -125.0],  # Left mouth corner
        [150.0, -150.0, -125.0],   # Right mouth corner
    ], dtype=np.float64)

    image_points = np.array(
        [_lm_np(landmarks[i], w, h) for i in HEAD_POSE_IDX],
        dtype=np.float64,
    )

    focal_length = w
    center = (w / 2, h / 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1],
    ], dtype=np.float64)
    dist_coeffs = np.zeros((4, 1))

    success, rvec, _ = cv2.solvePnP(
        model_points, image_points, camera_matrix, dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not success:
        return 0.0, 0.0

    rmat, _ = cv2.Rodrigues(rvec)
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
    return angles[1], angles[0]  # yaw, pitch


# ── Calibration ──────────────────────────────────────────────────────────────

def run_calibration_live(cap, face_landmarker, w, h, duration=None):
    """Capture frames from a live camera for *duration* seconds, return average
    (yaw, pitch) as the neutral baseline.  Prints a countdown to stdout."""
    if duration is None:
        duration = config.CALIBRATION_DURATION

    print(f"[Calibration] Look straight at the road for {duration:.0f} seconds...")
    yaw_samples = []
    pitch_samples = []
    start = time.time()

    while time.time() - start < duration:
        ret, frame = cap.read()
        if not ret:
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int((time.time() - start) * 1000)
        result = face_landmarker.detect_for_video(mp_image, timestamp_ms)
        if result.face_landmarks:
            yaw, pitch = estimate_head_pose(result.face_landmarks[0], w, h)
            yaw_samples.append(yaw)
            pitch_samples.append(pitch)
        remaining = duration - (time.time() - start)
        if remaining > 0:
            print(f"\r[Calibration] {remaining:.1f}s remaining  ", end="", flush=True)

    print()  # newline after countdown

    if len(yaw_samples) < 5:
        print("[Calibration] WARNING: Too few face samples — using zero offset.")
        return 0.0, 0.0

    yaw_off = float(np.mean(yaw_samples))
    pitch_off = float(np.mean(pitch_samples))
    print(f"[Calibration] Neutral baseline: yaw={yaw_off:+.1f}°, pitch={pitch_off:+.1f}°  "
          f"({len(yaw_samples)} samples)")
    return yaw_off, pitch_off


def run_calibration_video(cap, face_landmarker, w, h, fps, duration=None):
    """Read the first *duration* seconds of a video file and return the average
    (yaw, pitch) as the neutral baseline.  The seek position is left at the
    end of the calibration window — the caller should continue reading from
    there for the main analysis."""
    if duration is None:
        duration = config.CALIBRATION_DURATION

    cal_frames = int(duration * fps)
    yaw_samples = []
    pitch_samples = []

    print(f"[Calibration] Using first {duration:.1f}s ({cal_frames} frames) as neutral baseline...")

    for i in range(cal_frames):
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int((i / fps) * 1000)
        result = face_landmarker.detect_for_video(mp_image, timestamp_ms)
        if result.face_landmarks:
            yaw, pitch = estimate_head_pose(result.face_landmarks[0], w, h)
            yaw_samples.append(yaw)
            pitch_samples.append(pitch)

    if len(yaw_samples) < 5:
        print("[Calibration] WARNING: Too few face samples — using zero offset.")
        return 0.0, 0.0

    yaw_off = float(np.mean(yaw_samples))
    pitch_off = float(np.mean(pitch_samples))
    print(f"[Calibration] Neutral baseline: yaw={yaw_off:+.1f}°, pitch={pitch_off:+.1f}°  "
          f"({len(yaw_samples)} samples)")
    return yaw_off, pitch_off


def save_calibration(session_dir, yaw_offset, pitch_offset):
    path = os.path.join(session_dir, "calibration.json")
    with open(path, "w") as f:
        json.dump({"yaw_offset": yaw_offset, "pitch_offset": pitch_offset}, f, indent=2)
    return path


def load_calibration(session_dir):
    path = os.path.join(session_dir, "calibration.json")
    if not os.path.exists(path):
        return 0.0, 0.0
    with open(path) as f:
        data = json.load(f)
    return data.get("yaw_offset", 0.0), data.get("pitch_offset", 0.0)


# ── Zone Classification ──────────────────────────────────────────────────────

class HeadZone(Enum):
    ROAD = auto()
    GLANCE = auto()          # transient — may become DISTRACTED
    HEAD_DOWN = auto()       # pitch below threshold
    PASSENGER = auto()       # turned toward passenger seat
    OFFROAD = auto()         # generic off-road (not down, not passenger side)


def classify_zone(yaw, pitch, yaw_offset=0.0, pitch_offset=0.0):
    """Map (yaw, pitch) to a HeadZone.

    *yaw_offset* and *pitch_offset* are the calibration baseline — subtracted
    from the raw angles so that zone boundaries are relative to the driver's
    natural forward-facing pose rather than absolute zero.
    """
    yaw = yaw - yaw_offset
    pitch = pitch - pitch_offset

    in_road = (
        abs(yaw) <= config.ROAD_YAW_TOLERANCE
        and config.ROAD_PITCH_DOWN_LIMIT <= pitch <= config.ROAD_PITCH_UP_LIMIT
    )
    if in_road:
        return HeadZone.ROAD

    # Head down?
    if pitch < config.HEAD_DOWN_PITCH_THRESHOLD:
        return HeadZone.HEAD_DOWN

    # Toward passenger?
    if config.PASSENGER_SIDE == "right" and yaw > config.PASSENGER_YAW_MIN:
        return HeadZone.PASSENGER
    if config.PASSENGER_SIDE == "left" and yaw < -config.PASSENGER_YAW_MIN:
        return HeadZone.PASSENGER

    return HeadZone.OFFROAD


# ── Zone Tracker (duration-gated distraction state machine) ──────────────────

class ZoneTracker:
    """
    Tracks how long the driver has been in each off-road zone and decides
    whether the duration crosses into a distraction event.

    State:
        current_zone  — last classified HeadZone
        zone_start    — timestamp when the current zone began
        talking_frames — rolling count of consecutive mouth-open frames
        active_events — set of event types whose start has been recorded
    """

    def __init__(self):
        self.current_zone = HeadZone.ROAD
        self.zone_start = None
        self.talking_frames = 0
        self.active_events = set()  # event types currently "open"

    def update(self, zone: HeadZone, is_mouth_open: bool, now: float, tracker):
        """
        Called every analysed frame. Decides whether to open / close events
        on the EventTracker based on zone + duration.
        """
        # ── Zone transition ──
        if zone != self.current_zone:
            # Close any events from the previous zone that no longer apply
            self._close_stale_events(zone, now, tracker)
            self.current_zone = zone
            self.zone_start = now
            self.talking_frames = 0

        if zone == HeadZone.ROAD:
            # Driver is attentive — close every open zone event
            self._close_all(now, tracker)
            self.talking_frames = 0
            return

        # ── Off-road: compute dwell time ──
        dwell = now - self.zone_start if self.zone_start else 0.0

        # Still within the glance grace period → no events
        if dwell < config.GLANCE_GRACE_PERIOD:
            return

        # Track talking
        if is_mouth_open:
            self.talking_frames += 1
        else:
            self.talking_frames = max(0, self.talking_frames - 1)

        is_talking = self.talking_frames >= config.MAR_TALK_MIN_FRAMES

        # ── HEAD DOWN ──
        if zone == HeadZone.HEAD_DOWN and dwell >= config.HEAD_DOWN_DISTRACTION_THRESHOLD:
            if "head_down" not in self.active_events:
                tracker.start_event("head_down", config.SEVERITY_MEDIUM, self.zone_start)
                self.active_events.add("head_down")

        # ── PASSENGER CONVERSATION ──
        if zone == HeadZone.PASSENGER and is_talking and dwell >= config.PASSENGER_TALK_DISTRACTION_THRESHOLD:
            if "passenger_convo" not in self.active_events:
                tracker.start_event("passenger_convo", config.SEVERITY_MEDIUM, self.zone_start)
                self.active_events.add("passenger_convo")

        # ── GENERIC OFF-ROAD (any non-ROAD zone, catch-all) ──
        if dwell >= config.OFFROAD_DISTRACTION_THRESHOLD:
            if "distracted_offroad" not in self.active_events:
                tracker.start_event("distracted_offroad", config.SEVERITY_HIGH, self.zone_start)
                self.active_events.add("distracted_offroad")

    # ── Internal helpers ──

    def _close_stale_events(self, new_zone, now, tracker):
        """Close events that no longer match the new zone."""
        to_close = set()
        if new_zone != HeadZone.HEAD_DOWN and "head_down" in self.active_events:
            to_close.add("head_down")
        if new_zone != HeadZone.PASSENGER and "passenger_convo" in self.active_events:
            to_close.add("passenger_convo")
        # Off-road catch-all closes whenever we return to ROAD (handled in _close_all)
        # or when the specific sub-zone changes — keep it open across sub-zone
        # transitions since the driver is still off-road.
        if new_zone == HeadZone.ROAD and "distracted_offroad" in self.active_events:
            to_close.add("distracted_offroad")
        for etype in to_close:
            tracker.end_event(etype, now)
            self.active_events.discard(etype)

    def _close_all(self, now, tracker):
        for etype in list(self.active_events):
            tracker.end_event(etype, now)
        self.active_events.clear()


# ── Event Tracker (CSV writer) ──────────────────────────────────────────────

class EventTracker:
    """Tracks ongoing distraction events and writes completed ones to CSV."""

    def __init__(self, csv_path, drive_start_time):
        self.csv_path = csv_path
        self.drive_start = drive_start_time
        self._active = {}  # event_type -> {"start": float, "severity": str}
        self._csv_file = open(csv_path, "w", newline="")
        self._writer = csv.writer(self._csv_file)
        self._writer.writerow(["timestamp", "event_type", "severity", "duration"])
        self._csv_file.flush()

    def start_event(self, event_type, severity, now):
        if event_type not in self._active:
            self._active[event_type] = {"start": now, "severity": severity}

    def end_event(self, event_type, now):
        if event_type in self._active:
            ev = self._active.pop(event_type)
            duration = now - ev["start"]
            if duration >= config.MIN_EVENT_DURATION:
                severity = ev["severity"]
                if duration >= config.HIGH_RISK_DURATION:
                    severity = config.SEVERITY_HIGH
                rel_ts = ev["start"] - self.drive_start
                self._writer.writerow([
                    f"{rel_ts:.2f}",
                    event_type,
                    severity,
                    f"{duration:.2f}",
                ])
                self._csv_file.flush()

    def end_all(self, now):
        for etype in list(self._active):
            self.end_event(etype, now)

    def close(self):
        self._csv_file.close()


# ── Main Recording Loop ─────────────────────────────────────────────────────

def run_session(duration=None, skip_calibration=False):
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = os.path.join(config.OUTPUT_DIR, session_id)
    os.makedirs(session_dir, exist_ok=True)

    video_path = os.path.join(session_dir, "dashcam.mp4")
    csv_path = os.path.join(session_dir, "events.csv")
    meta_path = os.path.join(session_dir, "meta.txt")

    print(f"[Driver Monitor] Session: {session_id}")
    print(f"[Driver Monitor] Video:   {video_path}")
    print(f"[Driver Monitor] Events:  {csv_path}")

    # ── Camera ──
    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, config.FPS)

    if not cap.isOpened():
        print("[Driver Monitor] ERROR: Cannot open camera. Exiting.")
        sys.exit(1)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS) or config.FPS

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(video_path, fourcc, actual_fps, (actual_w, actual_h))

    # ── FaceLandmarker (Tasks API) ──
    model_path = ensure_face_landmarker_model()
    BaseOptions = mp.tasks.BaseOptions
    FaceLandmarker = mp.tasks.vision.FaceLandmarker
    FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    face_landmarker = FaceLandmarker.create_from_options(FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.VIDEO,
        num_faces=config.FACE_LANDMARKER_MAX_FACES,
        min_face_detection_confidence=config.FACE_LANDMARKER_MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=config.FACE_LANDMARKER_MIN_TRACKING_CONFIDENCE,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    ))

    # ── Calibration ──
    if skip_calibration:
        yaw_offset, pitch_offset = 0.0, 0.0
        print("[Driver Monitor] Calibration skipped — using absolute zero as baseline.")
    else:
        yaw_offset, pitch_offset = run_calibration_live(
            cap, face_landmarker, actual_w, actual_h,
        )
        # Calibration consumed the landmarker's timestamp sequence; recreate it
        # so the recording starts from timestamp 0.
        face_landmarker.close()
        face_landmarker = FaceLandmarker.create_from_options(FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.VIDEO,
            num_faces=config.FACE_LANDMARKER_MAX_FACES,
            min_face_detection_confidence=config.FACE_LANDMARKER_MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=config.FACE_LANDMARKER_MIN_TRACKING_CONFIDENCE,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        ))

    save_calibration(session_dir, yaw_offset, pitch_offset)

    # ── YOLO ──
    print("[Driver Monitor] Loading YOLOv8n for phone detection...")
    yolo = YOLO(config.YOLO_MODEL_PATH)
    yolo.predict(np.zeros((64, 64, 3), dtype=np.uint8), verbose=False)  # warm-up

    # ── State ──
    drive_start = time.time()
    tracker = EventTracker(csv_path, drive_start)
    zone_tracker = ZoneTracker()
    frame_count = 0
    last_phone_detected = False

    # Graceful shutdown
    stop_flag = [False]

    def _on_signal(sig, frame):
        stop_flag[0] = True

    signal.signal(signal.SIGINT, _on_signal)

    print("[Driver Monitor] Recording... Press Ctrl+C to stop.")

    try:
        while not stop_flag[0]:
            ret, frame = cap.read()
            if not ret:
                break

            now = time.time()
            frame_count += 1
            writer.write(frame)

            if duration and (now - drive_start) >= duration:
                break

            # ── FaceLandmarker Analysis ──
            if frame_count % config.FACE_MESH_PROCESS_INTERVAL == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                timestamp_ms = int((now - drive_start) * 1000)

                result = face_landmarker.detect_for_video(mp_image, timestamp_ms)

                if result.face_landmarks:
                    lm = result.face_landmarks[0]

                    # ── Head pose → zone classification (calibration-adjusted) ──
                    yaw, pitch = estimate_head_pose(lm, actual_w, actual_h)
                    zone = classify_zone(yaw, pitch, yaw_offset, pitch_offset)

                    # ── Mouth movement (talking detection) ──
                    mar = compute_mar(lm, actual_w, actual_h)
                    is_mouth_open = mar > config.MAR_OPEN_THRESHOLD

                    # ── Zone-based distraction logic ──
                    zone_tracker.update(zone, is_mouth_open, now, tracker)

                    # ── EAR (drowsiness / eye closure) ──
                    ear_l = compute_ear(lm, LEFT_EYE_IDX, actual_w, actual_h)
                    ear_r = compute_ear(lm, RIGHT_EYE_IDX, actual_w, actual_h)
                    ear = (ear_l + ear_r) / 2.0

                    if ear < config.EAR_THRESHOLD:
                        tracker.start_event("eye_closure", config.SEVERITY_MEDIUM, now)
                    else:
                        # Only log if the closure exceeded the drowsiness threshold
                        if "eye_closure" in tracker._active:
                            dwell = now - tracker._active["eye_closure"]["start"]
                            if dwell >= config.DROWSINESS_TIME_THRESHOLD:
                                tracker.end_event("eye_closure", now)
                            else:
                                # Eyes opened before threshold — discard silently
                                tracker._active.pop("eye_closure", None)
                        # (if not active, nothing to do)

                else:
                    # No face detected → treat as off-road
                    zone_tracker.update(HeadZone.OFFROAD, False, now, tracker)
                    # Discard any in-progress eye closure (can't measure without face)
                    tracker._active.pop("eye_closure", None)

            # ── YOLO Phone Detection ──
            if frame_count % config.YOLO_PROCESS_INTERVAL == 0:
                results = yolo.predict(frame, verbose=False, conf=config.PHONE_CONFIDENCE_THRESHOLD)
                phone_found = False
                for r in results:
                    for box in r.boxes:
                        if int(box.cls[0]) == config.PHONE_CLASS_ID:
                            phone_found = True
                            break

                if phone_found:
                    tracker.start_event("phone_detected", config.SEVERITY_HIGH, now)
                    last_phone_detected = True
                elif last_phone_detected:
                    tracker.end_event("phone_detected", now)
                    last_phone_detected = False

    finally:
        drive_end = time.time()
        zone_tracker._close_all(drive_end, tracker)
        tracker.end_all(drive_end)
        tracker.close()

        cap.release()
        writer.release()
        face_landmarker.close()

        total_seconds = drive_end - drive_start
        with open(meta_path, "w") as f:
            f.write(f"session_id={session_id}\n")
            f.write(f"start_time={datetime.fromtimestamp(drive_start).isoformat()}\n")
            f.write(f"end_time={datetime.fromtimestamp(drive_end).isoformat()}\n")
            f.write(f"total_seconds={total_seconds:.2f}\n")
            f.write(f"total_frames={frame_count}\n")
            f.write(f"resolution={actual_w}x{actual_h}\n")
            f.write(f"fps={actual_fps}\n")

        mins = int(total_seconds // 60)
        secs = int(total_seconds % 60)
        print(f"\n[Driver Monitor] Session ended. Duration: {mins}m {secs}s, Frames: {frame_count}")
        print(f"[Driver Monitor] Run report:  python analyze.py {session_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Driver Attention Monitor — Recording Mode")
    parser.add_argument("--duration", type=int, default=None,
                        help="Auto-stop after N seconds (default: manual stop with Ctrl+C)")
    parser.add_argument("--skip-calibration", action="store_true",
                        help="Skip the calibration step (use absolute zero as baseline)")
    args = parser.parse_args()
    run_session(duration=args.duration, skip_calibration=args.skip_calibration)
