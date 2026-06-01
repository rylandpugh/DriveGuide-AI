"""
analyze_video.py — Analyze a pre-recorded video file and generate a distraction report.

Usage:
    python analyze_video.py dashcam.mp4
    python analyze_video.py /path/to/iphone_video.mov --output my_report.html
    python analyze_video.py video.mp4 --skip-calibration
    python analyze_video.py video.mp4 --calibration-seconds 5

Accepts any video format OpenCV can read (mp4, mov, avi, mkv, etc.).
Runs the same zone-based detection pipeline as record.py, writes events.csv
and meta.txt into a session folder, then generates the HTML report.

By default the first 3 seconds of video are used as a calibration window
to establish the driver's neutral forward-facing pose.  Use --skip-calibration
to treat absolute zero as the baseline (e.g. for desk testing).
"""

import argparse
import os
import shutil
import sys
from datetime import datetime

import cv2
import mediapipe as mp
import numpy as np
from ultralytics import YOLO

import config
from analyze import generate_report
from record import (
    EventTracker,
    HeadZone,
    ZoneTracker,
    classify_zone,
    compute_ear,
    compute_mar,
    ensure_face_landmarker_model,
    estimate_head_pose,
    load_calibration,
    run_calibration_video,
    save_calibration,
    LEFT_EYE_IDX,
    RIGHT_EYE_IDX,
)


def _create_face_landmarker(model_path):
    """Create a FaceLandmarker instance in VIDEO running mode."""
    BaseOptions = mp.tasks.BaseOptions
    FaceLandmarker = mp.tasks.vision.FaceLandmarker
    FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    return FaceLandmarker.create_from_options(FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.VIDEO,
        num_faces=config.FACE_LANDMARKER_MAX_FACES,
        min_face_detection_confidence=config.FACE_LANDMARKER_MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=config.FACE_LANDMARKER_MIN_TRACKING_CONFIDENCE,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    ))


def analyze_video(video_path, output_path=None, skip_calibration=False,
                  calibration_seconds=None):
    if not os.path.isfile(video_path):
        print(f"ERROR: Video file not found: {video_path}")
        sys.exit(1)

    # ── Open video and read properties ──
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: Cannot open video: {video_path}")
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        print("WARNING: Could not read FPS from video, defaulting to 30.")
        fps = 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_seconds = total_frames / fps

    print(f"[Analyze Video] Input:  {video_path}")
    print(f"[Analyze Video] Frames: {total_frames}, FPS: {fps:.1f}, "
          f"Resolution: {w}x{h}, Duration: {total_seconds:.1f}s")

    # ── Create session directory ──
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_imported"
    session_dir = os.path.join(config.OUTPUT_DIR, session_id)
    os.makedirs(session_dir, exist_ok=True)

    csv_path = os.path.join(session_dir, "events.csv")
    meta_path = os.path.join(session_dir, "meta.txt")

    # Symlink / copy video so the report can extract thumbnails
    dest_video = os.path.join(session_dir, "dashcam.mp4")
    abs_src = os.path.abspath(video_path)
    try:
        os.symlink(abs_src, dest_video)
    except (OSError, NotImplementedError):
        print("[Analyze Video] Copying video into session folder...")
        shutil.copy2(video_path, dest_video)

    # ── Initialize models ──
    model_path = ensure_face_landmarker_model()
    face_landmarker = _create_face_landmarker(model_path)

    # ── Calibration ──
    cal_duration = calibration_seconds if calibration_seconds is not None else config.CALIBRATION_DURATION

    if skip_calibration:
        yaw_offset, pitch_offset = 0.0, 0.0
        print("[Analyze Video] Calibration skipped — using absolute zero as baseline.")
    else:
        yaw_offset, pitch_offset = run_calibration_video(
            cap, face_landmarker, w, h, fps, duration=cal_duration,
        )
        # Rewind video and recreate landmarker so the main pass starts fresh
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        face_landmarker.close()
        face_landmarker = _create_face_landmarker(model_path)

    save_calibration(session_dir, yaw_offset, pitch_offset)

    # ── YOLO ──
    print("[Analyze Video] Loading YOLOv8n for phone detection...")
    yolo = YOLO(config.YOLO_MODEL_PATH)
    yolo.predict(np.zeros((64, 64, 3), dtype=np.uint8), verbose=False)

    # ── Run detection pipeline ──
    drive_start = 0.0
    tracker = EventTracker(csv_path, drive_start)
    zone_tracker = ZoneTracker()
    last_phone_detected = False
    frame_count = 0
    last_progress = -1

    print("[Analyze Video] Processing frames...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        now = frame_count / fps  # synthetic timestamp in seconds

        # Progress indicator (every 10%)
        pct = int(frame_count / total_frames * 100) if total_frames > 0 else 0
        decade = pct // 10
        if decade != last_progress:
            last_progress = decade
            print(f"  {pct}% ({frame_count}/{total_frames})")

        # ── FaceLandmarker ──
        if frame_count % config.FACE_MESH_PROCESS_INTERVAL == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int(now * 1000)

            result = face_landmarker.detect_for_video(mp_image, timestamp_ms)

            if result.face_landmarks:
                lm = result.face_landmarks[0]

                # Head pose → zone (calibration-adjusted)
                yaw, pitch = estimate_head_pose(lm, w, h)
                zone = classify_zone(yaw, pitch, yaw_offset, pitch_offset)

                # Mouth movement
                mar = compute_mar(lm, w, h)
                is_mouth_open = mar > config.MAR_OPEN_THRESHOLD

                # Zone-based distraction
                zone_tracker.update(zone, is_mouth_open, now, tracker)

                # EAR (drowsiness)
                ear_l = compute_ear(lm, LEFT_EYE_IDX, w, h)
                ear_r = compute_ear(lm, RIGHT_EYE_IDX, w, h)
                ear = (ear_l + ear_r) / 2.0

                if ear < config.EAR_THRESHOLD:
                    tracker.start_event("eye_closure", config.SEVERITY_MEDIUM, now)
                else:
                    if "eye_closure" in tracker._active:
                        dwell = now - tracker._active["eye_closure"]["start"]
                        if dwell >= config.DROWSINESS_TIME_THRESHOLD:
                            tracker.end_event("eye_closure", now)
                        else:
                            tracker._active.pop("eye_closure", None)
            else:
                zone_tracker.update(HeadZone.OFFROAD, False, now, tracker)
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

    # ── Finalise ──
    drive_end = total_frames / fps
    zone_tracker._close_all(drive_end, tracker)
    tracker.end_all(drive_end)
    tracker.close()

    cap.release()
    face_landmarker.close()

    # Write meta.txt
    with open(meta_path, "w") as f:
        f.write(f"session_id={session_id}\n")
        f.write(f"start_time={datetime.now().isoformat()}\n")
        f.write(f"end_time={datetime.now().isoformat()}\n")
        f.write(f"total_seconds={drive_end:.2f}\n")
        f.write(f"total_frames={frame_count}\n")
        f.write(f"resolution={w}x{h}\n")
        f.write(f"fps={fps}\n")

    mins = int(drive_end // 60)
    secs = int(drive_end % 60)
    print(f"[Analyze Video] Detection complete. {frame_count} frames, {mins}m {secs}s")
    print(f"[Analyze Video] Events CSV: {csv_path}")

    # ── Generate HTML report ──
    report_path = generate_report(session_dir, output_path)
    return report_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze a pre-recorded video for driver distraction and generate an HTML report.",
    )
    parser.add_argument("video", help="Path to the video file (mp4, mov, avi, mkv, etc.)")
    parser.add_argument("--output", "-o", default=None,
                        help="Output HTML path (default: sessions/<id>/report.html)")
    parser.add_argument("--skip-calibration", action="store_true",
                        help="Skip calibration; use absolute zero as baseline (for desk testing)")
    parser.add_argument("--calibration-seconds", type=float, default=None,
                        help=f"Seconds of video to use for calibration (default: {config.CALIBRATION_DURATION})")
    args = parser.parse_args()
    analyze_video(args.video, args.output,
                  skip_calibration=args.skip_calibration,
                  calibration_seconds=args.calibration_seconds)
