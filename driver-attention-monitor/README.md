# Driver Attention Monitoring System

A Python-based system that silently records dashcam footage during a drive, performs background distraction analysis, and generates a detailed HTML report after the drive ends. **No real-time alerts or on-screen overlays** — the system runs quietly and produces results only when the session is complete.

## Features

- **Silent recording** — captures webcam video with no visual output
- **Background analysis** — detects distractions in real time and logs them to CSV
- **Post-drive HTML report** — attention timeline, distraction breakdown, worst moments with thumbnails
- **Distraction detection**:
  - Eye closure / drowsiness via Eye Aspect Ratio (EAR)
  - Gaze aversion via iris tracking
  - Head pose deviation (looking down, sideways, up)
  - Phone presence via YOLOv8n object detection
- **High risk flagging** — any event lasting >2 seconds is marked high risk

## Setup

```bash
# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # macOS/Linux
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

YOLOv8n weights (`yolov8n.pt`) will auto-download on first run.

## Usage

### 1. Record a drive session

```bash
# Record until you press Ctrl+C
python record.py

# Or record for a fixed duration (seconds)
python record.py --duration 120
```

This creates a timestamped session folder under `sessions/` containing:
- `dashcam.mp4` — the recorded video
- `events.csv` — all detected distraction events
- `meta.txt` — session metadata

### 2. Analyze an existing video (iPhone, dashcam file, etc.)

```bash
# Analyze any video file — mp4, mov, avi, mkv
python analyze_video.py /path/to/dashcam.mov

# Optionally specify output location
python analyze_video.py recording.mp4 --output my_report.html
```

This runs the full detection pipeline on the video, creates a session folder
with `events.csv` and `meta.txt`, and generates the HTML report in one step.

### 3. Generate a report from an existing session

```bash
python analyze.py sessions/20260407_143022
```

Opens or saves an HTML report with:
- Total drive time and distraction percentage
- Attention timeline (green = attentive, red = distracted)
- Pie chart breakdown of distraction types
- Top worst moments with video frame thumbnails

## Configuration

All thresholds and parameters are in `config.py`:

| Setting | Default | Description |
|---------|---------|-------------|
| `EAR_THRESHOLD` | 0.21 | Eye aspect ratio below which eyes are "closed" |
| `GAZE_HORIZONTAL_THRESHOLD` | 0.35 | Horizontal gaze offset ratio for aversion |
| `HEAD_YAW_THRESHOLD` | 30.0 | Degrees of head yaw for "looking sideways" |
| `HEAD_PITCH_DOWN_THRESHOLD` | -20.0 | Degrees of pitch for "looking down" |
| `PHONE_CONFIDENCE_THRESHOLD` | 0.40 | YOLO confidence for phone detection |
| `HIGH_RISK_DURATION` | 2.0 | Seconds after which an event is flagged high risk |
| `MIN_EVENT_DURATION` | 0.3 | Minimum event duration to be logged |

## Project Structure

```
driver-attention-monitor/
├── config.py          # Thresholds and settings
├── record.py          # Recording + background analysis
├── analyze.py         # Post-drive report generation from session data
├── analyze_video.py   # Analyze a pre-recorded video file end-to-end
├── requirements.txt   # Python dependencies
├── README.md          # This file
└── sessions/          # Created at runtime
    └── 20260407_143022/
        ├── dashcam.mp4
        ├── events.csv
        ├── meta.txt
        └── report.html
```

## Technical Stack

- **OpenCV** — video capture and frame extraction
- **MediaPipe Face Mesh** — 478 facial landmarks + iris tracking
- **YOLOv8n** — lightweight phone detection
- **Matplotlib** — timeline and breakdown charts
- **Jinja2** — HTML report templating

## CSV Format

The `events.csv` has columns:

| Column | Description |
|--------|-------------|
| `timestamp` | Seconds from drive start |
| `event_type` | `eye_closure`, `head_down`, `distracted_offroad`, `passenger_convo`, `phone_detected` |
| `severity` | `low`, `medium`, `high` |
| `duration` | Event duration in seconds |
