"""
analyze.py — Post-drive report generator.

Usage:
    python analyze.py sessions/20260407_143022
    python analyze.py sessions/20260407_143022 --output report.html

Reads the events.csv and meta.txt from a completed session and generates
an HTML report with attention timeline, distraction breakdown, and worst moments.
"""

import argparse
import base64
import csv
import io
import os
import sys
from collections import defaultdict
from datetime import timedelta

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from jinja2 import Template

import config

# ── Data Loading ─────────────────────────────────────────────────────────────

def load_meta(session_dir):
    meta = {}
    meta_path = os.path.join(session_dir, "meta.txt")
    if not os.path.exists(meta_path):
        print(f"ERROR: meta.txt not found in {session_dir}")
        sys.exit(1)
    with open(meta_path) as f:
        for line in f:
            key, val = line.strip().split("=", 1)
            meta[key] = val
    meta["total_seconds"] = float(meta["total_seconds"])
    meta["total_frames"] = int(meta["total_frames"])
    meta["fps"] = float(meta["fps"])
    return meta


def load_events(session_dir):
    csv_path = os.path.join(session_dir, "events.csv")
    events = []
    if not os.path.exists(csv_path):
        print(f"ERROR: events.csv not found in {session_dir}")
        sys.exit(1)
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            events.append({
                "timestamp": float(row["timestamp"]),
                "event_type": row["event_type"],
                "severity": row["severity"],
                "duration": float(row["duration"]),
            })
    return events


# ── Analysis ─────────────────────────────────────────────────────────────────

def compute_stats(events, total_seconds):
    stats = {
        "total_events": len(events),
        "total_distracted_time": 0.0,
        "high_risk_events": [],
        "type_breakdown": defaultdict(lambda: {"count": 0, "total_duration": 0.0}),
    }

    covered = []  # list of (start, end) intervals for union calculation

    for ev in events:
        start = ev["timestamp"]
        end = start + ev["duration"]
        covered.append((start, end))

        stats["type_breakdown"][ev["event_type"]]["count"] += 1
        stats["type_breakdown"][ev["event_type"]]["total_duration"] += ev["duration"]

        if ev["duration"] >= config.HIGH_RISK_DURATION:
            stats["high_risk_events"].append(ev)

    # Merge overlapping intervals to get true distracted time
    if covered:
        covered.sort()
        merged = [covered[0]]
        for s, e in covered[1:]:
            if s <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], e))
            else:
                merged.append((s, e))
        stats["total_distracted_time"] = sum(e - s for s, e in merged)

    stats["distracted_pct"] = (
        (stats["total_distracted_time"] / total_seconds * 100) if total_seconds > 0 else 0
    )

    # Top worst moments sorted by duration
    worst = sorted(events, key=lambda e: e["duration"], reverse=True)
    stats["worst_moments"] = worst[: config.TOP_WORST_MOMENTS]

    return stats


def build_timeline(events, total_seconds):
    """Build per-second attention timeline. 1 = attentive, 0 = distracted."""
    n_bins = max(1, int(total_seconds / config.TIMELINE_RESOLUTION_SECONDS) + 1)
    timeline = np.ones(n_bins, dtype=float)

    for ev in events:
        start_bin = int(ev["timestamp"] / config.TIMELINE_RESOLUTION_SECONDS)
        end_bin = int((ev["timestamp"] + ev["duration"]) / config.TIMELINE_RESOLUTION_SECONDS)
        start_bin = max(0, start_bin)
        end_bin = min(n_bins - 1, end_bin)
        for b in range(start_bin, end_bin + 1):
            timeline[b] = 0.0

    return timeline


# ── Visualisation ────────────────────────────────────────────────────────────

def render_timeline_chart(timeline, total_seconds):
    """Render attention timeline as a PNG returned as base64 string."""
    fig, ax = plt.subplots(figsize=(12, 2.5))
    n = len(timeline)
    x = np.arange(n) * config.TIMELINE_RESOLUTION_SECONDS

    colors = ["#e74c3c" if v == 0 else "#2ecc71" for v in timeline]
    ax.bar(x, np.ones(n), width=config.TIMELINE_RESOLUTION_SECONDS,
           color=colors, edgecolor="none")

    ax.set_xlim(0, total_seconds)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_xlabel("Time (seconds)")
    ax.set_title("Attention Timeline")

    green_patch = mpatches.Patch(color="#2ecc71", label="Attentive")
    red_patch = mpatches.Patch(color="#e74c3c", label="Distracted")
    ax.legend(handles=[green_patch, red_patch], loc="upper right", fontsize=8)

    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def render_breakdown_chart(type_breakdown):
    """Render distraction type breakdown as a pie chart, base64 PNG."""
    if not type_breakdown:
        return ""
    labels = list(type_breakdown.keys())
    durations = [type_breakdown[k]["total_duration"] for k in labels]

    display_labels = [l.replace("_", " ").title() for l in labels]

    fig, ax = plt.subplots(figsize=(5, 4))
    colors = ["#e74c3c", "#e67e22", "#f1c40f", "#9b59b6", "#3498db"]
    ax.pie(durations, labels=display_labels, autopct="%1.1f%%",
           colors=colors[: len(labels)], startangle=140)
    ax.set_title("Distraction Breakdown by Duration")

    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def extract_thumbnail(video_path, timestamp_sec, fps):
    """Extract a single frame from the video at the given timestamp, return base64 JPEG."""
    cap = cv2.VideoCapture(video_path)
    frame_no = int(timestamp_sec * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return ""
    # Resize for thumbnail
    h, w = frame.shape[:2]
    scale = 320 / w
    thumb = cv2.resize(frame, (320, int(h * scale)))
    _, buf = cv2.imencode(".jpg", thumb, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return base64.b64encode(buf.tobytes()).decode("utf-8")


# ── HTML Report ──────────────────────────────────────────────────────────────

REPORT_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Drive Report — {{ session_id }}</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
         background: #f5f6fa; color: #2d3436; padding: 2rem; }
  .container { max-width: 960px; margin: auto; }
  h1 { font-size: 1.8rem; margin-bottom: .3rem; }
  .subtitle { color: #636e72; margin-bottom: 2rem; }
  .cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
           gap: 1rem; margin-bottom: 2rem; }
  .card { background: #fff; border-radius: 12px; padding: 1.5rem;
          box-shadow: 0 2px 8px rgba(0,0,0,.07); text-align: center; }
  .card .value { font-size: 2rem; font-weight: 700; }
  .card .label { color: #636e72; font-size: .85rem; margin-top: .3rem; }
  .card.high-risk .value { color: #e74c3c; }
  .section { background: #fff; border-radius: 12px; padding: 1.5rem;
             box-shadow: 0 2px 8px rgba(0,0,0,.07); margin-bottom: 1.5rem; }
  .section h2 { font-size: 1.2rem; margin-bottom: 1rem; border-bottom: 2px solid #dfe6e9;
                 padding-bottom: .5rem; }
  .chart-img { width: 100%; max-width: 100%; }
  table { width: 100%; border-collapse: collapse; }
  th, td { padding: .6rem .8rem; text-align: left; border-bottom: 1px solid #dfe6e9; }
  th { font-weight: 600; font-size: .85rem; color: #636e72; text-transform: uppercase; }
  .severity-high { color: #e74c3c; font-weight: 700; }
  .severity-medium { color: #e67e22; font-weight: 600; }
  .severity-low { color: #f39c12; }
  .worst-card { display: flex; gap: 1rem; align-items: flex-start;
                padding: 1rem 0; border-bottom: 1px solid #dfe6e9; }
  .worst-card:last-child { border-bottom: none; }
  .worst-card img { border-radius: 8px; flex-shrink: 0; }
  .worst-card .info { flex: 1; }
  .worst-card .info h3 { font-size: 1rem; margin-bottom: .3rem; }
  .worst-card .info p { color: #636e72; font-size: .9rem; }
  .risk-badge { display: inline-block; background: #e74c3c; color: #fff;
                font-size: .7rem; padding: 2px 8px; border-radius: 10px;
                text-transform: uppercase; font-weight: 700; margin-left: .5rem; }
  .footer { text-align: center; color: #b2bec3; font-size: .8rem; margin-top: 2rem; }
</style>
</head>
<body>
<div class="container">
  <h1>Drive Attention Report</h1>
  <p class="subtitle">Session {{ session_id }} &mdash; {{ start_time }} to {{ end_time }}</p>

  <div class="cards">
    <div class="card">
      <div class="value">{{ drive_duration }}</div>
      <div class="label">Drive Duration</div>
    </div>
    <div class="card {% if distracted_pct > 20 %}high-risk{% endif %}">
      <div class="value">{{ "%.1f"|format(distracted_pct) }}%</div>
      <div class="label">Time Distracted</div>
    </div>
    <div class="card">
      <div class="value">{{ total_events }}</div>
      <div class="label">Distraction Events</div>
    </div>
    <div class="card high-risk">
      <div class="value">{{ high_risk_count }}</div>
      <div class="label">High Risk (&gt;2s)</div>
    </div>
  </div>

  <div class="section">
    <h2>Attention Timeline</h2>
    <img class="chart-img" src="data:image/png;base64,{{ timeline_chart }}" alt="Timeline">
  </div>

  {% if breakdown_chart %}
  <div class="section">
    <h2>Distraction Breakdown</h2>
    <div style="display:flex; gap:2rem; align-items:flex-start; flex-wrap:wrap;">
      <img src="data:image/png;base64,{{ breakdown_chart }}" alt="Breakdown" style="max-width:340px;">
      <table style="flex:1; min-width:300px;">
        <thead><tr><th>Type</th><th>Count</th><th>Total Duration</th></tr></thead>
        <tbody>
        {% for name, info in type_breakdown.items() %}
        <tr>
          <td>{{ name.replace("_", " ").title() }}</td>
          <td>{{ info.count }}</td>
          <td>{{ "%.1f"|format(info.total_duration) }}s</td>
        </tr>
        {% endfor %}
        </tbody>
      </table>
    </div>
  </div>
  {% endif %}

  {% if worst_moments %}
  <div class="section">
    <h2>Worst Moments</h2>
    {% for m in worst_moments %}
    <div class="worst-card">
      {% if m.thumbnail %}
      <img src="data:image/jpeg;base64,{{ m.thumbnail }}" width="320" alt="Frame at {{ m.timestamp_fmt }}">
      {% endif %}
      <div class="info">
        <h3>{{ m.event_type.replace("_", " ").title() }}
          {% if m.duration >= 2.0 %}<span class="risk-badge">High Risk</span>{% endif %}
        </h3>
        <p><strong>At:</strong> {{ m.timestamp_fmt }} &nbsp;|&nbsp;
           <strong>Duration:</strong> {{ "%.1f"|format(m.duration) }}s &nbsp;|&nbsp;
           <strong>Severity:</strong>
           <span class="severity-{{ m.severity }}">{{ m.severity.upper() }}</span>
        </p>
      </div>
    </div>
    {% endfor %}
  </div>
  {% endif %}

  <p class="footer">Generated by Driver Attention Monitor</p>
</div>
</body>
</html>
"""


def fmt_duration(seconds):
    td = timedelta(seconds=int(seconds))
    parts = []
    if td.seconds // 3600:
        parts.append(f"{td.seconds // 3600}h")
    mins = (td.seconds % 3600) // 60
    if mins:
        parts.append(f"{mins}m")
    secs = td.seconds % 60
    parts.append(f"{secs}s")
    return " ".join(parts)


def generate_report(session_dir, output_path=None):
    meta = load_meta(session_dir)
    events = load_events(session_dir)
    total_seconds = meta["total_seconds"]

    stats = compute_stats(events, total_seconds)
    timeline = build_timeline(events, total_seconds)

    # Charts
    timeline_chart = render_timeline_chart(timeline, total_seconds)
    breakdown_chart = render_breakdown_chart(dict(stats["type_breakdown"]))

    # Thumbnails for worst moments
    video_path = os.path.join(session_dir, "dashcam.mp4")
    has_video = os.path.exists(video_path)
    worst_moments = []
    for m in stats["worst_moments"]:
        thumb = ""
        if has_video:
            thumb = extract_thumbnail(video_path, m["timestamp"], meta["fps"])
        worst_moments.append({
            **m,
            "thumbnail": thumb,
            "timestamp_fmt": fmt_duration(m["timestamp"]),
        })

    # Render HTML
    template = Template(REPORT_HTML)
    html = template.render(
        session_id=meta["session_id"],
        start_time=meta["start_time"],
        end_time=meta["end_time"],
        drive_duration=fmt_duration(total_seconds),
        distracted_pct=stats["distracted_pct"],
        total_events=stats["total_events"],
        high_risk_count=len(stats["high_risk_events"]),
        timeline_chart=timeline_chart,
        breakdown_chart=breakdown_chart,
        type_breakdown=dict(stats["type_breakdown"]),
        worst_moments=worst_moments,
    )

    if output_path is None:
        output_path = os.path.join(session_dir, "report.html")

    with open(output_path, "w") as f:
        f.write(html)

    print(f"[Driver Monitor] Report saved: {output_path}")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Driver Attention Monitor — Post-Drive Report")
    parser.add_argument("session_dir", help="Path to session directory (e.g. sessions/20260407_143022)")
    parser.add_argument("--output", "-o", default=None, help="Output HTML path (default: <session>/report.html)")
    args = parser.parse_args()

    if not os.path.isdir(args.session_dir):
        print(f"ERROR: Session directory not found: {args.session_dir}")
        sys.exit(1)

    generate_report(args.session_dir, args.output)
