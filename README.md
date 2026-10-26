# DriveGuide AI 

An AI-powered dashboard camera system that detects driver distraction, drowsiness, and phone usage in real-time using computer vision.

## The Problem

Distracted driving is a leading cause of preventable car crashes:
- **16 million** car crashes annually caused by driver distraction
- **6 out of 10** teen driving accidents are caused by distracted driving
- Existing solutions are expensive, invasive, or unreliable

## The Solution

DriveGuide AI uses a simple dashboard-mounted webcam and machine learning to monitor driver attention in real-time. It detects:

**Head Position** - Alerts when driver turns head away from road  
**Eye Gaze** - Detects when eyes look sideways (not at road)  
**Drowsiness** - Identifies prolonged eye closure  
**Phone Usage** - Detects hands/objects near face  

When distraction is detected, the system provides **immediate visual and text alerts** to the driver.

## Features

- Real-time face and eye detection
- Color-coded alerts (Green/Orange/Red)
- Runs entirely locally (no cloud/privacy issues)
- Fast performance (30+ FPS)
- Simple, clear visual feedback

## Installation

### Requirements
- Python 3.12 or higher
- macOS, Windows, or Linux
- Webcam

### Setup

1. **Clone this repository:**
   ```bash
   git clone https://github.com/rylandpugh/DriveGuide-AI.git
   cd DriveGuide-AI
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Or manually install:**
   ```bash
   pip install opencv-python numpy
   ```

## How to Use

1. **Run the detector:**
   ```bash
   python distraction_detector_v2.py
   ```

2. **Position your face:**
   - Keep your face centered in the camera frame
   - Make sure eyes are clearly visible
   - Good lighting helps!

3. **See real-time feedback:**
   - **Green "ATTENTIVE"** - You're looking at the road ✓
   - **Orange "Looking away..."** - You're distracted (wait to trigger alert)
   - **Red "DISTRACTED!"** - Alert! Eyes off road for 1.5+ seconds
   - **Orange "DROWSY!"** - Eyes closed for 2+ seconds
   - **Red "PHONE DETECTED!"** - Phone/hand near face (critical)

4. **Press 'q' to quit**

## How It Works

### Detection Pipeline

```
Webcam Feed
    ↓
Face Detection (OpenCV Cascade Classifier)
    ↓
Eye Detection & Tracking
    ↓
Head Position Analysis
    ↓
Eye Gaze Direction Estimation
    ↓
Temporal Analysis (smoothing)
    ↓
Alert Generation
```

### Key Algorithms

- **Face Detection:** Haar Cascade Classifier (OpenCV)
- **Eye Detection:** Cascade Classifier + symmetry analysis
- **Gaze Detection:** Eye position relative to face width
- **Phone Detection:** Body region overlap detection
- **Smoothing:** 5-frame history buffer to reduce false positives

## Technologies Used

- **Python 3.12** - Core language
- **OpenCV** - Computer vision library
- **NumPy** - Numerical computations
- **Anaconda** - Environment management

## Project Development

### What Went Well
- Face detection works reliably across lighting conditions
- Real-time performance is solid (30+ FPS)
- Alert system is intuitive and easy to understand
- Code is modular and readable

### Challenges & Solutions

| Challenge | Solution |
|-----------|----------|
| Python 3.14 incompatibility | Downgraded to Python 3.12 via Anaconda |
| Poor eye tracking | Added histogram equalization + 5-frame smoothing |
| False phone alerts | Implemented overlap detection instead of hand tracking |
| MediaPipe library conflicts | Switched to simpler OpenCV cascade classifiers |

### What I Learned
1. **Iteration is key** - First version didn't work; refactoring made all the difference
2. **Debugging is normal** - Spent more time fixing than coding
3. **Simple solutions win** - Complex ML models weren't needed; classic CV worked better
4. **Environment setup matters** - More time on setup than expected for a HS project

## Future Improvements

- [ ] **Data Logging** - Save distraction events with timestamps
- [ ] **Dashboard UI** - Historical data visualization
- [ ] **Mobile App** - iOS/Android version
- [ ] **Deep Learning** - Train custom model for better accuracy
- [ ] **Car Integration** - Connect to vehicle's CAN bus
- [ ] **Audio Alerts** - Add sound warnings in addition to visual
- [ ] **Calibration Tool** - Let users calibrate system for their car/lighting

## Files in This Repository

- `distraction_detector_v2.py` - Main detection system
- `face_detector.py` - Standalone face detection (optional)
- `requirements.txt` - Python dependencies
- `README.md` - This file

## Performance Metrics

- **Face Detection Accuracy:** ~95% in good lighting
- **Eye Detection Accuracy:** ~85% (varies with head angle)
- **FPS:** 25-35 frames per second
- **Latency:** <50ms average detection time

## Known Limitations

- Works best with forward-facing camera angle
- Poor performance in very low light
- Sunglasses interfere with eye detection
- Requires clear view of face
- May have false positives if multiple people visible

## License

MIT License - Feel free to use, modify, and share!

## About

**Created by:** Ryland Pugh '28  
**School:** Sacred Heart Cathedral  
**Program:** Inquiry & Innovation (i2)  
**Start Date:** Spring 2026  
**Status:** Working Prototype

---

## Questions or Feedback?

Found a bug? Have an improvement idea? Open an issue on GitHub!

**Happy coding and stay safe on the roads!*
