# DrowsyGuard v2.0 — Driver Drowsiness Detection System

DrowsyGuard v2.0 is an AI-powered driver monitoring system designed to detect signs of driver fatigue and drowsiness in real time using computer vision and deep learning techniques.  
The project combines OpenCV, MediaPipe, and TensorFlow/Keras to analyze facial landmarks, detect eye states, and trigger alerts when drowsiness is detected.

The system can be used as:

- A real-time web application with FastAPI
- A standalone desktop detection application
- A training and evaluation environment for eye-state classification models

---

# Features

- Real-time face and eye detection
- Eye state classification (Open / Closed)
- Drowsiness alert system
- Deep learning model using TensorFlow/Keras
- FastAPI web interface
- Standalone OpenCV application
- Model evaluation metrics and visualizations
- MediaPipe facial landmark tracking for accurate eye localization

---

# Technologies Used

- Python
- OpenCV
- TensorFlow / Keras
- FastAPI
- MediaPipe
- NumPy
- Matplotlib

---

# Role of MediaPipe

MediaPipe is used for facial landmark detection and face mesh tracking.  
It provides highly accurate and lightweight real-time detection of facial key points, especially around the eyes.

In DrowsyGuard, MediaPipe helps to:

- Detect facial landmarks in real time
- Track eye regions precisely
- Improve eye localization accuracy
- Reduce false detections caused by lighting or head movement
- Enhance drowsiness analysis performance

By combining MediaPipe with OpenCV and the deep learning eye classifier, the system achieves more robust and efficient driver monitoring.

---

# Project Structure

````text
drowsyguard/
├── app.py                  ← FastAPI server (entry point)
├── requirements.txt
├── src/
│   ├── config.py            ← Application settings and constants
│   ├── detector.py          ← Core drowsiness detection engine
│   ├── detect_drowsiness.py ← Standalone OpenCV application
│   ├── train_model.py       ← CNN model training script
│   ├── metrics.py           ← Model evaluation metrics
│   └── visualisation.py     ← Accuracy/loss visualization graphs
│
├── static/
│   ├── index.html           ← Frontend interface
│   ├── style.css            ← Web application styling
│   └── app.js               ← Frontend logic
│
├── Models/
│   └── eye_model.keras      ← Trained eye-state classification model
│
├── haarcascade/
│   ├── haarcascade_frontalface_default.xml
│   ├── haarcascade_eye.xml
│   └── haarcascade_smile.xml
│
└── mrleyedataset/      ← download the mrleyedataset from kaggle
    ├── Close-Eyes/
    └── Open-Eyes/
## Setup

```bash
pip install -r requirements.txt
````

## Run web app

```bash
uvicorn app:app --reload
```

Open: http://localhost:8000

## Run standalone

```bash
python src/detect_drowsiness.py
```

## Train model

```bash
python src/train_model.py
```

## Evaluate model

```bash
python src/metrics.py
```
