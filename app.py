"""
DrowsyGuard — FastAPI WebSocket server
"""
"""
DrowsyGuard — FastAPI WebSocket server (Updated for MediaPipe)
"""
import cv2
import numpy as np
import base64
import asyncio
import threading
import winsound
import queue
import time
from datetime import datetime
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from config import *
from detector import Detector, DriverStatus

# ── App ────────────────────────────────────────────
app = FastAPI(title="DrowsyGuard", version="2.1.0")
app.mount("/static", StaticFiles(directory="static"), name="static")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Shared state ───────────────────────────────────
class AppState:
    def __init__(self):
        self.session_start = datetime.now()
        self.fps           = 0
        self.log           = []
        self.prev_status   = None

    def reset(self):
        self.session_start = datetime.now()
        self.fps           = 0
        self.log           = []
        self.prev_status   = None

    def add_log(self, msg: str, level: str = "ok"):
        self.log.append({
            "time":  datetime.now().strftime("%H:%M:%S"),
            "msg":   msg,
            "level": level
        })
        if len(self.log) > 60:
            self.log.pop(0)

    def session_time(self) -> str:
        e = (datetime.now() - self.session_start).seconds
        return f"{e//60:02d}:{e%60:02d}"

app_state = AppState()

# ── Alert ──────────────────────────────────────────
alert_playing = False

def play_alert():
    global alert_playing
    alert_playing = True
    winsound.Beep(ALERT_FREQUENCY, ALERT_DURATION)
    alert_playing = False

# ── Routes ─────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def root():
    path = os.path.join(BASE_DIR, "static", "index.html")
    with open(path, encoding="utf-8") as f:
        return f.read()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    global alert_playing

    await websocket.accept()

    # Fresh detector per session (Now using MediaPipe version)
    detector = Detector()
    app_state.reset()
    app_state.add_log("Session started", "ok")

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, 30)

    frame_q  = queue.Queue(maxsize=1)
    result_q = queue.Queue(maxsize=1)

    def worker():
        while True:
            try:
                frm = frame_q.get(timeout=2)
                if frm is None: break
                res = detector.process(frm)
                try: result_q.get_nowait()
                except queue.Empty: pass
                result_q.put(res)
            except queue.Empty: break

    t = threading.Thread(target=worker, daemon=True)
    t.start()

    prev_time   = time.time()
    last_result = None
    prev_status = None

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                await asyncio.sleep(0.01)
                continue

            if not frame_q.full():
                try: frame_q.put_nowait(frame)
                except queue.Full: pass

            try: last_result = result_q.get_nowait()
            except queue.Empty: pass

            if last_result is None:
                await asyncio.sleep(0.01)
                continue

            processed, state = last_result

            # FPS calculation
            now = time.time()
            app_state.fps = int(1 / max(now - prev_time, 0.001))
            prev_time = now

            # Log status changes (Simplified for new MediaPipe logic)
            cur_status = state["status"]
            if cur_status != prev_status:
                if cur_status == DriverStatus.DROWSY:
                    app_state.add_log("⚠ DROWSY ALERT!", "alert")
                elif cur_status == DriverStatus.NO_FACE:
                    app_state.add_log("Face lost", "warn")
                elif cur_status == DriverStatus.ACTIVE:
                    if prev_status == DriverStatus.DROWSY:
                        app_state.add_log("Driver alert again", "ok")
                prev_status = cur_status

            # Sound alert trigger
            if cur_status == DriverStatus.DROWSY and not alert_playing:
                threading.Thread(target=play_alert, daemon=True).start()

            # Build full payload for frontend
            state["fps"]          = int(app_state.fps)
            state["session_time"] = app_state.session_time()
            state["log"]          = app_state.log[-12:]

            # Encode frame for WebSocket transmission
            _, buf = cv2.imencode(".jpg", processed, [cv2.IMWRITE_JPEG_QUALITY, 80])
            frame_b64 = base64.b64encode(buf).decode("utf-8")

            await websocket.send_json({
                "frame": frame_b64,
                "state": state
            })

            await asyncio.sleep(0.02)

    except WebSocketDisconnect:
        app_state.add_log("Session ended", "ok")
    except Exception as e:
        app_state.add_log(f"Error: {e}", "alert")
    finally:
        frame_q.put(None)
        cap.release()