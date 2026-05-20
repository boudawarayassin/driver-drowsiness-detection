import cv2
import numpy as np
import mediapipe as mp
from scipy.spatial import distance as dist
from config import *

class DriverStatus:
    ACTIVE   = "ACTIVE"
    DROWSY   = "DROWSY"
    NO_FACE  = "NO FACE"

class Detector:
    def __init__(self):
        # Initialize MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Landmarks for EAR (Eye Aspect Ratio) and MAR (Mouth Aspect Ratio)
        self.LEFT_EYE = [362, 385, 387, 263, 373, 380]
        self.RIGHT_EYE = [33, 160, 158, 133, 153, 144]
        self.MOUTH = [13, 14, 78, 308] # Inner lip landmarks

        # Thresholds (Adjust these in your config.py)
        self.EYE_AR_THRESH = 0.22 
        self.MOUTH_AR_THRESH = 0.60
        
        # State
        self.closed_frames = 0
        self.yawn_frames = 0
        self.status = DriverStatus.ACTIVE
        self.total_alerts = 0

    def _calculate_ear(self, landmarks, eye_indices, w, h):
        # Extract eye coordinates
        pts = []
        for i in eye_indices:
            pt = landmarks[i]
            pts.append((pt.x * w, pt.y * h))
        
        # EAR formula: (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)
        v1 = dist.euclidean(pts[1], pts[5])
        v2 = dist.euclidean(pts[2], pts[4])
        hor = dist.euclidean(pts[0], pts[3])
        return (v1 + v2) / (2.0 * hor)

    def _calculate_mar(self, landmarks, w, h):
        # Vertical distance between inner lips
        p1 = (landmarks[13].x * w, landmarks[13].y * h)
        p2 = (landmarks[14].x * w, landmarks[14].y * h)
        # Horizontal distance of mouth
        p3 = (landmarks[78].x * w, landmarks[78].y * h)
        p4 = (landmarks[308].x * w, landmarks[308].y * h)
        
        ver = dist.euclidean(p1, p2)
        hor = dist.euclidean(p3, p4)
        return ver / hor

    def process(self, frame: np.ndarray) -> tuple[np.ndarray, dict]:
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)

        if not results.multi_face_landmarks:
            self.status = DriverStatus.NO_FACE
            cv2.putText(frame, "NO FACE DETECTED", (20, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return frame, self._state_dict()

        landmarks = results.multi_face_landmarks[0].landmark
        
        # Calculate Ratios
        left_ear = self._calculate_ear(landmarks, self.LEFT_EYE, w, h)
        right_ear = self._calculate_ear(landmarks, self.RIGHT_EYE, w, h)
        ear = (left_ear + right_ear) / 2.0
        mar = self._calculate_mar(landmarks, w, h)

        # ── Eye Logic ────────────
        if ear < self.EYE_AR_THRESH:
            self.closed_frames += 1
        else:
            self.closed_frames = 0

        # ── Yawn Logic ───────────
        if mar > self.MOUTH_AR_THRESH:
            self.yawn_frames += 1
        else:
            self.yawn_frames = 0

        # ── Status Logic ─────────
        drowsy = (self.closed_frames > CLOSED_FRAMES_THRESHOLD or 
                self.yawn_frames > YAWN_FRAMES_THRESHOLD)

        if drowsy:
            if self.status != DriverStatus.DROWSY:
                self.total_alerts += 1
            self.status = DriverStatus.DROWSY
            cv2.putText(frame, "DROWSY ALERT!", (50, 100), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
        else:
            self.status = DriverStatus.ACTIVE

        # Visual Debugging (HUD)
        self._draw_debug(frame, ear, mar)

        return frame, self._state_dict()

    def _draw_debug(self, img, ear, mar):
        cv2.putText(img, f"EAR: {ear:.2f}", (10, 30), 1, 1, (0, 255, 0), 1)
        cv2.putText(img, f"MAR: {mar:.2f}", (10, 60), 1, 1, (0, 255, 0), 1)
        # Draw a bar for drowsiness
        bar_len = int(min(self.closed_frames / CLOSED_FRAMES_THRESHOLD, 1) * 200)
        cv2.rectangle(img, (10, 80), (10 + bar_len, 95), (0, 0, 255), -1)

    def _state_dict(self) -> dict:
        return {
            "status": self.status,
            "closed_frames": self.closed_frames,
            "yawn_frames": self.yawn_frames,
            "total_alerts": self.total_alerts
        }