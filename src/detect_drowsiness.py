import cv2
import sys
import os
import threading
import winsound

sys.path.append(os.path.dirname(__file__))
from config import *
from detector import Detector, DriverStatus

detector      = Detector()
alert_playing = False

def play_alert():
    global alert_playing
    alert_playing = True
    # Standard warning beep
    winsound.Beep(ALERT_FREQUENCY, ALERT_DURATION)
    alert_playing = False

cap = cv2.VideoCapture(CAMERA_INDEX)
# Set high resolution for better landmark detection
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
cap.set(cv2.CAP_PROP_FPS, 30)

print("System Active. Press ESC to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # The new detector returns the processed frame and a state dictionary
    processed, state = detector.process(frame)

    # Trigger alert if the status is DROWSY
    if state["status"] == DriverStatus.DROWSY and not alert_playing:
        # Use a thread so the video doesn't freeze while the beep plays
        threading.Thread(target=play_alert, daemon=True).start()

    cv2.imshow("DrowsyGuard v2.0", processed)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()