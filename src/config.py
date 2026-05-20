import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# ── Paths ──────────────────────────────────────────
MODEL_PATH      = os.path.join(BASE_DIR, 'Models', 'eye_model.keras')
FACE_CASCADE    = os.path.join(BASE_DIR, 'haarcascade', 'haarcascade_frontalface_default.xml')
EYE_CASCADE     = os.path.join(BASE_DIR, 'haarcascade', 'haarcascade_eye.xml')
SMILE_CASCADE   = os.path.join(BASE_DIR, 'haarcascade', 'haarcascade_smile.xml')

# ── Camera ─────────────────────────────────────────
CAMERA_INDEX    = 0
FRAME_WIDTH     = 640
FRAME_HEIGHT    = 480

# ── Image ──────────────────────────────────────────
IMAGE_SIZE      = (24, 24)
COLOR_MODE      = 'grayscale'

# ── Detection thresholds ───────────────────────────
PREDICTION_THRESHOLD    = 0.5
CLOSED_FRAMES_THRESHOLD = 20
YAWN_FRAMES_THRESHOLD   = 15
NO_EYE_BUFFER           = 4     # frames before counting closed

# ── Alert ──────────────────────────────────────────
ALERT_FREQUENCY = 2500        # Hz
ALERT_DURATION = 1000         # ms

# ── Training ───────────────────────────────────────
DATASET_PATH        = os.path.join(BASE_DIR, 'mrleyedataset')
BATCH_SIZE          = 64
SEED                = 123
VALIDATION_SPLIT    = 0.2
TEST_SPLIT          = 0.1
EPOCHS              = 15
DROPOUT_RATE        = 0.5

EYE_AR_THRESH=0.2
MOUTH_AR_THRESH=0.5
