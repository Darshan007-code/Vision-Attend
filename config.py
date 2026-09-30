"""
VisionAttend - Automated Face Recognition Attendance Tracking System
Configuration Module
"""

import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
DATABASE_PATH = BASE_DIR / "attendance.db"

# Ensure runtime directories exist
DATASET_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Model Artifacts
TRAINER_FILE = MODELS_DIR / "trainer.yml"
LABELS_FILE = MODELS_DIR / "labels.json"

# Face Detection & Recognition Parameters
CASCADE_FILENAME = "haarcascade_frontalface_default.xml"
EYE_CASCADE_FILENAME = "haarcascade_eye.xml"

DEFAULT_CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FACE_IMG_SIZE = (200, 200)

# LBPH Classifier Hyperparameters
LBPH_RADIUS = 1
LBPH_NEIGHBORS = 8
LBPH_GRID_X = 8
LBPH_GRID_Y = 8
# LBPH distance threshold: below 75 is typically a good match.
# In OpenCV LBPH, lower distance = better match.
CONFIDENCE_DISTANCE_THRESHOLD = 75.0

# Attendance Logic Defaults
COOLDOWN_MINUTES = 60  # Prevent multiple marks for same student within this window
CLASS_START_TIME = "09:00"  # 24-hour HH:MM format
LATE_GRACE_MINUTES = 15     # Marks after 09:15 marked as 'Late'
ATTENDANCE_MIN_PERCENTAGE = 75.0  # University standard defaulter threshold

# Anti-Spoofing / Liveness Settings
LIVENESS_ENABLED = True
BLUR_THRESHOLD = 80.0  # Laplacian variance threshold to reject blurry/screen artifacts
EYE_CHECK_ENABLED = True

# Voice Feedback
VOICE_ENABLED = True

# Web Server Settings
WEB_HOST = os.environ.get("HOST", "0.0.0.0")
WEB_PORT = int(os.environ.get("PORT", 5000))
SECRET_KEY = os.environ.get("SECRET_KEY", "visionattend-enterprise-secret-key-2026")
