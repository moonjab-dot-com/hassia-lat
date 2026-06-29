"""Shared configuration for training and inference.

Single source of truth for paths and the class list so the training
script and the Flask app never drift out of sync with each other.
"""
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Dataset (Hass avocado, Disease/Healthy binary labels)
DATASET_DIR = os.path.join(BASE_DIR, "data", "Avocado Augmneted_Dataset")
TRAIN_DIR = os.path.join(DATASET_DIR, "train")
VALID_DIR = os.path.join(DATASET_DIR, "valid")
TEST_DIR = os.path.join(DATASET_DIR, "test")

# Model artifacts
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "hassia_model.keras")
TFLITE_MODEL_PATH = os.path.join(MODEL_DIR, "hassia_model.tflite")
CLASS_NAMES_PATH = os.path.join(MODEL_DIR, "class_names.json")
CONFUSION_MATRIX_PATH = os.path.join(MODEL_DIR, "confusion_matrix.png")
TRAINING_HISTORY_PATH = os.path.join(MODEL_DIR, "training_history.png")
CLASSIFICATION_REPORT_PATH = os.path.join(MODEL_DIR, "classification_report.txt")

# Model / training hyperparameters
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS_FROZEN = 10      # head-only training with the backbone frozen
EPOCHS_FINE_TUNE = 10   # fine-tuning with the top backbone layers unfrozen
LEARNING_RATE_FROZEN = 1e-3
LEARNING_RATE_FINE_TUNE = 1e-5
BACKBONE = "efficientnetb0"  # "efficientnetb0" or "resnet50"

# Live-video detection (phase 2): color/contour segmentation finds candidate
# avocado/leaf regions per frame before each crop is sent to the classifier.
# These HSV bounds and size thresholds are a starting point for green/brown
# fruit against a roughly uniform background — tune them against real
# camera footage. When phase 2 moves to YOLO, only this module needs to change.
DETECTION_HSV_LOWER = (15, 40, 30)   # yellowish-green to brown lower bound
DETECTION_HSV_UPPER = (95, 255, 255)  # green upper bound
DETECTION_MIN_AREA = 1200    # px^2, discard tiny noise blobs
DETECTION_MAX_AREA_RATIO = 0.6  # discard blobs covering most of the frame (likely background)
DETECTION_MAX_REGIONS = 12   # cap detections per frame for latency

# Flask app
UPLOAD_FOLDER = os.path.join(BASE_DIR, "app", "static", "uploads")
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
MAX_CONTENT_LENGTH = 8 * 1024 * 1024  # 8 MB
SECRET_KEY = os.environ.get("HASSIA_SECRET_KEY", "dev-secret-key-change-in-production")
