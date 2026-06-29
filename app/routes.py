import os
import sys
import uuid

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import cv2
from flask import Blueprint, current_app, jsonify, render_template, request, url_for
from PIL import Image
from werkzeug.utils import secure_filename

import config
from app.inference import classify_image
from app.detector import detect_regions
from app.recommendations import get_display_name, get_recommendation

bp = Blueprint("main", __name__)


def _allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in config.ALLOWED_EXTENSIONS


@bp.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@bp.route("/predict", methods=["POST"])
def predict():
    file = request.files.get("photo")

    if file is None or file.filename == "":
        return render_template("index.html", error="Por favor seleccione una imagen.")

    if not _allowed_file(file.filename):
        return render_template(
            "index.html", error="Formato no soportado. Use JPG o PNG."
        )

    filename = f"{uuid.uuid4().hex}_{secure_filename(file.filename)}"
    save_path = os.path.join(current_app.config["UPLOAD_FOLDER"], filename)
    file.save(save_path)

    image = Image.open(save_path)
    result = classify_image(image)

    display_name = get_display_name(result["label"])
    recommendation = get_recommendation(result["label"])
    confidence_pct = round(result["confidence"] * 100, 1)

    probabilities = [
        {"display_name": get_display_name(label), "value": round(prob * 100, 1)}
        for label, prob in sorted(result["probabilities"].items(), key=lambda kv: -kv[1])
    ]

    return render_template(
        "index.html",
        image_url=url_for("static", filename=f"uploads/{filename}"),
        predicted_class=display_name,
        confidence=confidence_pct,
        recommendation=recommendation,
        probabilities=probabilities,
    )


@bp.route("/video", methods=["GET"])
def video():
    return render_template("live.html")


@bp.route("/detect", methods=["POST"])
def detect():
    """Receives one webcam frame, finds candidate regions, classifies each one.

    Request: multipart form with a "frame" image file.
    Response: {"detections": [{x, y, w, h, label, confidence}, ...]}
    """
    file = request.files.get("frame")
    if file is None:
        return jsonify({"detections": []})

    file_bytes = np.frombuffer(file.read(), dtype=np.uint8)
    frame_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if frame_bgr is None:
        return jsonify({"detections": []})

    boxes = detect_regions(frame_bgr)

    detections = []
    for (x, y, w, h) in boxes:
        crop_bgr = frame_bgr[y : y + h, x : x + w]
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        crop_image = Image.fromarray(crop_rgb)

        result = classify_image(crop_image)
        detections.append({
            "x": int(x),
            "y": int(y),
            "w": int(w),
            "h": int(h),
            "label": get_display_name(result["label"]),
            "is_healthy": result["label"] == "Healthy",
            "confidence": round(result["confidence"] * 100, 1),
        })

    return jsonify({"detections": detections})
