"""Image classifier wrapper used by the Flask routes.

Kept deliberately decoupled from Flask and from file I/O: `classify_image`
takes a PIL Image in memory and returns a prediction. This lets a future
detection step (e.g. cropping leaf/fruit regions out of a video frame) call
the same function per crop without any changes here.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import threading

import numpy as np
import tensorflow as tf
from PIL import Image

import config

_model = None
_class_names = None
_lock = threading.Lock()


def _load():
    global _model, _class_names
    with _lock:
        if _model is None:
            _model = tf.keras.models.load_model(config.MODEL_PATH)
            with open(config.CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
                _class_names = json.load(f)
    return _model, _class_names


def classify_image(image: Image.Image):
    """Runs the classifier on a single PIL image.

    Returns a dict: {"label": str, "confidence": float, "probabilities": {label: float}}
    """
    model, class_names = _load()

    image = image.convert("RGB").resize(config.IMG_SIZE)
    array = tf.keras.utils.img_to_array(image)
    batch = np.expand_dims(array, axis=0)

    preds = model.predict(batch, verbose=0)[0]
    probabilities = {class_names[i]: float(preds[i]) for i in range(len(class_names))}
    best_idx = int(np.argmax(preds))

    return {
        "label": class_names[best_idx],
        "confidence": float(preds[best_idx]),
        "probabilities": probabilities,
    }
