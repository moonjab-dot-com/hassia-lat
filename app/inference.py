"""Image classifier wrapper used by the Flask routes.

Kept deliberately decoupled from Flask and from file I/O: `classify_image`
takes a PIL Image in memory and returns a prediction. This lets a future
detection step (e.g. cropping leaf/fruit regions out of a video frame) call
the same function per crop without any changes here.

Uses a TFLite interpreter rather than loading the full Keras model — far
less memory at runtime, which matters on small deploy targets (e.g.
Render's free tier, 512MB RAM) where the full TensorFlow + Keras model
load was getting OOM-killed.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import threading

import numpy as np
from PIL import Image

try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    from tensorflow import lite as tflite

import config

_interpreter = None
_input_details = None
_output_details = None
_class_names = None
_lock = threading.Lock()


def _load():
    global _interpreter, _input_details, _output_details, _class_names
    with _lock:
        if _interpreter is None:
            _interpreter = tflite.Interpreter(model_path=config.TFLITE_MODEL_PATH)
            _interpreter.allocate_tensors()
            _input_details = _interpreter.get_input_details()
            _output_details = _interpreter.get_output_details()
            with open(config.CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
                _class_names = json.load(f)
    return _interpreter, _input_details, _output_details, _class_names


def classify_image(image: Image.Image):
    """Runs the classifier on a single PIL image.

    Returns a dict: {"label": str, "confidence": float, "probabilities": {label: float}}
    """
    interpreter, input_details, output_details, class_names = _load()

    image = image.convert("RGB").resize(config.IMG_SIZE)
    array = np.asarray(image, dtype=np.float32)
    batch = np.expand_dims(array, axis=0)

    interpreter.set_tensor(input_details[0]["index"], batch)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_details[0]["index"])[0]

    probabilities = {class_names[i]: float(preds[i]) for i in range(len(class_names))}
    best_idx = int(np.argmax(preds))

    return {
        "label": class_names[best_idx],
        "confidence": float(preds[best_idx]),
        "probabilities": probabilities,
    }
