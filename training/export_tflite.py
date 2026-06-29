"""Converts the trained Keras model to TensorFlow Lite for lightweight serving.

The Flask app loads the .tflite file at inference time instead of the full
.keras model — far less memory, which matters on small deploy targets
(e.g. Render's free tier, 512MB RAM) where loading full TensorFlow plus a
Keras model can exceed the limit and get OOM-killed.

Usage:
    python training/export_tflite.py
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tensorflow as tf
import config


def main():
    model = tf.keras.models.load_model(config.MODEL_PATH)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    with open(config.TFLITE_MODEL_PATH, "wb") as f:
        f.write(tflite_model)

    print(f"Saved TFLite model to {config.TFLITE_MODEL_PATH}")
    print(f"Size: {os.path.getsize(config.TFLITE_MODEL_PATH) / (1024 * 1024):.1f} MB")


if __name__ == "__main__":
    main()
