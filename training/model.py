"""Builds a transfer-learning classifier on top of a pretrained ImageNet backbone."""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tensorflow as tf
import config


def _get_backbone():
    if config.BACKBONE == "resnet50":
        base = tf.keras.applications.ResNet50(
            include_top=False, weights="imagenet", input_shape=config.IMG_SIZE + (3,)
        )
        preprocess = tf.keras.applications.resnet50.preprocess_input
    else:
        base = tf.keras.applications.EfficientNetB0(
            include_top=False, weights="imagenet", input_shape=config.IMG_SIZE + (3,)
        )
        preprocess = tf.keras.applications.efficientnet.preprocess_input
    return base, preprocess


def build_model(num_classes):
    base, preprocess = _get_backbone()
    base.trainable = False

    inputs = tf.keras.Input(shape=config.IMG_SIZE + (3,))
    x = preprocess(inputs)
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs)
    return model, base


def compile_for_frozen_training(model):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(config.LEARNING_RATE_FROZEN),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )


def compile_for_fine_tuning(model, base, unfreeze_last_n=None):
    if unfreeze_last_n is None:
        unfreeze_last_n = config.UNFREEZE_LAST_N_LAYERS
    base.trainable = True
    for layer in base.layers[:-unfreeze_last_n]:
        layer.trainable = False
    model.compile(
        optimizer=tf.keras.optimizers.Adam(config.LEARNING_RATE_FINE_TUNE),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
