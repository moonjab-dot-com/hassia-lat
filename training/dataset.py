"""Loads the Hass avocado train/valid/test splits as tf.data.Dataset objects."""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tensorflow as tf
import config


def _load_split(directory, shuffle):
    return tf.keras.utils.image_dataset_from_directory(
        directory,
        image_size=config.IMG_SIZE,
        batch_size=config.BATCH_SIZE,
        shuffle=shuffle,
        label_mode="categorical",
    )


def load_datasets():
    """Returns (train_ds, valid_ds, test_ds, class_names)."""
    train_ds = _load_split(config.TRAIN_DIR, shuffle=True)
    class_names = train_ds.class_names
    valid_ds = _load_split(config.VALID_DIR, shuffle=False)
    test_ds = _load_split(config.TEST_DIR, shuffle=False)

    augment = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
        tf.keras.layers.RandomContrast(0.1),
    ])
    train_ds = train_ds.map(lambda x, y: (augment(x, training=True), y))

    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    valid_ds = valid_ds.prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.prefetch(tf.data.AUTOTUNE)
    return train_ds, valid_ds, test_ds, class_names
