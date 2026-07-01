"""Trains the Hass avocado health classifier and reports test-set performance.

Usage:
    python training/train.py
"""
import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.utils.class_weight import compute_class_weight

import config
from training.dataset import load_datasets
from training.model import build_model, compile_for_frozen_training, compile_for_fine_tuning


def compute_weights(class_names):
    """Count files on disk to compute balanced class weights."""
    counts = []
    for cls in class_names:
        p = Path(config.TRAIN_DIR) / cls
        counts.append(len(list(p.glob("*.*"))))
    labels = []
    for i, c in enumerate(counts):
        labels.extend([i] * c)
    labels = np.array(labels)
    weights = compute_class_weight("balanced", classes=np.unique(labels), y=labels)
    cw = dict(enumerate(weights))
    print(f"Class weights: { {class_names[i]: round(w, 3) for i, w in cw.items()} }")
    return cw


def plot_history(history_frozen, history_fine_tune):
    acc = history_frozen.history["accuracy"] + history_fine_tune.history["accuracy"]
    val_acc = history_frozen.history["val_accuracy"] + history_fine_tune.history["val_accuracy"]
    loss = history_frozen.history["loss"] + history_fine_tune.history["loss"]
    val_loss = history_frozen.history["val_loss"] + history_fine_tune.history["val_loss"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(acc, label="train")
    axes[0].plot(val_acc, label="valid")
    axes[0].set_title("Accuracy")
    axes[0].legend()
    axes[1].plot(loss, label="train")
    axes[1].plot(val_loss, label="valid")
    axes[1].set_title("Loss")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(config.TRAINING_HISTORY_PATH)
    plt.close(fig)


def evaluate_on_test(model, test_ds, class_names):
    y_true, y_pred = [], []
    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        y_true.extend(np.argmax(labels.numpy(), axis=1))
        y_pred.extend(np.argmax(preds, axis=1))

    report = classification_report(y_true, y_pred, target_names=class_names)
    print("\n=== Test set classification report ===")
    print(report)
    with open(config.CLASSIFICATION_REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report)

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(5, 5))
    disp.plot(ax=ax, cmap="Blues")
    fig.tight_layout()
    fig.savefig(config.CONFUSION_MATRIX_PATH)
    plt.close(fig)
    print(f"Confusion matrix saved to {config.CONFUSION_MATRIX_PATH}")


def main():
    os.makedirs(config.MODEL_DIR, exist_ok=True)

    train_ds, valid_ds, test_ds, class_names = load_datasets()
    print(f"Classes (folder order): {class_names}")

    with open(config.CLASS_NAMES_PATH, "w", encoding="utf-8") as f:
        json.dump(class_names, f, ensure_ascii=False, indent=2)

    class_weight = compute_weights(class_names)

    model, base = build_model(num_classes=len(class_names))

    compile_for_frozen_training(model)
    print(f"\n=== Backbone: {config.BACKBONE} ===")
    print("=== Phase 1: training classifier head (backbone frozen) ===")
    history_frozen = model.fit(
        train_ds,
        validation_data=valid_ds,
        epochs=config.EPOCHS_FROZEN,
        class_weight=class_weight,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                patience=config.EARLY_STOPPING_PATIENCE, restore_best_weights=True
            ),
        ],
    )

    compile_for_fine_tuning(model, base)
    print("\n=== Phase 2: fine-tuning top backbone layers ===")
    history_fine_tune = model.fit(
        train_ds,
        validation_data=valid_ds,
        epochs=config.EPOCHS_FINE_TUNE,
        class_weight=class_weight,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                patience=config.EARLY_STOPPING_PATIENCE, restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=4, min_lr=1e-7
            ),
            tf.keras.callbacks.ModelCheckpoint(
                config.MODEL_PATH, save_best_only=True, monitor="val_accuracy"
            ),
        ],
    )

    model.save(config.MODEL_PATH)
    print(f"\nModel saved to {config.MODEL_PATH}")

    plot_history(history_frozen, history_fine_tune)
    evaluate_on_test(model, test_ds, class_names)


if __name__ == "__main__":
    main()
