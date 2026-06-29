"""Locates candidate avocado/leaf regions in a video frame.

Phase-2 placeholder for a real object detector. Uses classic OpenCV
color thresholding + contours (same family of technique as a Haar
cascade pipeline: find regions first, classify each region after).
Swapping this module for a trained YOLO model later requires no changes
to the classifier or the routes that call it — `detect_regions` always
returns the same (x, y, w, h) box format.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np

import config


def detect_regions(frame_bgr: np.ndarray):
    """Returns a list of (x, y, w, h) boxes for candidate fruit/leaf regions."""
    frame_h, frame_w = frame_bgr.shape[:2]
    frame_area = frame_h * frame_w

    blurred = cv2.GaussianBlur(frame_bgr, (5, 5), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(
        hsv,
        np.array(config.DETECTION_HSV_LOWER, dtype=np.uint8),
        np.array(config.DETECTION_HSV_UPPER, dtype=np.uint8),
    )
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < config.DETECTION_MIN_AREA:
            continue
        if area > frame_area * config.DETECTION_MAX_AREA_RATIO:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        boxes.append((x, y, w, h))

    boxes.sort(key=lambda b: b[2] * b[3], reverse=True)
    return boxes[: config.DETECTION_MAX_REGIONS]
