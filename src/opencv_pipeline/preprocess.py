import cv2
import numpy as np

def load_video(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {path}")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        yield frame
    cap.release()