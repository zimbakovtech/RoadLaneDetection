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


def gaussian_blur(img, kernel_size=5):
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def color_threshold(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # White mask
    lower_white = np.array([0, 0, 200], dtype=np.uint8)
    upper_white = np.array([255, 30, 255], dtype=np.uint8)
    white_mask = cv2.inRange(hsv, lower_white, upper_white)
    # Yellow mask
    lower_yellow = np.array([15, 100, 100], dtype=np.uint8)
    upper_yellow = np.array([35, 255, 255], dtype=np.uint8)
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    # Combine masks
    combined = cv2.bitwise_or(white_mask, yellow_mask)
    return combined


def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    if len(img.shape) > 2:
        channel_count = img.shape[2]
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    return cv2.bitwise_and(img, mask)