import cv2
import numpy as np

# Example source and destination points; adjust per camera calibration
SRC = np.float32([
    [580, 460],
    [700, 460],
    [1040, 680],
    [260, 680]
])
DST = np.float32([
    [260, 0],
    [1040, 0],
    [1040, 720],
    [260, 720]
])


def get_transform_matrices(src=SRC, dst=DST):
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    return M, Minv


def warp_perspective(img, M, size):
    return cv2.warpPerspective(img, M, size, flags=cv2.INTER_LINEAR)


def unwarp_perspective(img, Minv, size):
    return cv2.warpPerspective(img, Minv, size, flags=cv2.INTER_LINEAR)