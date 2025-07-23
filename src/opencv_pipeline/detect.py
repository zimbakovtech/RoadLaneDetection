import cv2
import numpy as np
import math

def region_of_interest(img, vertices):
    # Create a blank mask matching the image dimensions
    mask = np.zeros_like(img)
    # Determine mask color based on image channels
    if len(img.shape) > 2:
        channel_count = img.shape[2]
        match_mask_color = (255,) * channel_count
    else:
        match_mask_color = 255
    # Fill the polygon defined by vertices
    cv2.fillPoly(mask, [vertices], match_mask_color)
    # Return the image only in the masked region
    return cv2.bitwise_and(img, mask)


def draw_lines(img, lines, color=(0, 0, 255), thickness=5):
    # Make a blank image to draw lines on
    line_img = np.zeros_like(img)
    if lines is None:
        return img
    # Draw each line segment
    for x1, y1, x2, y2 in lines:
        cv2.line(line_img, (x1, y1), (x2, y2), color, thickness)
    # Overlay the lines on the original image
    return cv2.addWeighted(img, 0.8, line_img, 1.0, 0.0)


def pipeline(frame):
    """
    Process a video frame or image to detect and overlay lane lines.
    """
    height, width = frame.shape[:2]
    # Define a triangular region of interest
    vertices = np.array([
        (0, height),
        (width // 2, height * 3 // 5),
        (width, height)
    ], dtype=np.int32)

    # 1. Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # 2. Apply Gaussian blur
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # 3. Perform Canny edge detection
    edges = cv2.Canny(blur, 100, 200)
    # 4. Mask edges image to region of interest
    masked = region_of_interest(edges, vertices)
    # 5. Run Hough transform to find line segments
    segments = cv2.HoughLinesP(
        masked,
        rho=6,
        theta=np.pi / 60,
        threshold=160,
        minLineLength=40,
        maxLineGap=25
    )

    if segments is None:
        return frame

    # 6. Separate segments into left and right based on slope
    left_x, left_y, right_x, right_y = [], [], [], []
    for seg in segments:
        for x1, y1, x2, y2 in seg:
            if x2 == x1:
                continue
            slope = (y2 - y1) / (x2 - x1)
            # Filter out nearly horizontal lines
            if abs(slope) < 0.5:
                continue
            if slope < 0:
                left_x.extend([x1, x2])
                left_y.extend([y1, y2])
            else:
                right_x.extend([x1, x2])
                right_y.extend([y1, y2])

    # 7. Fit a single line to each side using polyfit
    y_max = height
    y_min = int(height * 3 / 5)
    lines = []

    if left_x and left_y:
        left_fit = np.poly1d(np.polyfit(left_y, left_x, deg=1))
        x_start = int(left_fit(y_max))
        x_end = int(left_fit(y_min))
        lines.append((x_start, y_max, x_end, y_min))

    if right_x and right_y:
        right_fit = np.poly1d(np.polyfit(right_y, right_x, deg=1))
        x_start = int(right_fit(y_max))
        x_end = int(right_fit(y_min))
        lines.append((x_start, y_max, x_end, y_min))

    # 8. Draw the lane lines back onto the original frame
    output = draw_lines(frame, lines)
    return output


def detect_lane(frame):
    """
    Alias for pipeline, for compatibility.
    """
    return pipeline(frame)
