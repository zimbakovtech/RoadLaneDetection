import numpy as np
import cv2

YM_PER_PIX = 30/720  # meters per pixel in y dimension
XM_PER_PIX = 3.7/700 # meters per pixel in x dimension


def fit_polynomial(binary_warped):
    # Histogram of bottom half
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    midpoint = histogram.shape[0] // 2
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Sliding windows parameters
    nwindows = 9
    window_height = binary_warped.shape[0] // nwindows
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    leftx_current = leftx_base
    rightx_current = rightx_base
    margin = 100
    minpix = 50
    left_lane_inds = []
    right_lane_inds = []

    # Slide windows
    for window in range(nwindows):
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                           (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        if len(good_left_inds) > minpix:
            leftx_current = int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = int(np.mean(nonzerox[good_right_inds]))

    left_lane_inds = np.concatenate(left_lane_inds) if left_lane_inds else np.array([])
    right_lane_inds = np.concatenate(right_lane_inds) if right_lane_inds else np.array([])

    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Handle empty lane pixel case
    if leftx.size == 0 or lefty.size == 0 or rightx.size == 0 or righty.size == 0:
        return None, None, None

    # Fit polynomial
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
    return left_fit, right_fit, ploty


def draw_lane(original_img, left_fit, right_fit, ploty, Minv):
    warp_zero = np.zeros_like(original_img[:,:,0]).astype(np.uint8)
    color_warp = np.zeros_like(original_img).astype(np.uint8)

    # Generate x values
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    # Recast for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw lane onto blank
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    # Warp back to original space
    newwarp = cv2.warpPerspective(color_warp, Minv, (original_img.shape[1], original_img.shape[0]))
    result = cv2.addWeighted(original_img, 1, newwarp, 0.3, 0)
    return result


def measure_curvature(ploty, left_fit, right_fit):
    # Evaluate at the bottom of image
    y_eval = np.max(ploty)
    # Convert to real world
    left_fit_cr = np.array([left_fit[0]*XM_PER_PIX/(YM_PER_PIX**2),
                             left_fit[1]*XM_PER_PIX/YM_PER_PIX,
                             left_fit[2]*XM_PER_PIX])
    # Radius
    curverad = ((1 + (2*left_fit_cr[0]*y_eval*YM_PER_PIX + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    return curverad