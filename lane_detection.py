from collections import deque
import cv2
import numpy as np
import matplotlib.pyplot as plt


def filter_colors(frame):
    """
    Filters the frame to preserve only white and yellow components.
    """
    hls_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2HLS)

    # Filter white colors within RGB channel. The result is good enough.
    lower_white = np.array([150, 150, 150], dtype=np.int32)
    upper_white = np.array([255, 255, 255], dtype=np.int32)
    white_mask = cv2.inRange(frame, lower_white, upper_white)

    # Filter yellow colors in HSL channel. This is better than the result in the HLS channel.
    lower_yellow = np.array([20, 120, 80], dtype=np.int32)
    upper_yellow = np.array([45, 200, 255], dtype=np.int32)
    yellow_mask = cv2.inRange(hls_frame, lower_yellow, upper_yellow)

    return cv2.bitwise_and(frame, frame, mask=cv2.bitwise_or(white_mask, yellow_mask))


def select_region(frame):
    """
    Carves out the region where lane exists.
    """
    rows, cols = frame.shape[:2]
    # Boundaries are measured based on the input video.
    bottom_left = [cols * 0.25, rows]
    top_left = [cols * 0.25, rows * 0.5]
    top_right = [cols * 0.5, rows * 0.5]
    bottom_right = [cols * 0.95, rows]
    vertices = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)

    mask = np.zeros_like(frame)
    cv2.fillPoly(mask, vertices, 255)
    return cv2.bitwise_and(frame, mask)


def average_slope_intercept(lines):
    left_lines = []  # (slope, intercept)
    left_weights = []  # (length,)
    right_lines = []  # (slope, intercept)
    right_weights = []  # (length,)

    for line in lines:
        for x1, y1, x2, y2 in line:
            if y2 == y1 or (x2 == x1):
                continue  # ignore a vertical line
            slope = (y2 - y1) / (x2 - x1)
            intercept = y1 - slope * x1
            length = np.sqrt((y2-y1) ** 2 + (x2-x1) ** 2)
            if slope < 0:  # y is reversed in frame
                left_lines.append((slope, intercept))
                left_weights.append((length))
            else:
                right_lines.append((slope, intercept))
                right_weights.append((length))

    # add more weight to longer lines
    left_lane = np.dot(left_weights, left_lines) / np.sum(left_weights) if len(left_weights) > 0 else None
    right_lane = np.dot(right_weights, right_lines) / np.sum(right_weights) if len(right_weights) > 0 else None

    return left_lane, right_lane  # (slope, intercept), (slope, intercept)


def make_line_points(y1, y2, line):
    """
    Convert a line represented in slope and intercept into pixel points
    """
    if line is None:
        return None

    slope, intercept = line

    if not slope:
        return None

    # make sure everything is integer as cv2.line requires it
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    y1 = int(y1)
    y2 = int(y2)

    return (x1, y1), (x2, y2)


def lane_lines(frame, lines):
    left_lane, right_lane = average_slope_intercept(lines)

    y1 = frame.shape[0]  # bottom of the frame
    y2 = y1 * 0.6

    left_line = make_line_points(y1, y2, left_lane)
    right_line = make_line_points(y1, y2, right_lane)

    return left_line, right_line


def draw_lane_lines(frame, lines, color, thickness=20):
    # make a separate frame to draw lines and combine with the original later
    line_frame = np.zeros_like(frame)
    for line in lines:
        if line is not None:
            cv2.line(line_frame, line[0], line[1], color, thickness)
    # frame1 and frame2 must be the same shape.
    return cv2.addWeighted(frame, 1.0, line_frame, 0.95, 0.0)


def mean_line(line, lines):
    if line is not None:
        lines.append(line)

    if len(lines) > 0:
        line = np.mean(lines, axis=0, dtype=np.int32)
        line = tuple(map(tuple, line))  # make sure it's tuples not numpy array for cv2.line to work
    return line


def process_frame(frame):
    hls = filter_colors(frame)
    # plt.imshow(hls)
    # plt.show()
    grayscale = cv2.cvtColor(hls, cv2.COLOR_RGB2GRAY)
    # plt.imshow(grayscale)
    # plt.show()
    blurred = cv2.GaussianBlur(grayscale, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 200)
    # plt.imshow(edges)
    # plt.show()
    selected = select_region(edges)
    # plt.imshow(cv2.COLOR_GRAY2RGB(selected))
    # plt.show()

    lines = cv2.HoughLinesP(
        selected,
        rho=2,
        theta=np.pi / 180,
        threshold=20,
        minLineLength=25,
        maxLineGap=10
    )

    left_lines = deque(maxlen=50)
    right_lines = deque(maxlen=50)
    left_line, right_line = lane_lines(frame, lines)
    left_line = mean_line(left_line, left_lines)
    right_line = mean_line(right_line, right_lines)

    return draw_lane_lines(frame, (left_line, right_line), [0, 0, 255])

