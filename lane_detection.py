import cv2
import numpy as np
from collections import deque


def filter_region(image, vertices):
    """
    Create the mask using the vertices and apply it to the input image
    """
    mask = np.zeros_like(image)
    if len(mask.shape) == 2:
        cv2.fillPoly(mask, vertices, 255)
    else:
        cv2.fillPoly(mask, vertices, (255, ) * mask.shape[2])
    return cv2.bitwise_and(image, mask)


def select_region(image):
    """
    It keeps the region surrounded by the `vertices` (i.e. polygon). Other area is set to 0 (black).
    """
    # first, define the polygon by vertices
    rows, cols = image.shape[:2]
    bottom_left = [cols * 0.35, rows * 0.95]
    top_left = [cols * 0.45, rows * 0.55]
    bottom_right = [cols * 0.65, rows * 0.95]
    top_right = [cols * 0.55, rows * 0.55]
    # the vertices are an array of polygons (i.e array of arrays) and the data type must be integer
    vertices = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)
    return filter_region(image, vertices)


def average_slope_intercept(lines):
    left_lines    = [] # (slope, intercept)
    left_weights  = [] # (length,)
    right_lines   = [] # (slope, intercept)
    right_weights = [] # (length,)

    for line in lines:
        for x1, y1, x2, y2 in line:
            if x2 == x1:
                continue # ignore a vertical line
            slope = float(y2 - y1) / (x2 - x1)
            intercept = y1 - slope*x1
            length = np.sqrt((y2-y1)**2+(x2-x1)**2)
            if slope < 0: # y is reversed in image
                left_lines.append((slope, intercept))
                left_weights.append((length))
            else:
                right_lines.append((slope, intercept))
                right_weights.append((length))

    # add more weight to longer lines
    left_lane  = np.dot(left_weights,  left_lines) / np.sum(left_weights)  if len(left_weights) >0 else None
    right_lane = np.dot(right_weights, right_lines)/ np.sum(right_weights) if len(right_weights)>0 else None

    return left_lane, right_lane # (slope, intercept), (slope, intercept)


def make_line_points(y1, y2, line):
    """
    Convert a line represented in slope and intercept into pixel points
    """
    if line is None:
        return None

    slope, intercept = line

    # make sure everything is integer as cv2.line requires it
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    y1 = int(y1)
    y2 = int(y2)

    return ((x1, y1), (x2, y2))


def lane_lines(image, lines):
    left_lane, right_lane = average_slope_intercept(lines)

    y1 = image.shape[0] # bottom of the image
    y2 = y1*0.6         # slightly lower than the middle

    left_line  = make_line_points(y1, y2, left_lane)
    right_line = make_line_points(y1, y2, right_lane)

    return left_line, right_line


def draw_lane_lines(image, lines, color=[255, 0, 0], thickness=20):
    # make a separate image to draw lines and combine with the original later
    line_image = np.zeros_like(image)
    for line in lines:
        if line is not None:
            cv2.line(line_image, line[0], line[1], color, thickness)
    # image1 and image2 must be the same shape.
    return cv2.addWeighted(image, 1.0, line_image, 0.95, 0.0)


def mean_line(line, lines):
    if line is not None:
        lines.append(line)

    if len(lines) > 0:
        line = np.mean(lines, axis=0, dtype=np.int32)
        line = tuple(map(tuple, line)) # make sure it's tuples not numpy array for cv2.line to work
    return line

img = cv2.imread('0000000000.png', 0)
cv2.imwrite('selected.png', select_region(img))
blurred = cv2.GaussianBlur(img, (15, 15), 0)
cv2.imwrite('blurred.png', blurred)
edges = cv2.Canny(blurred, 50, 100)
cv2.imwrite('edges.png', edges)

selected_img = select_region(edges)
# cv2.imwrite('selected.png', selected_img)

lines = cv2.HoughLinesP(
    selected_img,
    rho=1,
    theta=np.pi/180,
    threshold=20,
    minLineLength=100,
    maxLineGap=300
)

left_lines = deque(maxlen=50)
right_lines = deque(maxlen=50)
left_line, right_line = lane_lines(img, lines)
left_line = mean_line(left_line, left_lines)
right_line = mean_line(right_line, right_lines)

cv2.imwrite('lines.png', draw_lane_lines(img, (left_line, right_line), thickness=10))
