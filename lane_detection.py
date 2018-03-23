from collections import deque

import cv2
import numpy as np

from constants import *


class LaneDetector(object):
    """
    Detects the boundaries of the current lane on a given frame by fitting a parabola to each lane line.

    This class memoizes coefficients of polynomials fitted on the most recent 10 frames to avoid significant deviation
    of coefficients on the most recent frame.
    """
    def __init__(self):
        self.left_coefficients = deque(maxlen=MAX_NUM_COEFFICIENTS)
        self.right_coefficients = deque(maxlen=MAX_NUM_COEFFICIENTS)

    def process_frame(self, frame):
        """
        Processes the given frame by marking lane boundaries on it.

        :param frame: a video frame
        :return: a frame with lane boundaries marked.
        """
        hls = self._filter_colors(frame)
        grayscale = cv2.cvtColor(hls, cv2.COLOR_RGB2GRAY)
        blurred = cv2.GaussianBlur(grayscale, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        selected = self._select_region(edges)

        lines = cv2.HoughLinesP(
            selected,
            rho=1,
            theta=np.pi/180,
            threshold=20,
            minLineLength=15,
            maxLineGap=30
        )

        left_lines, right_lines = self._group_lines(lines[0])

        left_coefficients = self._get_coefficients(left_lines, True)
        right_coefficients = self._get_coefficients(right_lines, False)

        left_points = self._get_fitting_points(frame, left_coefficients)
        right_points = self._get_fitting_points(frame, right_coefficients)

        return self._draw_lane_lines(frame, [left_points, right_points], [255, 0, 0])

    def _filter_colors(self, frame):
        """
        Filters the frame to preserve only white and yellow components.

        Filter both colors in the HSL channel. This is better than the result in the RGB channel.
        """
        hls_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2HLS)
    
        lower_white = np.array([0, 200, 0], dtype=np.int32)
        upper_white = np.array([180, 255, 255], dtype=np.int32)
        white_mask = cv2.inRange(hls_frame, lower_white, upper_white)
    
        lower_yellow = np.array([15, 38, 115], dtype=np.int32)
        upper_yellow = np.array([35, 204, 255], dtype=np.int32)
        yellow_mask = cv2.inRange(hls_frame, lower_yellow, upper_yellow)
    
        return cv2.bitwise_and(frame, frame, mask=cv2.bitwise_or(white_mask, yellow_mask))

    def _select_region(self, frame):
        """
        Carves out the region where lane exists.
        """
        # Boundaries are measured based on the input video.
        bottom_left = (200, 680)
        top_left = (600, 450)
        top_right = (750, 450)
        bottom_right = (1100, 650)
        vertices = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)
    
        mask = np.zeros_like(frame)
        cv2.fillPoly(mask, vertices, 255)
        return cv2.bitwise_and(frame, mask)

    def _calculate_slope(self, line):
        x1, y1, x2, y2 = line
        if x1 == x2:
            return None
        else:
            return float(y2 - y1) / (x2 - x1)

    def _group_lines(self, lines):
        """
        Group lines to left or right lanes.

        Lines with negative slopes belong to the left lane, where lines with positive slopes belong to the right lane.

        :param lines: a list of lines, with each represented by its two endpoints
        :return: lines belonging to the left and right lane.
        """
        left_lines = np.array([line for line in lines if self._calculate_slope(line) < 0])
        right_lines = np.array([line for line in lines if self._calculate_slope(line) > 0])
        return left_lines, right_lines

    def _get_coefficients(self, lines, left=True):
        """
        Returns the coefficients of the fitting parabola.

        First fits a parabola to the given line segments. If the resulting coefficients do not deviate too much from
        the mean coefficient from previous frames, memoizes the new coefficients and returns them. Otherwise, returns
        the mean coefficients of previous frames.

        :param lines: an array of lines to fit
        :param left: whether the parabola is fitting the left lane
        :return: the final coefficients of the fitting parabola.
        """
        past_coefficients = self.left_coefficients if left else self.right_coefficients
        mean_coefficients = np.mean(past_coefficients, axis=0)

        if len(lines) == 0:
            return mean_coefficients

        coefficients = self._fit_polynomial(lines)

        if len(past_coefficients) < 2:
            past_coefficients.append(coefficients)
            return coefficients

        stds = np.std(past_coefficients, axis=0)
        if np.all([
            abs(coefficient - mean) <= 3 * std
            for coefficient, mean, std in zip(coefficients, mean_coefficients, stds)
        ]):
            past_coefficients.append(coefficients)
            return coefficients
        else:
            return mean_coefficients

    def _get_fitting_points(self, frame, coefficients):
        """
        Returns points on the fitting parabola.

        The returned points can be used to draw the parabola on the frame.

        :param frame: a video frame
        :param coefficients: coefficients of the fitting parabola
        :return: an array of points on the fitting parabola.
        """
        rows = frame.shape[0]
        poly = np.poly1d(coefficients)
        return np.array([(poly(y), y) for y in np.linspace(rows, 0.65 * rows, num=100)], dtype=np.int32)

    def _draw_lane_lines(self, frame, points, color, thickness=5):
        """
        Draw lane lines one the given frame.

        :param frame: a video frame
        :param points: points on the fitting parabolas
        :param color: color of the lane lines
        :param thickness: thickness of the lane lines
        :return: the video frame with lane lines drawn on it
        """
        line_frame = np.zeros_like(frame)
        cv2.polylines(frame, points, isClosed=False, color=color, thickness=thickness)
        return cv2.addWeighted(frame, 1.0, line_frame, 0.95, 0.0)

    def _fit_polynomial(self, lines):
        """
        Fits a parabola using the endpoints of given line segments.

        :param lines: a list of lines, with each represented by its two endpoints
        :return: a list of coefficients of the parabola.
        """
        x_coords = np.append(lines[:, 0], lines[:, 2])
        y_coords = np.append(lines[:, 1], lines[:, 3])
        # Given rows (y-coordinates), we want to get the corresponding columns (x-coordinates).
        return np.polyfit(y_coords, x_coords, 2)
