from collections import deque

import cv2
import numpy as np

MAX_NUM_COEFFICIENTS = 10


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
        #
        # if not TIMES:
        #     print left_lines
        #     print right_lines
        #     cv2.imwrite('left.png', cv2.cvtColor(_draw_lane_lines(frame, left_points, [255, 0, 0]), cv2.COLOR_BGR2RGB))
        #     cv2.imwrite('right.png', cv2.cvtColor(_draw_lane_lines(frame, right_points, [255, 0, 0]), cv2.COLOR_BGR2RGB))
        #     TIMES += 1

        # left_lines = deque(maxlen=50)
        # right_lines = deque(maxlen=50)
        # left_line, right_line = lane_lines(frame, lines)
        # left_line = mean_line(left_line, left_lines)
        # right_line = mean_line(right_line, right_lines)
        # cv2.imwrite('poly.png', cv2.cvtColor(_draw_lane_lines(frame, [left_points, right_points], [255, 0, 0]), cv2.COLOR_BGR2RGB))

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
        rows, cols = frame.shape[:2]
        # Boundaries are measured based on the input video.
        bottom_left = [cols * 0.15, rows]
        top_left = [cols * 0.45, rows * 0.6]
        top_right = [cols * 0.6, rows * 0.6]
        bottom_right = [cols * 0.86, rows]
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

        if len(lines):
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
    
    
    # def average_slope_intercept(lines):
    #     left_lines = []  # (slope, intercept)
    #     left_weights = []  # (length,)
    #     right_lines = []  # (slope, intercept)
    #     right_weights = []  # (length,)
    #
    #     for line in lines:
    #         for x1, y1, x2, y2 in line:
    #             if y2 == y1 or (x2 == x1):
    #                 continue  # ignore a vertical line
    #             slope = (y2 - y1) / (x2 - x1)
    #             intercept = y1 - slope * x1
    #             length = np.sqrt((y2-y1) ** 2 + (x2-x1) ** 2)
    #             if slope < 0:  # y is reversed in frame
    #                 left_lines.append((slope, intercept))
    #                 left_weights.append((length))
    #             else:
    #                 right_lines.append((slope, intercept))
    #                 right_weights.append((length))
    #
    #     # add more weight to longer lines
    #     left_lane = np.dot(left_weights, left_lines) / np.sum(left_weights) if len(left_weights) > 0 else None
    #     right_lane = np.dot(right_weights, right_lines) / np.sum(right_weights) if len(right_weights) > 0 else None
    #
    #     return left_lane, right_lane  # (slope, intercept), (slope, intercept)
    #
    #
    # def make_line_points(y1, y2, line):
    #     """
    #     Convert a line represented in slope and intercept into pixel points
    #     """
    #     if line is None:
    #         return None
    #
    #     slope, intercept = line
    #
    #     if not slope:
    #         return None
    #
    #     # make sure everything is integer as cv2.line requires it
    #     x1 = int((y1 - intercept) / slope)
    #     x2 = int((y2 - intercept) / slope)
    #     y1 = int(y1)
    #     y2 = int(y2)
    #
    #     return (x1, y1), (x2, y2)
    #
    #
    # def lane_lines(frame, lines):
    #     left_lane, right_lane = average_slope_intercept(lines)
    #
    #     y1 = frame.shape[0]  # bottom of the frame
    #     y2 = y1 * 0.6
    #
    #     left_line = make_line_points(y1, y2, left_lane)
    #     right_line = make_line_points(y1, y2, right_lane)
    #
    #     return left_line, right_line

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
    
    def draw_straight_lines(self, frame, lines, color, thickness=5):
        line_frame = np.zeros_like(frame)
        for line in lines[0]:
            if line is not None:
                cv2.line(line_frame, (line[0], line[1]), (line[2], line[3]), color, thickness)
        return cv2.addWeighted(frame, 1.0, line_frame, 0.95, 0.0)
    
    
    # def mean_line(line, lines):
    #     if line is not None:
    #         lines.append(line)
    #
    #     if len(lines) > 0:
    #         line = np.mean(lines, axis=0, dtype=np.int32)
    #         line = tuple(map(tuple, line))  # make sure it's tuples not numpy array for cv2.line to work
    #     return line


