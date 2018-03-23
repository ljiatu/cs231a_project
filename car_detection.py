import cv2
import numpy as np

from constants import *
from hog_utils import compute_hog_features

WINDOW_SIZES = [
    (128, 128),
    (96, 96),
    (80, 80)
]


class CarDetector(object):
    def __init__(self, svm_clf, cnn_clf, scaler):
        self.svm_clf = svm_clf
        self.cnn_clf = cnn_clf
        self.scaler = scaler

    def process_frame(self, frame):
        # Gets a flattened list of all windows.
        windows = []
        for size in WINDOW_SIZES:
            windows.extend(self._slide_window(frame, size, overlap=0.5))

        bboxes, confidences = self._get_bboxes_svm(frame, windows)
        bboxes, confidences = self._get_bboxes_cnn(frame, bboxes)
        with open('bboxes_cnn.text', 'a+') as f:
            f.write('%s\n' % str(bboxes))
        nms_bboxes = self._non_max_suppression(bboxes, confidences)
        frame_with_bboxes = self._draw_boxes(frame, nms_bboxes, color=(0, 0, 255))

        return frame_with_bboxes

    def _slide_window(self, frame, size, overlap):
        """
        Returns a list of sliding windows.

        :param frame: a video frame
        :param size: windows size
        :param overlap: amount of overlap between windows
        :returns: a list of bounding boxes.
        """
        rows, cols = frame.shape[:2]
        # Boundaries are measured based on the input video.
        xmin = 0
        xmax = cols
        ymin = rows / 2
        ymax = rows

        stride_x = int(size[1] * overlap)
        stride_y = int(size[0] * overlap)

        return [
            ((x, y), (x + size[1], y + size[0]))
            for y in xrange(ymin, ymax, stride_y) for x in xrange(xmin, xmax, stride_x)
        ]

    def _get_bboxes_svm(self, frame, windows):
        """
        Returns bounding boxes that contain cars by applying the SVM classifier.

        :param frame: a video frame
        :param windows: a list of windows that we apply the classifier to
        :returns: a list of bounding boxes, as well as the confidence of each bounding box.
        """
        bboxes = []
        confidences = []

        for window in windows:
            bounded_area = cv2.resize(
                frame[window[0][1]:window[1][1], window[0][0]:window[1][0]],
                (64, 64),
                interpolation=cv2.INTER_AREA
            )

            features = compute_hog_features(bounded_area, ORIENT, PIXELS_PER_CELL, CELLS_PER_BLOCK)
            test_features = self.scaler.transform(np.array(features).reshape(1, -1))
            confidence = self.svm_clf.decision_function(test_features)
            if confidence >= 1:
                bboxes.append(window)
                confidences.append(confidence)

        return bboxes, confidences

    def _get_bboxes_cnn(self, frame, windows):
        """
        Returns bounding boxes that contain cars by applying the CNN classifier.

        :param frame: a video frame
        :param windows: a list of windows that we apply the classifier to
        :returns: a list of bounding boxes, as well as the confidence of each bounding box.
        """
        bboxes = []
        confidences = []

        for window in windows:
            bounded_area = cv2.resize(
                frame[window[0][1]:window[1][1], window[0][0]:window[1][0]],
                (64, 64),
                interpolation=cv2.INTER_AREA
            )
            # Normalize the image.
            bounded_area = bounded_area / np.linalg.norm(bounded_area)
            # Augment at axis 0 to make it four-dimensional.
            bounded_area = np.expand_dims(bounded_area, axis=0)

            _, vehicle = self.cnn_clf.predict(bounded_area)[0]
            if vehicle >= 0.99:
                bboxes.append(window)
                confidences.append(vehicle)

        return bboxes, confidences

    def _non_max_suppression(self, bboxes, confidences):
        def _overlaps(nms_bboxes, bbox):
            """
            Tests whether the given bbox overlaps with any already in the list.
            """
            center_x = (bbox[0][0] + bbox[1][0]) / 2.0
            center_y = (bbox[0][1] + bbox[1][1]) / 2.0
            for nms_bbox in nms_bboxes:
                xmin = nms_bbox[0][0]
                xmax = nms_bbox[1][0]
                ymin = nms_bbox[0][1]
                ymax = nms_bbox[1][1]
                if (xmin <= center_x <= xmax) and (ymin <= center_y <= ymax):
                    return True
            return False

        sorted_bboxes = [bbox for _, bbox in sorted(zip(confidences, bboxes), reverse=True)]
        nms_bboxes = []

        for bbox in sorted_bboxes:
            if not _overlaps(nms_bboxes, bbox):
                nms_bboxes.append(bbox)

        return nms_bboxes

    def _draw_boxes(self, frame, bboxes, color, thickness=5):
        """
        Draws windows or bounding boxes on the image

        :param frame: a video frame
        :param bboxes: bounding boxes
        :param color: bounding box color
        :param thickness: thickness of the bounding box
        :returns: the frame with bounding boxes drawn on it
        """
        for bbox in bboxes:
            cv2.rectangle(frame, bbox[0], bbox[1], color, thickness)
        return frame
