import math

import cv2
import numpy as np
from keras.preprocessing.image import img_to_array

from constants import *
from hog_utils import compute_hog_features

WINDOW_SIZES = [
    (128, 128),
    (96, 96),
    (80, 80)
]


class CarDetector(object):
    def __init__(self, svc_clf, cnn_clf, scaler):
        self.svc_clf = svc_clf
        self.cnn_clf = cnn_clf
        self.scaler = scaler

    def process_frame(self, frame):
        # Uncomment the following line if you extracted training
        # data from .png images (scaled 0 to 1 by mpimg) and the
        # image you are searching is a .jpg (scaled 0 to 255)
        # image = image_orig.astype(np.float32)/255

        bboxes, confidences = self._get_bboxes_svc(frame, ORIENT, PIXELS_PER_CELL, CELLS_PER_BLOCK, overlap=0.5)

        # bboxes, confidences = get_bboxes_cnn(frame, cnn_clf, bboxes, confidences)

        nms_bboxes = self._non_max_suppression(bboxes, confidences)

        frame_with_bboxes = self._draw_boxes(frame, nms_bboxes, color=(0, 0, 255))

        return frame_with_bboxes

    def _search_windows(self, frame, windows, orient, pixels_per_cell, cells_per_block):
        """
        Returns bounding boxes that contain cars.

        :param frame: a video frame
        :param orient: number of HoG orientations
        :param pixels_per_cell: number of pixels per cell
        :param cells_per_block: number of HoG cells per block
        :returns: a list of bounding boxes.
        """
        bboxes = []
        confidences = []

        for window in windows:
            test_img = cv2.resize(
                frame[window[0][1]:window[1][1], window[0][0]:window[1][0]],
                (64, 64),
                interpolation=cv2.INTER_AREA
            )
            features = compute_hog_features(test_img, orient, pixels_per_cell, cells_per_block)
            test_features = self.scaler.transform(np.array(features).reshape(1, -1))
            confidence = self.svc_clf.decision_function(test_features)
            if confidence >= 1:
                bboxes.append(window)
                confidences.append(confidence)

        return bboxes, confidences

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
        xmin = cols / 2
        xmax = cols
        ymin = 0
        ymax = rows

        sizex = xmax - xmin
        sizey = ymax - ymin

        stepx = int(size[0] * overlap)
        stepy = int(size[1] * overlap)

        step_count_x = int(math.floor(1.0 * sizex / stepx)) - 1
        step_count_y = int(math.floor(1.0 * sizey / stepy)) - 1

        return [
            ((xmin + j * stepx, ymin + i * stepy), (xmin + j * stepx + size[0], ymin + i * stepy + size[1]))
            for i in xrange(step_count_y) for j in xrange(step_count_x)
        ]

    def _get_bboxes_svc(self, frame, orient, pixels_per_cell, cells_per_block, overlap):
        """
        Returns bounding boxes that contain cars by applying the SVM classifier.

        :param frame: a video frame
        :param orient: number of HoG orientations
        :param pixels_per_cell: number of pixels per cell
        :param cells_per_block: number of HoG cells per block
        :param overlap: overlap between windows
        :returns: a list of bounding boxes, as well as the confidence of each bounding box.
        """
        all_bboxes = []
        all_confidences = []

        for size in WINDOW_SIZES:
            windows = self._slide_window(frame, size=size, overlap=overlap)
            # print windows
            bboxes, confidences = self._search_windows(
                frame,
                windows,
                orient=orient,
                pixels_per_cell=pixels_per_cell,
                cells_per_block=cells_per_block,
            )
            all_bboxes.extend(bboxes)
            all_confidences.extend(confidences)

        return all_bboxes, all_confidences

    def _get_bboxes_cnn(self, frame, bboxes, confidences):
        new_bboxes = []
        new_confidences = []
        for index, bbox in enumerate(bboxes):
            # Pre-process the image for classification.
            image_temp = frame[bbox[0][1]:bbox[1][1], bbox[0][0]:bbox[1][0], :]
            image_temp = cv2.resize(image_temp, (64, 64))
            image_temp = image_temp.astype(np.float) / 255.0
            image_temp = img_to_array(image_temp)
            image_temp = np.expand_dims(image_temp, axis=0)

            # Classify the input image.
            non_vehicle, vehicle = self.cnn_clf.predict(image_temp)[0]
            if vehicle > non_vehicle:
                new_bboxes.append(bbox)
                new_confidences.append(confidences[index] * vehicle)

        return new_bboxes, new_confidences

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
