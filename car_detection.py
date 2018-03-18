import math

import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage import exposure

from constants import *
from hog_utils import get_hog_features


def single_img_features(img, color_space='RGB', spatial_size=(32, 32),
                        hist_bins=32, orient=9,
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True):
    """Extract spatial, color and hog features from single image
    Args:
        img (numpy.array): image in RGB format
        color_space: GRAY, RGB, HSV, LUV, HLS, YUV, YCrCb
        spatial_size (tuple): resize img before calculating spatial features
            default value is (32, 32)
        hist_bins (int): number of histogram bins, 32 by default
        orient (int): number of HOG orientations
        pix_per_cell (int): number of pixels in HOG cell
        cell_per_block (int): number of HOG cells in block
        hog_channel (int): channel to use for HOG features calculating, default 0
        spatial_feat (boolean): calculate spatial featues, default True
        hist_feat (boolean): calculate histogram featues, default True
        hog_feat (boolean): calculate HOG featues, default True
    Returns:
        features_vector (list(numpy.array)): list of feature vectors
    """
    # Define an empty list to receive features
    img_features = []
    # Apply color conversion if other than 'RGB'
    if color_space != 'RGB':
        feature_image = cv2.cvtColor(img, getattr(cv2, 'COLOR_RGB2' + color_space))
    else:
        feature_image = np.copy(img)
    if hog_feat:
        if color_space == 'GRAY':
            hog_features = get_hog_features(feature_image, orient,
                                            pix_per_cell, cell_per_block, vis=False, feature_vec=True)
        elif hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.extend(get_hog_features(feature_image[:,:,channel],
                                                     orient, pix_per_cell, cell_per_block,
                                                     vis=False, feature_vec=True))
        else:
            hog_features = get_hog_features(feature_image[:,:,hog_channel], orient,
                                            pix_per_cell, cell_per_block, vis=False, feature_vec=True)
        # Append features to list
        img_features.append(hog_features)

    # Return concatenated array of features
    return np.concatenate(img_features)


def get_s_from_hls(img):
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    return hls[:, :, 2]


def plot_image(img, hog_img):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

    ax1.axis('off')
    ax1.imshow(img, cmap=plt.cm.gray)
    ax1.set_title('Input image')
    ax1.set_adjustable('box-forced')

    # Rescale histogram for better display
    hog_image_rescaled = exposure.rescale_intensity(hog_img, in_range=(0, 10))

    ax2.axis('off')
    ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
    ax2.set_title('Histogram of Oriented Gradients')
    ax1.set_adjustable('box-forced')
    plt.show()


def search_windows(img, windows, clf, scaler, color_space='RGB',
                   spatial_size=(32, 32), hist_bins=32,
                   hist_range=(0, 256), orient=9,
                   pix_per_cell=8, cell_per_block=2,
                   hog_channel=0, spatial_feat=True,
                   hist_feat=True, hog_feat=True):
    """Apply specified classificator to specified windows
    and returns hot windows - windows classified as holding interesting object
    Args:
        img (numpy.array): image to search
        windows (list): list of coordinates of sliding windows in form of
            ((top left x, top left y), (bottom right x, bottom right y))
        spatial_size (tuple): resize img before calculating spatial features
            default value is (32, 32)
        hist_bins (int): number of histogram bins, 32 by default
        orient (int): number of HOG orientations
        pix_per_cell (int): number of pixels in HOG cell
        cell_per_block (int): number of HOG cells in block
        hog_channel (int): channel to use for HOG features calculating, default 0
        spatial_feat (boolean): calculate spatial featues, default True
        hist_feat (boolean): calculate histogram featues, default True
        hog_feat (boolean): calculate HOG featues, default True
    Returns:
        list of hot windows
    """
    # Create an empty list to receive positive detection windows
    on_windows = []
    confidences = []
    # Iterate over all windows in the list
    for window in windows:
        # Extract the test window from original image
        if len(window) == 0:
            continue
        test_img = cv2.resize(
            img[window[0][1]:window[1][1], window[0][0]:window[1][0]],
            (64, 64),
            interpolation=cv2.INTER_AREA
        )
        # Extract features for that window using single_img_features()
        features = single_img_features(test_img, color_space=color_space,
                                       spatial_size=spatial_size, hist_bins=hist_bins,
                                       orient=orient, pix_per_cell=pix_per_cell,
                                       cell_per_block=cell_per_block,
                                       hog_channel=hog_channel, spatial_feat=spatial_feat,
                                       hist_feat=hist_feat, hog_feat=hog_feat)
        # Scale extracted features to be fed to classifier
        test_features = scaler.transform(np.array(features).reshape(1, -1))
        # Predict using your classifier
        confidence = clf.decision_function(test_features)
        # If positive then save the window
        if confidence >= 1:
            on_windows.append(window)
            confidences.append(confidence)
    # Return windows for positive detections
    return on_windows, confidences


def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None],
                 xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    """Create and return sliding window lattice as list of sliding windows
    Args:
        img (numpy.array): image to search
        x_start_stop (list): horizontal limits, if [None, None] then [0, image width]  will be used
        y_start_stop (list): vertical   limits, if [None, None] then [0, image height] will be used
        xy_window (tuple): sliding window size, default is (64, 64)
        xy_overlap (tuple): sliding window overlap factor, default is (0.5, 0.5)
    Returns:
        list of windows
    """
    # If x and/or y start/stop positions not defined, set to image size
    imgsizey = img.shape [0]
    imgsizex = img.shape [1]
    x_start_stop[0] = 0 if x_start_stop [0] is None else x_start_stop [0]
    x_start_stop[1] = imgsizex if x_start_stop [1] is None else x_start_stop [1]
    y_start_stop[0] = 0 if y_start_stop [0] is None else y_start_stop [0]
    y_start_stop[1] = imgsizey if y_start_stop [1] is None else y_start_stop [1]
    # Compute the span of the region to be searched
    sizex = x_start_stop [1] - x_start_stop [0]
    sizey = y_start_stop [1] - y_start_stop [0]
    # Compute the number of pixels per step in x/y
    stepx = int (xy_window [0] * xy_overlap [0])
    stepy = int (xy_window [1] * xy_overlap [1])
    # Compute the number of windows in x/y
    step_count_x = int (math.floor(1.0 * sizex / stepx)) - 1
    step_count_y = int (math.floor(1.0 * sizey / stepy)) - 1
    # Initialize a list to append window positions to
    window_list = []
    for i in xrange (step_count_y):
        for j in xrange (step_count_x):
            # Calculate each window position
            # Append window position to list
            window_list.append((
                (x_start_stop[0] + j*stepx, y_start_stop[0] + i*stepy),
                (x_start_stop[0] + j*stepx + xy_window[0], y_start_stop[0] + i*stepy + xy_window[1])
            ))
    # Return the list of windows
    return window_list


def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    """Draws windows or bounding boxes on the image
    Args:
        img (numpy.array): image to search
        bboxes (list): bounding boxes
        color (tuple): bounding box color, default is (0, 0, 255)
        thick (int): thickness of bounding box, default is 6 pixels
    Returns:
        image copy with boxes drawn
    """
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy


sw_x_limits = [
    [640, None],
    [640, None],
    [640, None]
]

sw_y_limits = [
    [None, None],
    [None, None],
    [None, None]
]

sw_window_size = [
    (128, 128),
    (96, 96),
    (80, 80)
]

sw_overlap = [
    (0.5, 0.5),
    (0.5, 0.5),
    (0.5, 0.5)
]


def get_hot_boxes(
        image,
        svc,
        X_scaler,
        color_space='RGB',
        spatial_size=(32, 32),
        hist_bins=32,
        orient=9,
        pix_per_cell=8,
        cell_per_block=2,
        hog_channel=0,
        spatial_feat=True,
        hist_feat=True,
        hog_feat=True
):
    """Applies sliding windows to images
    and finds hot windows. Also returns image with all hot boxes are drawn
    Args:
        image (numpy.array): image
    Returns:
        bboxes(list), image_with_bboxes_drawn(numpy.array)
    """
    all_bboxes = []
    all_confidences = []

    # iterate over previously defined sliding windows
    for x_limits, y_limits, window_size, overlap in zip(
            sw_x_limits,
            sw_y_limits,
            sw_window_size,
            sw_overlap
    ):

        windows = slide_window(
            np.copy(image),
            x_start_stop=x_limits,
            y_start_stop=y_limits,
            xy_window=window_size,
            xy_overlap=overlap
        )

        bboxes, confidences = search_windows(image, windows, svc, X_scaler, color_space=color_space,
                                     spatial_size=spatial_size, hist_bins=hist_bins,
                                     orient=orient, pix_per_cell=pix_per_cell,
                                     cell_per_block=cell_per_block,
                                     hog_channel=hog_channel, spatial_feat=spatial_feat,
                                     hist_feat=hist_feat, hog_feat=hog_feat)

        all_bboxes.extend(bboxes)
        all_confidences.append(confidences)

    return all_bboxes, all_confidences


def non_max_suppression(bboxes, confidences):
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


def process_car_detection(frame, clf, X_scaler):
    # Uncomment the following line if you extracted training
    # data from .png images (scaled 0 to 1 by mpimg) and the
    # image you are searching is a .jpg (scaled 0 to 255)
    # image = image_orig.astype(np.float32)/255

    bboxes, confidences = get_hot_boxes(
        frame,
        clf,
        X_scaler,
        color_space,
        spatial_size,
        hist_bins,
        orient,
        pix_per_cell,
        cell_per_block,
        hog_channel,
        spatial_feat,
        hist_feat,
        hog_feat
    )

    nms_bboxes = non_max_suppression(bboxes, confidences)

    image_with_hot_boxes = draw_boxes(np.copy(frame), nms_bboxes, color=(0, 0, 1), thick=4)

    return image_with_hot_boxes

