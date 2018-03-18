import cv2
import numpy as np
from skimage.feature import hog


def get_hog_features(
        img,
        orient,
        pix_per_cell,
        cell_per_block,
        vis=False,
        feature_vec=True
):
    """Return HOG features and its visualization if needed
    Args:
        img (numpy.array): image for feature extraction
        orient (int): number of orientations
        pix_per_cell (int): number of pixels in HOG cell
        cell_per_block (int): number of HOG cells in block
        feature_vec (boolean): default True, return features as vector
    Returns:
        features or features, hog_image(image)
    """
    features = hog(
        img,
        orientations=orient,
        pixels_per_cell=(pix_per_cell, pix_per_cell),
        cells_per_block=(cell_per_block, cell_per_block),
        transform_sqrt=True,
        visualise=vis,
        feature_vector=feature_vec
    )

    return features


def extract_features(
        imgs,
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
    """Extract spatial, color and hog features from specified list of images
    Args:
        imgs (list): list of images in RGB format
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
        features_vector (numpy.array)
    """
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for image in imgs:
        file_features = []
        # apply color conversion if other than 'RGB'
        if color_space != 'RGB':
            feature_image = cv2.cvtColor(image, getattr(cv2, 'COLOR_RGB2' + color_space))
        else:
            feature_image = np.copy(image)

        if hog_feat:
            # Call get_hog_features() with vis=False, feature_vec=True
            if color_space == 'GRAY':
                hog_features = get_hog_features(
                    feature_image,
                    orient,
                    pix_per_cell,
                    cell_per_block,
                    vis=False,
                    feature_vec=True
                )
            elif hog_channel == 'ALL':
                hog_features = []
                for channel in range(feature_image.shape[2]):
                    hog_features.append(get_hog_features(feature_image[:,:,channel],
                                                         orient, pix_per_cell, cell_per_block,
                                                         vis=False, feature_vec=True))
                hog_features = np.ravel(hog_features)
            else:
                hog_features = get_hog_features(feature_image[:,:,hog_channel], orient,
                                                pix_per_cell, cell_per_block, vis=False, feature_vec=True)
            # Append the new feature vector to the features list
            file_features.append(hog_features)
        features.append(np.concatenate(file_features))
    # Return list of feature vectors
    return features
