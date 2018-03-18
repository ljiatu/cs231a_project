import glob
import os

import cv2
import numpy as np
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from constants import *
from hog_utils import extract_features


def read_dataset():
    """
    Read the car and non-car images into two lists.

    The car and non-car images are under vehicles/ and
    non-vehicles/ directories, respectively.
    """
    car_images = glob.glob('dataset/vehicles/**/*.png')
    non_car_images = glob.glob('dataset/non-vehicles/**/*.png')

    cars = [cv2.imread(img) for img in car_images]
    non_cars = [cv2.imread(img) for img in non_car_images]

    return cars, non_cars


def get_model():
    cars, non_cars = read_dataset()

    car_features = extract_features(
        cars,
        color_space=color_space,
        spatial_size=spatial_size,
        hist_bins=hist_bins,
        orient=orient,
        pix_per_cell=pix_per_cell,
        cell_per_block=cell_per_block,
        hog_channel=hog_channel,
        spatial_feat=spatial_feat,
        hist_feat=hist_feat,
        hog_feat=hog_feat
    )
    non_car_features = extract_features(
        non_cars,
        color_space=color_space,
        spatial_size=spatial_size,
        hist_bins=hist_bins,
        orient=orient,
        pix_per_cell=pix_per_cell,
        cell_per_block=cell_per_block,
        hog_channel=hog_channel,
        spatial_feat=spatial_feat,
        hist_feat=hist_feat,
        hog_feat=hog_feat
    )

    X = np.vstack((car_features, non_car_features)).astype(np.float64)
    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X)
    # Apply the scaler to X
    scaled_X = X_scaler.transform(X)

    # Define the labels vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(non_car_features))))

    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(
        scaled_X,
        y,
        test_size=0.2,
        random_state=rand_state
    )

    print('Feature vector length:', len(X_train[0]))

    if os.path.exists('models/svc.pkl'):
        clf = joblib.load('models/svc.pkl')
    else:
        clf = SVC()

        # Check the training time for the SVC
        clf.fit(X_train, y_train)
        # Check the score of the SVC
        print('Test Accuracy of SVC = ', round(clf.score(X_test, y_test), 4))
        joblib.dump(clf, 'models/svc.pkl')

    return clf, X_scaler
