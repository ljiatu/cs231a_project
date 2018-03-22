import glob
import os

import cv2
import numpy as np
from keras.models import load_model
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from constants import *
from hog_utils import extract_features
from lenet import LeNet


def get_svm_model():
    cars, non_cars = _read_dataset()

    car_features = extract_features(
        cars,
        orient=ORIENT,
        pixels_per_cell=PIXELS_PER_CELL,
        cells_per_block=CELLS_PER_BLOCK
    )
    non_car_features = extract_features(
        non_cars,
        orient=ORIENT,
        pixels_per_cell=PIXELS_PER_CELL,
        cells_per_block=CELLS_PER_BLOCK
    )

    X = np.vstack((car_features, non_car_features)).astype(np.float64)
    scaler = StandardScaler().fit(X)

    if os.path.exists('models/svm.pkl'):
        model = joblib.load('models/svm.pkl')
    else:
        data = scaler.transform(X)
        labels = np.hstack((np.ones(len(car_features)), np.zeros(len(non_car_features))))

        rand_state = np.random.randint(0, 100)
        X_train, X_test, Y_train, Y_test = train_test_split(data, labels, test_size=0.2, random_state=rand_state)
        model = SVC()
        model.fit(X_train, Y_train)
        print 'Test Accuracy of SVM = ', round(model.score(X_test, Y_test), 4)
        joblib.dump(model, 'models/svm.pkl')

    return model, scaler


def get_cnn_model():
    if os.path.exists('models/lenet.model'):
        model = load_model('models/lenet.model')
    else:
        cars, non_cars = _read_dataset()
        normalized_cars = [car / np.linalg.norm(car) for car in cars]
        normalized_non_cars = [non_car / np.linalg.norm(non_car) for non_car in non_cars]
        num_cars = len(normalized_cars)
        num_non_cars = len(normalized_non_cars)
        normalized_cars.extend(normalized_non_cars)
        data = np.array(normalized_cars)
        labels = np.append(np.ones(num_cars), np.zeros(num_non_cars))

        # Partition the data into training and testing splits using 80% of
        # the data for training and the remaining 20% for testing.
        rand_state = np.random.randint(0, 100)
        X_train, X_test, Y_train, Y_test = train_test_split(data, labels, test_size=0.2, random_state=rand_state)

        # Convert the labels from integers to vectors.
        Y_train = to_categorical(Y_train, num_classes=2)
        Y_test = to_categorical(Y_test, num_classes=2)

        # Initialize the model. Depth is set to 1 since we are training on grayscale data.
        model = LeNet.build(width=64, height=64, depth=3, classes=2)
        opt = Adam(lr=INIT_LR, decay=INIT_LR/EPOCHS)
        model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

        # Construct the image generator for data augmentation.
        aug = ImageDataGenerator(
            rotation_range=30,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )

        # Train the network.
        model.fit_generator(
            aug.flow(X_train, Y_train, batch_size=BATCH_SIZE),
            validation_data=(X_test, Y_test),
            steps_per_epoch=len(X_train)/BATCH_SIZE,
            epochs=EPOCHS,
            verbose=1
        )

        # Save the model to disk.
        model.save('models/lenet.model')

    return model


def _read_dataset():
    """
    Reads the car and non-car images into two lists.
    """
    car_images = glob.glob('dataset/vehicles/**/*.png')
    non_car_images = glob.glob('dataset/non-vehicles/**/*.png')

    cars = [cv2.imread(img) for img in car_images]
    non_cars = [cv2.imread(img) for img in non_car_images]

    return cars, non_cars
