import glob
import os

import cv2
import numpy as np
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.preprocessing.image import img_to_array
from keras.utils import to_categorical
from pyimagesearch.lenet import LeNet
from keras.models import load_model

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

def get_CNN_model(frame, bboxes, confidences):
    
    if not os.path.exists('car_not_car.model'):
        cars, non_cars = read_dataset()
        data = np.append([cars, non_cars])
        labels = np.append(np.repeat('vehicle', len(cars)), np.repeat('non-vehicle', len(non_cars)))
        
        EPOCHS = 25
        INIT_LR = 1e-3
        BATCH_SIZE = 32
        # partition the data into training and testing splits using 75% of
        # the data for training and the remaining 25% for testing
        (trainX, testX, trainY, testY) = train_test_split(data,
        	labels, test_size=0.2, random_state=2018)
        
        # convert the labels from integers to vectors
        trainY = to_categorical(trainY, num_classes=2)
        testY = to_categorical(testY, num_classes=2)
        
        # initialize the model
        model = LeNet.build(width=28, height=28, depth=3, classes=2)
        opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
        model.compile(loss="binary_crossentropy", optimizer=opt,
        	metrics=["accuracy"])
        
        # construct the image generator for data augmentation
        aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
        	height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
        	horizontal_flip=True, fill_mode="nearest")

        # train the network
        H = model.fit_generator(aug.flow(trainX, trainY, batch_size=BATCH_SIZE),
        	validation_data=(testX, testY), steps_per_epoch=len(trainX) // BATCH_SIZE,
        	epochs=EPOCHS, verbose=1)
        
        # save the model to disk
        model.save("car_not_car.model")
        
    else:
        model = load_model("car_not_car.model")
    
    index = 0
    for box in bboxes:
        image_temp = frame[box[0][1]:box[1][1], box[0][0]:box[1][0],:]
        # pre-process the image for classification
        image_temp = cv2.resize(image_temp, (28, 28))
        image_temp = image_temp.astype("float") / 255.0
        image_temp = img_to_array(image_temp)
        image_temp = np.expand_dims(image_temp, axis=0)
        
        # classify the input image
        (nonVehicle, vehicle) = model.predict(image_temp)[0]
        if nonVehicle > vehicle:
            bboxes.remove(box)
            confidences.remove(confidences[index])
        else:
            confidences[index] *= vehicle
            index += 1
            
    return bboxes, confidences