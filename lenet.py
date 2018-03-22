# coding=utf-8
from keras import Sequential
from keras.layers import Activation, Conv2D, Dense, Flatten, K, MaxPooling2D


class LeNet(object):
    """
    Based on [LeCun98] LeCun, Y., Bottou, L., Bengio, Y., and Haffner, P. (1998d).
     Gradient-based learning applied to document recognition. Proceedings of the IEEE, 86(11), 2278–2324.
    """
    @staticmethod
    def build(width, height, depth, classes):
        model = Sequential()
        input_shape = (height, width, depth)

        if K.image_data_format() == 'channels_first':
            input_shape = (depth, height, width)

        model.add(Conv2D(20, (5, 5), padding='same', input_shape=input_shape))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        model.add(Conv2D(50, (5, 5), padding='same'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        model.add(Flatten())
        model.add(Dense(500))
        model.add(Activation('relu'))

        model.add(Dense(classes))
        model.add(Activation('softmax'))
        
        return model
