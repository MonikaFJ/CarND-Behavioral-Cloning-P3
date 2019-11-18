import csv
import cv2
import numpy as np
import os

from keras.models import Sequential
from keras.layers.core import Dense, Flatten, Dropout, Lambda
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Cropping2D
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
import sklearn
import matplotlib.pyplot as plt

import random


def generate_data():
    def read_image(path):
        img = cv2.imread(path)
        # images read with imread are BRG, images that will be read from simulator are RGB.
        # Before training we have to convert BRG to RGB.
        return img[:, :, [2, 1, 0]]

    # sources = ['data/2_backwards_oscilation', 'data/2_backwards', 'data/2_forward_tight', 'data/2_forward_2']
    sources = ['data/1_forward', 'data/1_backwards', 'data/1_forward_oscilation', 'data/1_backwards_oscilation']
    samples = []

    for source in sources:
        with open(os.path.join(source, 'driving_log.csv')) as logs:
            reader = csv.reader(logs)
            next(reader)
            for line in reader:
                samples.append(line)

    offset = 0.25
    angles = []
    images = []

    for sample in samples:

        measurement = float(sample[3])
        center_image = read_image(sample[0])

        # If the steering angle is different than 0, generate more data (add side cameras and flipped images)
        if measurement != 0:
            image_left = read_image(sample[1])
            image_right = read_image(sample[2])
            angles.extend(
                [measurement + offset, measurement - offset, measurement - offset, measurement + offset, -measurement])
            images.extend(
                [image_left, np.flip(image_left, 1), image_right, np.flip(image_right, 1), np.flip(center_image, 1)])
        elif random.randint(0, 1) == 1:
            images.append(center_image)
            angles.append(measurement)

    X_train = np.array(images)
    y_train = np.array(angles)
    return sklearn.utils.shuffle(X_train, y_train)


def visualize_dataset(out):
    max_val = out.max()
    res = 10
    representations = np.zeros(int(max_val * res) + 1)
    counter = 0
    for i in out:
        if abs(i) == 0.2:
            counter += 1
        val = int(abs(i) * res)
        representations[val] += 1

    plt.bar(np.arange(0, max_val, 1 / res), representations, width=1 / res / 2)
    plt.show()


def build_model(learning_rate=0.001, print_summary=True):
    model = Sequential()
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=input_shape))  # normalizing
    model.add(Cropping2D(cropping=((50, 20), (0, 0))))  # Cropping top and bottom
    model.add(Conv2D(3, (5, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(24, (5, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(36, (5, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Conv2D(48, (3, 1), activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(50, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1))

    adam = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, amsgrad=False)

    model.compile(loss='mse', optimizer=adam)
    if print_summary:
        model.summary()

    return model


# Remove the model if exists
if os.path.exists("./model.h5"):
    os.remove("./model.h5")

batch_size = 32

X_data, y_data = generate_data()
input_shape = X_data[0].shape

# visualize_dataset(y_data)

model = build_model()
early_stopping_monitor = EarlyStopping(patience=1)

history_object = model.fit(X_data, y_data, validation_split=0.2, shuffle=True, epochs=20, batch_size=batch_size,
                           callbacks=[early_stopping_monitor])

### print the keys contained in the history object
model.save('model.h5')

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()
