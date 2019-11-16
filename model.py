import csv
import cv2
import numpy as np
import os
from keras.applications.inception_v3 import InceptionV3
import tensorflow as tf
from keras.models import Model

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout, Lambda
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Cropping2D, GlobalAveragePooling2D, Input
from keras.optimizers import Adam
import random


import sklearn
from math import ceil
import matplotlib.pyplot as plt


if os.path.exists("./model.h5"):
    os.remove("./model.h5")

batch_size = 32

def my_preprocess(path):
    img = cv2.imread(path)
    #img = img[50:130, :]
    return img

#def generator(samples, batch_size=32):
#    num_samples = len(samples)
#    while 1: # Loop forever so the generator never terminates
#        sklearn.utils.shuffle(samples)
#        for offset in range(0, num_samples, batch_size):


def generate_batch(path, images, angles):
    samples = []

    with open(os.path.join(path,'driving_log.csv')) as logs:
        reader = csv.reader(logs)
        next(reader)
        for line in reader:
            samples.append(line)

    #angles = []
    #images = []
    offset = 0.2

    for sample in samples:
        #batch_samples = samples[offset:offsetbatch_size]

        measurment = float(sample[3])
        center_image = my_preprocess(sample[0])
        image_left = my_preprocess(sample[1])
        image_right = my_preprocess(sample[2])
        #if (center_image is not None):
            #print(measurment)
        angles.extend([measurment, measurment + offset, measurment - offset])
        images.extend([center_image, image_left, image_right])
        #X_train.append(center_image)
        #y_train.append(measurment)
    # trim image to only see section with road
    #x = np.array(images)
    #np.concatenate((X_data, x), axis = 0)
    #np.concatenate((y_data, np.array(angles)),axis=0)
    #return
    #

def generate_data():
    sources = ['data/1_forward', 'data/1_backwards', 'data/1_forward_oscilation',  'data/2_forward_2', 'data/2_backwards_oscilation', 'data/2_backwards','data/1_backwards_oscilation']
    samples = []

    for source in sources:
        with open(os.path.join(source, 'driving_log.csv')) as logs:
            reader = csv.reader(logs)
            next(reader)
            for line in reader:
                samples.append(line)

        # angles = []
        # images = []
    offset = 0.25
    angles = []
    images = []
    for sample in samples:
        # batch_samples = samples[offset:offsetbatch_size]

        measurment = float(sample[3])
        center_image = my_preprocess(sample[0])

        if (measurment != 0):
            image_left = my_preprocess(sample[1])
            image_right = my_preprocess(sample[2])
            angles.extend([measurment + offset, measurment - offset, measurment - offset, measurment + offset, -measurment])
            images.extend([image_left,np.flip(image_left), image_right,np.flip(image_right), np.flip(center_image)])
        elif (random.randint(0,1) == 1):
            images.append(center_image)
            angles.append(measurment)


    X_train = np.array(images)
    y_train = np.array(angles)
    return sklearn.utils.shuffle(X_train, y_train)

def visualize_dataset(out):
    max= out.max()
    res = 100
    representations = np.zeros(int(max*res) +1)
    counter = 0
    for i in out:
        if (abs(i) ==0.2):
            counter +=1
        #print(i)
        val = int(abs(i) * res)
        representations[val] += 1
    #print(counter)
    plt.bar(np.arange(max*res +1), representations)
    #plt.xticks(np.arange(bins), np.arange(max))
    plt.show()

batch_size = 32
#
# sources = ['data/1_forward', 'data/1_backwards', 'data/1_forward_oscilation']
# samples = []
# for source in sources:
#     with open(os.path.join(source, 'driving_log.csv')) as logs:
#         reader = csv.reader(logs)
#         next(reader)
#         for line in reader:
#             samples.append(line)
#
# X_data, y_data = generate_batch_old(samples)
X_data, y_data = generate_data()

#visualize_dataset(y_data)
#generate_batch('data/1_forward_oscilation', X_data, y_data)


#X_val, y_val = generate_batch(validation_samples)

#train_generator = generator(train_samples, batch_size=batch_size)
#validation_generator = generator(validation_samples, batch_size=batch_size)

input_shape=(160, 320, 3)

model = Sequential()
#model.add(Lambda(lambda x: tf.image.rgb_to_grayscale(x)))
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=input_shape))
model.add(Cropping2D(cropping=((50,20), (0,0))))
model.add(Conv2D(3, 5, 3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(24, 5, 3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(36, 5, 3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(48, 3, 1, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Conv2D(64, 3, 1, activation='relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
#model.add(Dense(1164, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(50, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='relu'))
#model.add(Dropout(0.2))
model.add(Dense(1))

adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)

model.compile(loss='mse', optimizer=adam)
model.summary()
from keras.callbacks import EarlyStopping
early_stopping_monitor = EarlyStopping(patience=2)

history_object = model.fit(X_data, y_data, validation_split=0.2, shuffle=True, epochs=10, callbacks=[early_stopping_monitor])


#model.fit_generator(datagen.flow(X_train, y_train, batch_size=batch_size),
#                    steps_per_epoch=len(X_train)/batch_size, epochs=2, verbose=1,
#                    validation_data=val_datagen.flow(X_val, y_val, batch_size=batch_size),
#                    validation_steps=len(X_val)/batch_size)


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

