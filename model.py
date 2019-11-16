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
from keras.applications.inception_v3 import preprocess_input
from sklearn.model_selection import train_test_split

import sklearn
from math import ceil
import matplotlib.pyplot as plt

samples = []
batch_size = 32
with open('./data/driving_log.csv') as logs:
    reader = csv.reader(logs)
    next(reader)
    for line in reader:
        samples.append(line)

samples = samples[1:100]

def my_preprocess(path):
    img = cv2.imread(path)
    #img = img[50:130, :]
    return preprocess_input(img)

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                measurment = float(batch_sample[3])
                center_image = my_preprocess(os.path.join('data', batch_sample[0]))
                image_left = my_preprocess(os.path.join('data', batch_sample[1].strip()))
                image_right = my_preprocess(os.path.join('data', batch_sample[2].strip()))
                offset = 0.2
                angles.extend([measurment, measurment - offset, measurment + offset])
                images.extend([center_image, image_left, image_right])

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)



train_samples, validation_samples = train_test_split(samples, test_size=0.2)

train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)

input_shape=(160, 320, 3)

input_shape_resized=(90, 320, 3)

freeze_flag = False  # `True` to freeze layers, `False` for full training
weights_flag = 'imagenet' # 'imagenet' or None
preprocess_flag = True # Should be true for ImageNet pre-trained typically

inception = InceptionV3(weights=weights_flag, include_top=False,
                        input_shape=input_shape_resized)


if freeze_flag == True:
    for layer in inception.layers:
        layer.trainable = False

input = Input(shape=input_shape)

# Re-sizes the input with Kera's Lambda layer & attach to cifar_input
resized_input = Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3))(input)
#Lambda(lambda image: image[50:130, :])(resized_input)

inp = inception(resized_input)


x = GlobalAveragePooling2D()(inp)
dense1 = Dense(512, activation = 'relu')(x)
dense2 = Dense(82, activation = 'relu')(dense1)
prediction = Dense(1, activation = 'relu')(dense2)

model = Model(inputs=input, outputs=prediction)

# Compile the model
model.compile(optimizer='Adam', loss='mse', metrics=['accuracy'])

# Check the summary of this new model to confirm the architecture
model.summary()

history_object = model.fit_generator(train_generator,
           steps_per_epoch=ceil(len(train_samples)/batch_size),
           validation_data=validation_generator,
           validation_steps=ceil(len(validation_samples)/batch_size),
           epochs=5, verbose=1)


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

