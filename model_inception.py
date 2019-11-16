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
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.inception_v3 import preprocess_input


import sklearn
from math import ceil
import matplotlib.pyplot as plt


if os.path.exists("./model.h5"):
    os.remove("./model.h5")

samples = []
batch_size = 32
with open('./data/driving_log.csv') as logs:
    reader = csv.reader(logs)
    next(reader)
    for line in reader:
        samples.append(line)

samples = samples[5000:6000]

def my_preprocess(path):
    img = cv2.imread(path)
    #img = img[50:130, :]
    return img

#def generator(samples, batch_size=32):
#    num_samples = len(samples)
#    while 1: # Loop forever so the generator never terminates
#        sklearn.utils.shuffle(samples)
#        for offset in range(0, num_samples, batch_size):
def generate_batch(samples_part):
    angles = []
    images = []
    for sample in samples_part:
        #batch_samples = samples[offset:offset+batch_size]

        measurment = float(sample[3])
        center_image = my_preprocess(os.path.join('data', sample[0]))
        image_left = my_preprocess(os.path.join('data', sample[1].strip()))
        image_right = my_preprocess(os.path.join('data', sample[2].strip()))

        offset = 0.2
        #print(measurment)
        angles.extend([measurment, measurment - offset, measurment + offset])
        images.extend([center_image, image_left, image_right])
        #X_train.append(center_image)
        #y_train.append(measurment)
    # trim image to only see section with road
    X_train = np.array(images)
    y_train = np.array(angles)
    return sklearn.utils.shuffle(X_train, y_train)

train_samples, validation_samples = train_test_split(samples, test_size=0.2)
X_train, y_train = generate_batch(train_samples)
X_val, y_val = generate_batch(validation_samples)

#train_generator = generator(train_samples, batch_size=batch_size)
#validation_generator = generator(validation_samples, batch_size=batch_size)

input_shape=(160, 320, 3)

input_shape_resized=(90, 320, 3)

freeze_flag = True  # `True` to freeze layers, `False` for full training
weights_flag = 'imagenet' # 'imagenet' or None
preprocess_flag = True # Should be true for ImageNet pre-trained typically

inception = InceptionV3(weights=weights_flag, include_top=False,
                        input_shape=input_shape_resized)


if freeze_flag == True:
    for layer in inception.layers:
        layer.trainable = False


if preprocess_flag == True:
    datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
else:
    datagen = ImageDataGenerator()
    val_datagen = ImageDataGenerator()

input = Input(shape=input_shape)

# Re-sizes the input with Kera's Lambda layer & attach to cifar_input
resized_input = Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3))(input)
#Lambda(lambda image: image[50:130, :])(resized_input)

inp = inception(resized_input)


x = GlobalAveragePooling2D()(inp)
dense1 = Dense(512, activation = 'relu')(x)
dense2 = Dense(82, activation = 'relu')(dense1)
prediction = Dense(1)(dense2)

model = Model(inputs=input, outputs=prediction)

# Compile the model
model.compile(optimizer='Adam', loss='mse', metrics=['accuracy'])

# Check the summary of this new model to confirm the architecture
model.summary()

# history_object = model.fit_generator(train_generator,
#            steps_per_epoch=ceil(len(train_samples)/batch_size),
#            validation_data=validation_generator,
#            validation_steps=ceil(len(validation_samples)/batch_size),
#            epochs=5, verbose=1)

history_object = model.fit_generator(datagen.flow(X_train, y_train, batch_size=batch_size),
                    steps_per_epoch=len(X_train)/batch_size, epochs=2, verbose=1,
                    validation_data=val_datagen.flow(X_val, y_val, batch_size=batch_size),
                    validation_steps=len(X_val)/batch_size)


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

