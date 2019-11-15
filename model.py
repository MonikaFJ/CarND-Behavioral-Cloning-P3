import csv
import cv2
import numpy as np
import os

from keras.models import Sequential
from keras.layers import  Flatten, Dense

lines = []
with open('./data/driving_log.csv') as logs:
    reader = csv.reader(logs)
    for line in reader:
        lines.append(line)

images = []
measurments=[]
for i in range (1, len(lines)):
    image_path = os.path.join('data', lines[i][0])
    image = cv2.imread(image_path)
    images.append(image)
    measurments.append(float(lines[i][3]))

X_train = np.array(images)
y_train = np.array(measurments)

model = Sequential()
model.add(Flatten(input_shape=(160, 320, 3)))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True)

model.save('model.h5')