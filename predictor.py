# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 19:47:07 2023

@author: joeli
"""

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import models

label_file = 'Labels.txt'
model_file = 'custom.model'
test_image = 'apple.jpg'

class_names = []

# open file and read the content in a list
with open(r'./'+label_file, 'r') as fp:
    for line in fp:
        # remove linebreak from a current name
        # linebreak is the last character of each line
        x = line[:-1]

        # add current item to the list
        class_names.append(x)

model = models.load_model(model_file)

img = cv.imread(test_image)
resized = cv.resize(img, (64,64), interpolation = cv.INTER_AREA)
img = cv.cvtColor(resized, cv.COLOR_BGR2RGB)

plt.imshow(img, cmap=plt.cm.binary)


prediction = model.predict(np.array([img])/255)
index = np.argmax(prediction)

plt.title(f'Prediction is {class_names[index]}')

print(f'Prediction is {class_names[index]}')