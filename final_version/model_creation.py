# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 20:43:40 2023

@author: joeli
"""

import cv2 as cv
from tensorflow.keras import layers, models
import os
import numpy.random as rd

rootfolder = "/Users/joeli/Documents/1 Penn State/CMPSC 483W/App Mock/Database" # The database folder
label_file = "Labels.txt" # The file to store the labels
model_file = "custom.model" # The folder that stores the model
TRAIN_TEST_SPLIT = 0.8 # Percentage of images to use for training
SIZE = 128 # Image size, in pixels

# Get folder names, used as labels
folders = os.listdir(rootfolder)


training_images, test_images, training_labels, test_labels = [],[],[],[]

# Pull all data from Model folders
for folder in folders:
    folderpath = rootfolder + '/' + folder + "/Model"
    
    # Get names of all files in the folder
    names = os.listdir(folderpath)
    num_images = len(names)
    
    for count, file in enumerate(names):
        img = cv.imread(folderpath + '/' + file)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        img = img/255
        img = img.tolist()
        
        # If last image is reached with no images in the test group, add the last image to the test group
        if(count == num_images-1) and (folders.index(folder) not in test_labels):
            test_images.append(img)
            test_labels.append(folders.index(folder))
            continue
        
        # Split data between training and testing
        # Can be modified using the TRAIN_TEST_SPLIT constant
        if rd.rand() < TRAIN_TEST_SPLIT:
            training_images.append(img)
            training_labels.append(folders.index(folder))
        else:
            test_images.append(img)
            test_labels.append(folders.index(folder))
                
# Create the neural net
model = models.Sequential()
model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(SIZE,SIZE,3)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64, (3,3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(128, (3,3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(len(folders), activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(training_images, training_labels, epochs=10, validation_data=(test_images, test_labels), verbose = 1)

# Save the model to use for testing
model.save(model_file)

# Write the labels to the specified file
with open(r'./'+label_file, 'w') as fp:
    fp.write('\n'.join(folders))
