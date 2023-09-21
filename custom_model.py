# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 20:43:40 2023

@author: joeli
"""

import cv2 as cv
from tensorflow.keras import layers, models
import os

rootfolder = "/Users/joeli/Documents/1 Penn State/CMPSC 483W/images"
label_file = "Labels.txt"
model_file = "custom.model"
TRAIN_TEST_SPLIT = 0.8

# Get folder names, used as labels
folders = os.listdir(rootfolder)


training_images, test_images, training_labels, test_labels = [],[],[],[]

# Pull all data from folders
for folder in folders:
    folderpath = rootfolder + '/' + folder
    
    # Get names of all files in the folder
    names = os.listdir(folderpath)
    num_images = len(names)
    print(f'{folder}: {num_images} images')
    
    for count, file in enumerate(names):
        img = cv.imread(folderpath + '/' + file)
        resized = cv.resize(img, (64, 64), interpolation = cv.INTER_AREA)
        img = cv.cvtColor(resized, cv.COLOR_BGR2RGB)
        img = img/255
        img = img.tolist()
        
        # Split data between training and testing
        # Can be modified using the TRAIN_TEST_SPLIT constant
        if count/num_images < TRAIN_TEST_SPLIT:
            training_images.append(img)
            training_labels.append(folders.index(folder))
        else:
            test_images.append(img)
            test_labels.append(folders.index(folder))
                

# Create the neural net
model = models.Sequential()
model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(64,64,3)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64, (3,3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64, (3,3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(training_images, training_labels, epochs=10, validation_data=(test_images, test_labels))

# Print statistics about the model
loss, accuracy = model.evaluate(test_images, test_labels)
print(f"Loss: {loss}")
print(f"Accuracy: {accuracy}")

# Save the model to use for testing
model.save(model_file)

with open(r'./'+label_file, 'w') as fp:
    fp.write('\n'.join(folders))
