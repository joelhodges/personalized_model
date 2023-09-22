# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 11:53:09 2023

@author: joeli
"""

import cv2 as cv
from tensorflow.keras import layers, models
import os

rootfolder = "/Users/joeli/Documents/1 Penn State/CMPSC 483W/Video_Test"
label_file = "Labels.txt"
model_file = "custom.model"
TRAIN_TEST_SPLIT = 0.8

# Get folder names, used as labels
videos = os.listdir(rootfolder)


training_images, test_images, training_labels, test_labels = [],[],[],[]

# Pull all data from folders
for video in videos:
    path = rootfolder + '/' + video
    
    frames = []
    vidcap = cv.VideoCapture(path)
    success,image = vidcap.read()
    count = 0
    while success:
      resized = cv.resize(image, (64, 64), interpolation = cv.INTER_AREA)
      img = cv.cvtColor(resized, cv.COLOR_BGR2RGB)
      img = img/255
      img = img.tolist()
      frames.append(img)
      
      success,image = vidcap.read()
    
    for count in range(len(frames)):
        
        # If last image is reached with no images in the test group, add the last image to the test group
        if(count == len(frames)-1) and (videos.index(video) not in test_labels):
            test_images.append(img)
            test_labels.append(videos.index(video))
            continue
        
        # Split data between training and testing
        # Can be modified using the TRAIN_TEST_SPLIT constant
        if count/len(frames) < TRAIN_TEST_SPLIT:
            training_images.append(img)
            training_labels.append(videos.index(video))
        else:
            test_images.append(img)
            test_labels.append(videos.index(video))
            
    print(f'{video}: {len(frames)} frames')
                

# Create the neural net
model = models.Sequential()
model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(64,64,3)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64, (3,3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64, (3,3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(len(videos), activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(training_images, training_labels, epochs=10, validation_data=(test_images, test_labels))

# Print statistics about the model
loss, accuracy = model.evaluate(test_images, test_labels)
print(f"Loss: {loss}")
print(f"Accuracy: {accuracy}")

# Save the model to use for testing
model.save(model_file)

with open(r'./'+label_file, 'w') as fp:
    fp.write('\n'.join(videos))