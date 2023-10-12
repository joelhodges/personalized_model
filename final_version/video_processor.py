# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 12:49:59 2023

@author: joeli
"""

import cv2 as cv
import os

rootfolder = "/Users/joeli/Documents/1 Penn State/CMPSC 483W/App Mock/Database" # The database folder
model_file = "custom.model" #The model folder

NUM_SAMPLES = 20 # Number of images to use to train the model
SIZE = 128 # Image size, in pixels

# Allowed extensions for videos and images
VIDEO_EXTENSIONS = ['.mov', '.mp4']
IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png']

# Gets the time that the latest update was made to a given folder
def get_update_time(folder):
    max_mod_time = 0
    for (root,dirs,files) in os.walk(folder, topdown=True):
            
            # Check latest update time for each file
            for file in files:
                max_mod_time = max(max_mod_time, os.path.getmtime(root+'/'+file))
                
            # Check latest update time for each directory
            for directory in dirs:
                max_mod_time = max(max_mod_time, os.path.getmtime(root+'/'+directory))
    return max_mod_time

# Take all media within folder and downsize it to correct resolution
def downsize(folder):
    gallery = folder + "/Gallery" # The path to the Gallery folder for the current object
    downsized = folder + "/Model" # The path to the Model folder for the current object
    step = get_step_size(gallery)
    count = 0
    
    # Remove all images from the current Model folder
    for file in os.listdir(downsized):
        os.remove(downsized + "/" + file)
    
    # Add new images to the Model folder
    for file in os.listdir(gallery):
        extension = file[file.index('.'):]
        
        # Process videos
        if extension.lower() in VIDEO_EXTENSIONS:
            vidcap = cv.VideoCapture(gallery + "/" + file)
            success,image = vidcap.read()
            while success:
                
                # Resize current frame and write to the Model folder
                resized = cv.resize(image, (SIZE, SIZE), interpolation = cv.INTER_AREA)
                cv.imwrite(f'{downsized}/image_{count}.jpg', resized)
                count+=1
              
                # Take a certain number of steps to get to approximately NUM_SAMPLES images
                for i in range(step):
                    success,image = vidcap.read()
                  
        # Process images
        elif extension.lower() in IMAGE_EXTENSIONS:
            
            # Resize image and write to the Model folder
            image = cv.imread(gallery + '/' + file)
            resized = cv.resize(image, (SIZE, SIZE), interpolation = cv.INTER_AREA)
            cv.imwrite(f'{downsized}/image_{count}.jpg', resized)
            count+=1
    
# Calculate the step size for videos based on number of total frames in a folder
def get_step_size(folder):
    frame_count = 0
    needed_samples = NUM_SAMPLES
    
    # Look through each file in the folder (all should be image or video format)
    for file in os.listdir(folder):
        extension = file[file.index('.'):]
        
        # Calculate number of frames in each video and add to the total frame count
        if extension.lower() in VIDEO_EXTENSIONS:
            vidcap = cv.VideoCapture(folder + "/" + file)
            frame_count += int(vidcap.get(cv.CAP_PROP_FRAME_COUNT))
            
        # All images are included in the model, so less samples are needed from the videos for each image found
        elif extension.lower() in IMAGE_EXTENSIONS:
            needed_samples -= 1
            
    return frame_count//needed_samples

folders = os.listdir(rootfolder) # Stores all folder names within the database

model_time = get_update_time(model_file) # The last update time of the model
need_proc = [] # Stores all folders that need to be reprocessed

# Look through each folder in the database
for folder in folders:
    
    # Gets the last time the Gallery was modified for the current object
    gallery_mod_time = os.path.getmtime(rootfolder + '/' + folder + '/Gallery')
    
    # Gallery only needs reprocessing if it has been updated since the last time the model was trained
    if (model_time < gallery_mod_time):
        need_proc.append(folder)

# Run downsizing for each folder that needs processing
for folder in need_proc:
    downsize(rootfolder + '/' + folder)
