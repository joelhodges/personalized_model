# Personalized Model

## Dependencies:
```
 numpy
 tensorflow
 opencv-python
 matplotlib
```

## Usage:
Create your own database by making a folder.

Within this folder, add folders for each type of item you want to identify.  The names of these folders will be used as the labels for the objects.  An example is given in this repository (the ```images``` folder).

For each object, save as many images as you want into the respective folder.  When the code runs, it will automatically split these images into training instances and testing instances.  The amount of training/testing images can be changed by modifying the variable ```TRAIN_TEST_SPLIT``` in ```custom_model.py```.  

Before running ```custom_model.py```, make sure to modify the variables ```label_file``` and ```model_file```.  These store the labels for your model and the model itself.  The label file should have the extension ```.txt``` and the model file should have the extension ```.model```.

After you run ```custom_model.py```, you can then run ```predictor.py``` to see what the model predicts for various inputs.  Before running, change the values of ```label_file``` and ```model_file``` to match the names you specified in ```custom_model.py```.  Also change the value of ```test_image``` to the image you want to test (I believe it has to be ```.jpg```, but other formats may work as well - not tested yet).  The program will print the prediction to console, as well as outputting a plot that shows the image and the prediction.

## Video Training
The file ```video_training.py``` can be used to train a model on videos rather than folders of images.  The process is the same as outlined above for images, except your 'database' folder should be a folder with videos inside, rather than a folder of folders of images.
