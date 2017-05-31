import os
import csv
import cv2
import itertools
import numpy as np
import sklearn
import sklearn.model_selection

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D

# Constants
MAX_ANGLE = 25.
DIR_SEPARATOR = "\\"
DATA_LOCATION = "data/"
CH, ROW, COL = 3, 160, 320  # Original image format

# Hyperparameters
crop_top, crop_down = 70, 26
h = 20.
bsize = 32
epochs = 3
center_only = False
validation_share = 0.2
dropout = 0.1

# Find all csv files and IMG folders in subfolders inside DATA_LOCATION
csv_locations = []
image_locations = []
for root, dirs, files in os.walk(DATA_LOCATION):
    for f in files:
        if f.endswith(".csv"):
            csv_locations.append(os.path.join(root, f))
            image_locations.append(os.path.join(root, "IMG/"))

# Read each line from all found csv files into a "samples" list
samples = []
for i in range(len(csv_locations)):
    with open(csv_locations[i]) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            samples.append([image_locations[i], line])

print("Num Samples:", len(samples))

# Augmentation configuration
# A list of all possible combinations of the lists "samples", "flipped" and
# "camera" is created to be used during data generation and easy shuffling.
# "samples" is overwritten and now contains the tuples of 3 items
camera = [-1, 0, 1] # right, center, left camera
if center_only:
    camera = [0]
flipped = [True, False]
samples = list(itertools.product(samples, camera, flipped))
print("Num Augmented Samples:", len(samples))

# Splitting the data into training and validation set
train_samples, validation_samples = sklearn.model_selection.train_test_split(samples, test_size=validation_share)
print("Num Samples in Training Set:", len(train_samples))
print("Num Samples in Validation Set:", len(validation_samples))

def generator(samples, batch_size=32):
    """
    Data generator used for sampling batches of size "batch_size" from the
    tuple configuration "sample, camera, is_flipped" where
    - sample is the line of the csv file from the car simulator
    - camera indicates which camera should be used from the data
    - is_flipped indicates if the image and steering angle should be flipped
    """
    num_samples = len(samples)
    # Loop forever so the generator never terminates
    while 1:
        # shuffle the data at the beginning
        sklearn.utils.shuffle(samples)
        # take batches until the data set is exhausted
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample, camera, is_flipped in batch_samples:
                # extract the image location
                image_location = batch_sample[0]
                sample_data = batch_sample[1]
                name = image_location + sample_data[camera%3].split(DIR_SEPARATOR)[-1]
                # read image
                image = cv2.imread(name)
                # read steering angle (target)
                angle = float(sample_data[3])
                # depending of the camera, adjust the steering angle
                # (see writeup for more details)
                if camera != 0:
                    alpha = np.radians(angle * MAX_ANGLE)
                    beta = np.arctan(np.tan(alpha) + float(camera) / h)
                    angle = np.degrees(beta) / MAX_ANGLE
                # flip image and steering angle
                if is_flipped:
                    image = cv2.flip(image, 1)
                    angle *= -1
                images.append(image)
                angles.append(angle)

            X_train = np.array(images)
            y_train = np.array(angles)
            yield X_train, y_train

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=bsize)
validation_generator = generator(validation_samples, batch_size=bsize)

def normalize_image(x):
    """
    Normalization layer for batches of images
    (Extra function so that saving and loading the model works properly)
    """
    from keras.backend import tf as ktf
    return ktf.map_fn(lambda img: ktf.image.per_image_standardization(img), x)

# Neural network model based on Nvidia's network
model = Sequential()
# Preprocess incoming data
# Crop image to road area
model.add(Cropping2D(cropping=((crop_top, crop_down), (0, 0)),
                     input_shape=(ROW, COL, CH)))
# Center around zero with small standard deviation
model.add(Lambda(normalize_image))
# Convolutional layers
model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Flatten())
# Fully connected layers with dropout
model.add(Dense(100, activation='relu'))
model.add(Dropout(dropout))
model.add(Dense(50, activation='relu'))
model.add(Dropout(dropout))
model.add(Dense(10, activation='relu'))
model.add(Dropout(dropout))
model.add(Dense(1))

# Train using adam optimizer
model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, samples_per_epoch=len(train_samples),
                    validation_data=validation_generator,
                    nb_val_samples=len(validation_samples), nb_epoch=epochs)
# Save the model
model.save('model.h5')
