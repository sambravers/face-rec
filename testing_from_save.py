import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import numpy as np
from images import *

'''
This file uses data that isn't accessible through the repository,
it's meant for use with anyone that actually intends to implement
this network for their own personal use.
'''

# Load numpy arrays to save time
x = np.load("data/x_file.npy")
y = np.load("data/y_file.npy")

# Shuffle the images and labels in unison
x, y = shuffle_in_unison(x, y)

# Train/test splitting
x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=(1/3), random_state=0)

# Load saved model
model = tf.keras.models.load_model("models/demo.h5")

# Evaluate
model.evaluate(x_te, y_te)

# Import accept images with out-of-training-set backgrounds
#x_te = batch_load("data/me_outdoor/")
#y_te = np.array([ [0, 1] for _ in range(len(x_te))])

show_eval(model, x_te, y_te, 10)
