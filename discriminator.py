import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import numpy as np
from images import *

'''
### Use this section to update the numpy files after a dataset change
# Image loading
x_rej= batch_load("data/144/faces/")
x_acc = batch_load("data/144/alfonso_cuaron/")

# Label creation
y_rej = np.array([ [1, 0] for _ in range(len(x_rej))])
y_acc = np.array([ [0, 1] for _ in range(len(x_acc))])


# Concatenation
x = np.concatenate((x_rej, x_acc))
y = np.concatenate((y_rej, y_acc))

# Save new files
np.save("data/144/x_file.npy", x, allow_pickle=False)
np.save("data/144/y_file.npy", y, allow_pickle=False)
'''

# Load numpy arrays to save time
x = np.load("data/144/x_file.npy")
y = np.load("data/144/y_file.npy")

# Shuffle the images and labels in unison
x, y = shuffle_in_unison(x, y)

# Train/test splitting
x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=(1/3), random_state=0)

# Model definition
model = models.Sequential()
model.add(layers.Conv2D(64, (4, 4), activation='relu', input_shape=(144, 144, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(32, (2, 2), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(32, (2, 2), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(16, (2, 2), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(2, activation='softmax'))

# Summary
model.summary()

# Compilation
model.compile(optimizer='Adadelta', loss='categorical_crossentropy', metrics=['accuracy'])

# Fit
model.fit(x_tr, y_tr, epochs=100, batch_size=64)

# Save model
#tf.keras.models.save_model(model, "models/144/demo.h5")

# Evaluate
model.evaluate(x_te, y_te)

# Show evaluation of 10 random test images
show_eval(model, x_te, y_te, 25)
