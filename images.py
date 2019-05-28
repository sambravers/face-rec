import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
from imageio import imread

def load_image(filepath):
    '''
    Takes a string (filepath, of the form "x.ext") as input and returns
    the image at that filepath as a numpy array
    '''
    img = cv2.imread(filepath)
    return img

def batch_load(dir_string):
    '''
    Takes a string (dir_string, of the form "0/") as input and returns
    a numpy array containing all the .jpg's in that directory
    '''
    path = dir_string + "*.jpg"
    paths = glob.glob(path)
    imgs = np.array([load_image(fname) for fname in paths])
    return imgs

def shuffle_in_unison(a, b):
    assert len(a) == len(b)
    shuffled_a = np.empty(a.shape, dtype=a.dtype)
    shuffled_b = np.empty(b.shape, dtype=b.dtype)
    permutation = np.random.permutation(len(a))
    for old_index, new_index in enumerate(permutation):
        shuffled_a[new_index] = a[old_index]
        shuffled_b[new_index] = b[old_index]
    return shuffled_a, shuffled_b

def show_image(img):
    '''
    Given a numpy array, this function will show the image until a key is pressed
    '''
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def show_eval(model, arr, labels, num_eval):
    indices = np.random.randint(0, high=len(arr), size=(num_eval))
    for index in indices:
        prediction = np.around(model.predict(arr[index].reshape((-1, 256, 256, 3)))).reshape((2)).astype(int)
        if np.array_equal(prediction, labels[index]):
            if np.array_equal(labels[index], [1, 0]):
                print("The model correctly rejected this user!")
            else:
                print("The model correctly accepted this user!")
        else:
            if np.array_equal(labels[index], [1, 0]):
                print("The model incorrectly accepted this user!")
            else:
                print("The model incorrectly rejected this user!")
        show_image(arr[index])

def load_img(filepath):
    '''
    Given a string (filepath), this function returns the image loaded into
    a numpy array
    '''
    img = cv2.imread(filepath)
    return img

def inplace_resize(filepath, x, y):
    '''
    Given a string (filepath) and an integer (size), this function will load
    the image at that filepath and resize it to the given size (assuming
    square images)
    '''
    img = cv2.imread(filepath)
    img = cv2.resize(img, dsize=(x, y))
    cv2.imwrite(filepath, img)

def get_all_paths(stub):
    '''
    Given a string (stub), this function will return a list of all paths from
    that stub
    '''
    paths = glob.glob(stub)
    return paths
