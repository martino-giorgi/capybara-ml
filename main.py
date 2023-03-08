import os

import numpy as np
from keras import Sequential
from keras.utils import np_utils
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Conv2D, MaxPooling2D, AveragePooling2D, Flatten
from tensorflow import optimizers

