import numpy as np
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Reshape, Activation, BatchNormalization
from tensorflow.keras.utils import to_categorical, plot_model


data = np.load('sudoku.npz')
train = data['train'] / 9
train = train.reshape((train.shape[0], 9, 9, 1))
target = data['target']
