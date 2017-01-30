import numpy as np
import glob
from skimage import data
from skimage.feature import hog

orientations = 9
pixels_per_cell = (5, 5)
cells_per_block = (1, 1)

def make_data(pattern):
    paths = glob.glob(pattern)
    features = []
    for path in paths:
        image = data.imread(path)
        feature = hog(image, orientations=orientations, pixels_per_cell=pixels_per_cell, cells_per_block=cells_per_block)
        features.append(feature.tolist())
    return features

def divide(data):
    length = len(data) // 10
    train = data[:2*length] + data[3*length:6*length] + data[7*length:8*length] + data[9*length:]
    test = data[2*length:3*length] + data[6*length:7*length] + data[8*length:9*length]
    return train, test

def format(ps, ns):
#    return np.array(ps + ns), np.array([1]*len(ps) + [-1]*len(ns))
    return np.vstack(ps + ns), np.vstack([1]*len(ps) + [0]*len(ns))

data_p = make_data('../dataset/DataSet_P/*')
data_n = make_data('../dataset/DataSet_N/*')

train_p, test_p = divide(data_p)
train_n, test_n = divide(data_n)

train_data, train_label = format(train_p, train_n)
test_data, test_label = format(test_p, test_n)
print(train_label.shape)
#### Save ####
np.save('hog_data/train_data', train_data)
np.save('hog_data/train_label', train_label)
np.save('hog_data/test_data', test_data)
np.save('hog_data/test_label', test_label)
