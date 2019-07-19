import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from model.cHawk import cHawk

data_path = os.path.abspath('./data/train_data.csv')
train_data = pd.DataFrame(pd.read_csv(data_path))
model = cHawk(train_data)
model.load()


def intensity_drawing(i, d):
    start = model.t[i][0]
    end = model.t[i][-1]

    time = np.arange(start - 1, end + 10, 0.01)
    intensity = []
    for t in time:
        intensity.append(model.intensity(t, i=i, d=d))
    plt.plot(np.array(intensity))
    plt.show()


def matrix_drawing(A):
    plt.imshow(A)
    plt.show()


intensity_drawing(41976, 0)
intensity_drawing(41976, 1)
intensity_drawing(41976, 5)
intensity_drawing(41976, 6)
intensity_drawing(41976, 9)

matrix_drawing(model.A)
matrix_drawing(model.u)
