import numpy as np
import pandas as pd

train_data = pd.DataFrame(pd.read_csv('./simulation/generated_data.csv'))
print(train_data)

A = np.load("./simulation/A0.npy")
u = np.load("./simulation/u0.npy")

learned_A = np.load("./simulation/A.npy")
learned_u = np.load("./simulation/u.npy")

import matplotlib.pyplot as plt

plt.subplot(2, 2, 1)
plt.imshow(A)
plt.subplot(2, 2, 2)
plt.imshow(learned_A)
plt.subplot(2, 2, 3)
plt.imshow(u)
plt.subplot(2, 2, 4)
plt.imshow(learned_u)
plt.show()