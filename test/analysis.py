import numpy as np

A = np.load("./model/A.npy")
u = np.load("./model/u.npy")

learned_A = np.load("./test/learned_A.npy")
learned_u = np.load("./test/learned_u.npy")

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