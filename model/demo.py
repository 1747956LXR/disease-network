import matplotlib.pyplot as plt 
import numpy as np 

A = np.load("./model/A.npy")
print(A)


u = np.load("./model/u.npy")
print(u)

plt.imshow(A)
plt.show()