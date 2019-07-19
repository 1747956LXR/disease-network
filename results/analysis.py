import os 
import numpy as np 
import matplotlib.pyplot as plt 

all_loss = []
for i in range(31):
    loss = np.load('./results/loss' + str(i) + '.npy')
    all_loss.append(loss[-1])

all_loss = np.array(all_loss)

plt.plot(all_loss, 'ro')
plt.show()