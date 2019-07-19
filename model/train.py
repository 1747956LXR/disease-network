import pandas as pd
import os
from cHawk import cHawk

data_path = os.path.abspath('./data/train_data.csv')
train_data = pd.DataFrame(pd.read_csv(data_path))

for i in range(100):
    model = cHawk(train_data)
    model.optimize()
    model.save('./results/A' + str(i) + '.npy',
               './results/u' + str(i) + '.npy',
               './results/loss' + str(i) + '.npy')
