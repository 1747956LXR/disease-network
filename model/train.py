import pandas as pd 
import os 

data_path = os.path.abspath('./data/train_data.csv')
train_data = pd.DataFrame(pd.read_csv(data_path))
print(train_data)