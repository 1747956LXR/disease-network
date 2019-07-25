import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from model.cHawk import cHawk
train_data = pd.DataFrame(pd.read_csv('./simulation/generated_data.csv'))
model = cHawk(train_data)
model.optimize()
model.save('./simulation/A.npy', './simulation/u.npy', './simulation/loss.npy')