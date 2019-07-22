import numpy as np

ts = np.load('./test/ts.npy')
ds = np.load('./test/ds.npy')
f = np.load('./test/f.npy')

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from model.cHawk import cHawk

import pandas as pd
train_data = pd.DataFrame(
    {
        'subject_id': 1,
        'age': ts,
        'weight': f[:][1],
        'primary': ds.astype(np.int32)
    }, pd.Index(range(len(ts))))
print(train_data)
# exit(0)

model = cHawk(train_data)
model.GD()
model.save('./test/learned_A.npy', './test/learned_u.npy',
           './test/learned_loss.npy')
