import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from model.cHawk import cHawk

data_path = os.path.abspath('./data/train_data.csv')
train_data = pd.DataFrame(pd.read_csv(data_path))
# print(train_data)

model = cHawk(train_data)
model.load()

###

import networkx as nx
import matplotlib.pyplot as plt

G = nx.DiGraph()

A = model.A
D = model.D
max_val = A.max()
threshold = max_val * (1 - 1 / (1 * np.e))

# print(A)

for i in range(D):
    for j in range(D):
        if A[i][j] > threshold:
            G.add_edge(j, i, weight=A[i][j])

pos = nx.spring_layout(G, k=20)

edges = G.edges()
weights = [(G[u][v]['weight'] - threshold) / (max_val - threshold) * 3 + 2
           for u, v in edges]

nx.draw(
    G,
    pos,
    edges=edges,
    width=2,
    edge_color=weights,
    node_color='skyblue',
    edge_cmap=plt.cm.Blues,
    with_labels=True,
    arrowstyle='->',
    arrowsize=10,
)

plt.show()
