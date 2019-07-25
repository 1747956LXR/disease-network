import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

A0 = np.load('./simulation/A0.npy')
A = np.load('./simulation/A.npy')


def draw_matrix(A):
    G = nx.DiGraph()
    D = A.shape[0]
    max_val = A.max()
    threshold = max_val * (1 - 1 / (2 * np.e))

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


plt.subplot(121)
draw_matrix(A0)
plt.subplot(122)
draw_matrix(A)
plt.show()
