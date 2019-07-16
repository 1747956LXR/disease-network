import numpy as np
import pandas as pd

# Hyperparameters:
L1 = 10.0  # L1 regularization
L2 = 10.0  # L2 regularization
L = 0.2  # decay kernel parameter

feature_num = 2  # age, weight


# Helper functions
# decay kernel
def g(t):
    return L * np.exp(-L * t)


class cHawk:
    def __init__(self, train_data):
        # load train_data
        D = len(set(train_data["primary"]))  # disease number
        self.D = D

        # parameters initialization
        self.A = np.random.rand(D, D)
        self.u = np.random.rand(D, feature_num)

    def intensity(self, i, d, t): # ?
        pass

    def loss(self):
        res = 0

        # L1, L2 regularization
        res += L1 * np.linalg.norm(self.A, 1)
        res += 1 / 2 * L2 * sum(
            np.linalg.norm(self.u[d], 2)**2 for d in range(self.D))

        # log-likelihood

        return res 

    def grad(self):
        grad_A = np.zeros_like(self.A)
        grad_u = np.zeros_like(self.u)

        ###

        return grad_A, grad_u

    def update(self, lr=0.001):
        grad_A, grad_u = self.grad()
        self.A -= lr * grad_A
        self.u -= lr * grad_u

    def optimize(self, e=1e-6):
        old_loss = self.loss()
        self.update()
        while np.abs(self.loss() - old_loss) > e:
            old_loss = self.loss()
            self.update()

    def save_model(self):
        np.save('A.npy', self.A)
        np.save('u.npy', self.u)

    def load_model(self):
        np.load('A.npy', self.A)
        np.load('u.npy', self.u)


if __name__ == '__main__':
    data_path = './data/train_data.csv'
    train_data = pd.DataFrame(pd.read_csv(data_path))
    print(train_data)

    model = cHawk(train_data)
    print(model.A.shape)
    print(model.loss())