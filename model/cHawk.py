import numpy as np
import pandas as pd
from scipy.integrate import quad
import matplotlib.pyplot as plt

# Hyperparameters:
L1 = 10.0  # L1 regularization
L2 = 10.0  # L2 regularization
L = 0.2  # decay kernel parameter

feature_num = 2  # age, weight


# Helper functions
# decay kernel
def g(t):
    return L * np.exp(-L * t)


def G(t):
    return 1 - np.exp(-L * t)


class cHawk:
    def __init__(self, train_data):
        # load train_data
        self.patients = set(train_data["subject_id"])
        self.diseases = set(train_data["primary"])

        self.f = dict()
        self.t = dict()
        self.d = dict()
        for i in self.patients:
            s = train_data[train_data["subject_id"] == i]

            self.f[i] = s[["age", "weight"]].values
            self.t[i] = s["age"].values
            self.d[i] = s["primary"].values

        # parameters initialization
        D = len(self.diseases)  # disease number
        self.D = D
        self.A = np.random.rand(D, D)  # subject to A >= 0
        self.u = np.random.rand(D, feature_num)  # subject to u_d >= 0

        # visualization
        self.loss_draw = []

    def intensity(self, t, i, d):
        j = len(self.t[i] < t)
        return self.u[d] @ self.f[i][j - 1] + sum(
            self.A[d][self.d[i][:j]] * g(t - self.t[i][:j]))

    def loss(self):
        res = 0

        # L1, L2 regularization
        res += L1 * np.sum(self.A)  # not np.linalg.norm(self.A, 1)
        res += 1 / 2 * L2 * sum(
            np.linalg.norm(self.u[d], 2)**2
            for d in range(self.D))  # todo: vectorization

        # log-likelihood
        log_likelihood = 0
        # for all patients
        for i in self.patients:
            # for all diseases of a particular patient
            for d in self.diseases:
                # for every time the patient gets this disease
                tijs = self.t[i][self.d[i] == d]
                T = tijs[-1] + 1 if len(tijs) else 0
                if len(tijs) != 0:
                    log_likelihood += sum(
                        np.log(self.intensity(tij, i, d)) for tij in tijs)
                    log_likelihood -= quad(self.intensity,
                                           0,
                                           T,
                                           args=(i, d))[0]

        res -= log_likelihood

        return res

    def grad(self):
        grad_A = np.zeros_like(self.A)
        grad_u = np.zeros_like(self.u)

        # grad_A
        for d in range(self.D):
            for dk in range(self.D):
                gradient = 0
                for i in self.patients:
                    tijs = self.t[i][self.d[i] == d]
                    T = tijs[-1] + 1 if len(tijs) else 0
                    tiks = self.t[i][self.d[i] == dk]

                    for tij in tijs:
                        intensity_ij = self.intensity(tij, i, d)
                        for tik in tiks:
                            if tik < tij:
                                gradient += g(tij - tik) / intensity_ij

                        gradient -= G(T - tij)

                gradient = -gradient
                gradient += L1
                grad_A[d][dk] = gradient

        # grad_u
        for d in range(self.D):
            gradient = 0
            for i in self.patients:
                tijs = self.t[i][self.d[i] == d]
                T = tijs[-1] + 1 if len(tijs) else 0
                fijs = self.f[i][self.d[i] == d]

                for k in range(len(tijs)):
                    tij = tijs[k]
                    intensity_ij = self.intensity(tij, i, d)

                    tij_1 = tijs[k - 1] if k != 0 else 0
                    fij = fijs[k]

                    gradient += fij / intensity_ij
                    gradient -= self.u[d] @ fij * (tij - tij_1)

                gradient -= self.u[d] @ fijs[-1] * (T - tijs[-1]) if len(tijs) else 0

            gradient = -gradient
            gradient += L2 * self.u[d]
            grad_u = gradient

        return grad_A, grad_u

    def project(self, val=1e-3): # ?
        self.A[self.A < 0] = val
        self.u[self.u < 0] = val

    def update(self, lr=1e-4):
        grad_A, grad_u = self.grad()
        self.A -= lr * grad_A
        self.u -= lr * grad_u

        self.project()

    def optimize(self, e=1e-6):
        old_loss = self.loss()
        self.update()

        while np.abs(self.loss() - old_loss) > e:
            print(old_loss)
            self.loss_draw.append(old_loss)

            old_loss = self.loss()
            self.update()

    def save(self):
        np.save('./model/A.npy', self.A)
        np.save('./model/u.npy', self.u)

    def load(self):
        np.load('./model/A.npy', self.A)
        np.load('./model/u.npy', self.u)

    def draw(self):
        plt.plot(np.array(self.loss_draw))
        plt.show()


if __name__ == '__main__':
    data_path = './data/train_data.csv'
    train_data = pd.DataFrame(pd.read_csv(data_path))
    print(train_data)

    model = cHawk(train_data)
    # print(model.A.shape)
    # print(model.loss())
    # print(model.intensity(i=41976, d=6, t=64))
    # print(model.grad())

    model.optimize()
    model.draw()
    model.save()
