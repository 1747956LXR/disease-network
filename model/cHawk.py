import numpy as np
import pandas as pd
# from scipy.integrate import quad
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
        D = len(self.diseases)  # disease number
        self.D = D

        self.f = dict()
        self.t = dict()
        self.d = dict()
        for i in self.patients:
            s = train_data[train_data["subject_id"] == i]

            self.f[i] = s[["age", "weight"]].values
            self.t[i] = s["age"].values
            self.d[i] = s["primary"].values

        # parameters initialization
        self.A = np.random.rand(D, D)  # subject to A >= 0
        self.u = np.random.rand(D, feature_num)  # subject to u_d >= 0

        self.vis = np.zeros_like(self.A)

        # visualization
        self.loss_draw = []

    # @profile
    def intensity(self, t, i, d):
        j = np.searchsorted(self.t[i], t)
        return self.u[d] @ self.f[i][j - 1] + np.sum(
            self.A[d][self.d[i][:j]] * g(t - self.t[i][:j]))

    # @profile
    def loss(self):
        res = 0

        # L1, L2 regularization
        res += L1 * np.sum(np.abs(self.A))
        res += 1 / 2 * L2 * np.sum(self.u * self.u)

        # log-likelihood
        log_likelihood = 0
        for i in self.patients:
            T = self.t[i][-1]  #

            for d in self.diseases:
                disease_d = (self.d[i] == d)

                if disease_d.any():
                    tijs = self.t[i][disease_d]
                    log_likelihood += sum(
                        np.log(self.intensity(tij, i, d)) for tij in tijs)

                log_likelihood -= self.u[d] @ sum(
                    self.f[i][j] * (self.t[i][j + 1] - self.t[i][j])
                    for j in range(len(self.t[i]) - 1))
                log_likelihood -= self.u[d] @ \
                    self.f[i][-1] * (T - self.t[i][-1])

                log_likelihood -= np.sum(
                    G(T - self.t[i]) * self.A[d][self.d[i]])

        res -= log_likelihood

        return res

    # @profile
    def grad(self):
        grad_A = np.zeros_like(self.A)
        grad_u = np.zeros_like(self.u)

        # grad_A
        for d in range(self.D):
            for dk in range(self.D):

                gradient = 0
                for i in self.patients:
                    disease_d = (self.d[i] == d)
                    if disease_d.any():
                        tijs = self.t[i][disease_d]
                        tiks = self.t[i][self.d[i] == dk]

                        T = self.t[i][-1]  #

                        for tij in tijs:
                            intensity_ij = self.intensity(tij, i, d)
                            for tik in tiks:
                                if tik < tij:
                                    self.vis[d][dk] = 1
                                    gradient += g(tij - tik) / intensity_ij

                                gradient -= G(T - tik)  #

                gradient = -gradient
                gradient += L1 * np.sign(grad_A[d][dk])
                grad_A[d][dk] = gradient

        # grad_u
        for d in range(self.D):
            gradient = 0
            for i in self.patients:
                disease_d = (self.d[i] == d)
                if disease_d.any():
                    gradient += sum(
                        self.f[i][disease_d][k] /
                        self.intensity(self.t[i][disease_d][k], i, d)
                        for k in range(np.sum(disease_d)))

                tijs = self.t[i]
                fijs = self.f[i]
                T = self.t[i][-1]  #

                gradient -= sum(fijs[j] * (tijs[j + 1] - tijs[j])
                                for j in range(len(tijs) - 1))

            gradient = -gradient
            gradient += L2 * self.u[d]
            grad_u[d] = gradient

        return grad_A, grad_u

    def project(self, val=0):
        self.A[self.A < 0] = val
        self.u[self.u < 0] = val

    def update(self, lr=1e-5): # TODO: momentum
        grad_A, grad_u = self.grad()
        self.A -= lr * grad_A
        self.u -= lr * grad_u
        self.project()

    def optimize(self, e=1e-3):
        old_loss = self.loss()
        self.update()

        i = 0
        while np.abs(self.loss() - old_loss) > e:
            print(old_loss, '\t', i)
            self.loss_draw.append(old_loss)
            old_loss = self.loss()

            self.update()

            i += 1
            if i > 1000:
                break

        self.A[self.vis == 0] = 0

    def save(self, file_A='./model/A.npy', file_u='./model/u.npy', file_loss='./model/loss.npy'):
        np.save(file_A, self.A)
        np.save(file_u, self.u)
        np.save(file_loss,np.array(self.loss_draw))

    def load(self, file_A='./model/A.npy', file_u='./model/u.npy', file_loss='./model/loss.npy'):
        self.A = np.load(file_A)
        self.u = np.load(file_u)
        self.loss_draw = list(np.load(file_loss))

    def draw(self):
        plt.plot(np.array(self.loss_draw))
        plt.show()


if __name__ == '__main__':
    data_path = './data/train_data.csv'
    train_data = pd.DataFrame(pd.read_csv(data_path))
    print(train_data)

    model = cHawk(train_data)
    # print(model.intensity(i=41976, d=6, t=64))
    # print(model.grad())

    model.optimize()
    print(model.grad())
    model.draw()
    model.save()
