import numpy as np
import pandas as pd
import numba as nb

# Hyperparameters:
L1 = 10.0  # L1 regularization
L2 = 10.0  # L2 regularization
L = 0.2  # decay kernel parameter


# Helper functions
# decay kernel
def g(t):
    return L * np.exp(-L * t)


# [begin, end)
def rand_range(begin, end):
    return begin + (end - begin) * np.random.rand()


class data_generator:
    def __init__(self, patients_num, diseases_num):
        # init
        self.P = patients_num
        self.D = diseases_num
        self.features_num = 2
        self.init_Au()

        # generate
        self.generate_data()

        # concat
        self.dict_to_df()

        # save data
        self.df.to_csv("./simulation/generated_data.csv", index=None)

    def init_Au(self):
        self.A = np.random.rand(self.D, self.D) / 5
        self.random_set_zero(self.A, 0.8)
        self.u = np.random.rand(self.D, self.features_num) / 1000

        np.save('./simulation/A0.npy', self.A)
        np.save('./simulation/u0.npy', self.u)

    def random_set_zero(self, mat: np.array, zero_rate=0.9):
        l, w = mat.shape
        for i in range(l):
            for j in range(w):
                if np.random.rand() < zero_rate:
                    mat[i][j] = 0

    def intensity(self, t, i, d):
        # find j
        j = 1
        for idx, (tij, _, _) in enumerate(self.events[i]):
            if tij > t:
                j = idx
                break

        fij = np.array([self.events[i][j - 1][0], self.events[i][j - 1][2]])
        res = 0
        res += self.u[d] @ fij

        for k in range(j):
            tk, dk, _ = self.events[i][k]
            res += self.A[d][dk] * g(t - tk)

        return res

    def generate_data(self):
        # generate {(tij, dij, fij)} for j=1:n_i, for i=1:patients_num
        self.events = dict()

        dt = 0.1
        for i in range(self.P):
            print("generating patient", i)
            # start \in [40, 60), end \in [60, 80)
            start = rand_range(40, 60)
            start_disease = np.random.randint(0, self.D)
            start_weight = rand_range(40, 90)
            last_weight = start_weight

            # end = rand_range(60, 80)
            end = start + rand_range(0, 1)

            self.events[i] = list()
            self.events[i].append((start, start_disease, start_weight))

            for t in np.arange(start, end, dt):
                weight = last_weight + rand_range(-0.5, 0.5)

                ds = []
                for d in range(self.D):
                    if np.random.rand() < self.intensity(t, i, d) * dt:
                        ds.append(d)
                if not ds:
                    continue

                choice_d = ds[np.random.randint(0, len(ds))]

                self.events[i].append((t, choice_d, weight))

                last_weight = weight

    def dict_to_df(self):
        # self.events -> dataframe
        self.df = pd.DataFrame(
            columns=["subject_id", "age", "primary", "weight"])
        for i in range(self.P):
            print("write dataframe", i)
            for tij, dij, wij in self.events[i]:
                if len(self.events[i]) < 2:
                    break
                self.df = self.df.append(pd.DataFrame({
                    'subject_id': [i],
                    'age': [tij],
                    'primary': [dij],
                    'weight': [wij]
                }),
                                         ignore_index=True)


if __name__ == '__main__':
    data = data_generator(200, 30)
    # print(data.events)
    print(data.df)