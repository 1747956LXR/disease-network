import numpy as np
import matplotlib.pyplot as plt
# from model.cHawk import cHawk

L = 0.2  # decay kernel parameter


# decay kernel
def g(t):
    if t > 0:
        return L * np.exp(-L * t)
    else:
        return 0


g = np.vectorize(g)

A = np.load("./model/A.npy")
u = np.load("./model/u.npy")

ts = np.array([])
ds = np.array([])
f = np.array([60, 60])  # age, weight


def intensity(t, d):
    j = np.searchsorted(ts, t)
    if j == 0:
        return u[d] @ f
    return u[d] @ f + np.sum(A[d][ds[:j].astype(np.int32)] * g(t - ts[:j]))


def generate():
    global ts, ds, f
    for t in np.arange(60, 68, 0.01):
        for d in range(len(u)):
            if intensity(t, d) * 0.01 > np.random.rand():
                print(t, d)
                ts = np.append(ts, t)
                ds = np.append(ds, d)
                # f = np.array([t, f[1] + 2 * np.random.rand() - 1])


generate()

print(ts)
print(ds)


def intensity_drawing(d):
    start = 60
    end = 68

    time = np.arange(start - 1, end + 10, 0.01)
    val = []
    for t in time:
        val.append(intensity(t=t, d=d))
    plt.plot(np.array(val))
    plt.show()


intensity_drawing(1)

np.save('./test/ts.npy', ts)
np.save('./test/ds.npy', ds)
np.save('./test/f.npy', f)