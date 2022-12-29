import numpy as np
import matplotlib.pyplot as plt

n = 16
m = [2, 4, 8, 16]

X0 = np.zeros(n)
for i in range(n):
    X0[i] = 1 + i * 0.5

Y0 = np.array([14, 18.222, 18, 17.216, 16.444, 15.778, 15.219, 14.749,
              14.352, 14.014, 13.722, 13.469, 13.248, 13.052, 12.879, 12.724])


def func(X, B):
    Y = np.zeros_like(X)
    for n in range(len(X)):
        Y[n] = B[0]
        for i in range(1, len(B)):
            Y[n] += B[i] / np.power(X[n], i)
    return Y


def solve_sle(X, Y, m):
    A = np.zeros((m, m))
    A[:, 0] = 1
    for k in range(1, m):
        A[:, k] = 1 / np.power(X, k)
    B = np.linalg.solve(A, Y)
    return B

best_b = 1
for b in m:
    step = int(n/b)
    X = X0[::step]
    Y = Y0[::step]
    B = solve_sle(X, Y, b)
    Y_aprox = func(X0, B)
    MSE = np.power(Y0 - Y_aprox, 2)

    table = np.stack((X0, Y0, Y_aprox, MSE), axis=1)
    print(f"Max MSE for {b} coefs: {MSE.max()}")
    if MSE.max() < best_b:
        best_b = b
        print(table)

    function = f"Y = {B[0]} "
    for i in range(1, b):
        function += f" + {B[i]}/x^{i}"
    print(function)

    plt.plot(X0, Y_aprox, label=f"Approx {b}")
plt.plot(X0, Y0, label="Original")
plt.xlabel('X')
plt.ylabel('Y')
plt.title(f'Function graphs')
plt.legend()
plt.show()

print(f"Best number of coefs: {best_b}")
