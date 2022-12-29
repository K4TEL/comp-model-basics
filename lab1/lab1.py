import numpy as np
import math
import matplotlib.pyplot as plt

n = 10000
a = 5**13
c = 2**31
k = 20
start, end = 0, 1
Crit = 1.36

def generate(a, c, n):
    x = np.zeros(n)
    z = 1
    for i in range(n):
        x[i] = z/c
        z = a * z % c
    return x


for i in range(10):
    X = generate(a, c, n)
    X_mean = np.mean(X)
    X_std = np.std(X)
    h = (X.max() - X.min())/k
    p = h/(X.max() - X.min())

    print(f"n: {n}\ta: {a}\t c: {c}")
    print(f"Interval size {round(h, 4)} for {k} bins")
    print(f"Frequency for intervals: {p}")
    print(f"X mean: {round(X_mean, 4)}\ttheoretical: {(start+end)/2}")
    print(f"X std: {round(X_std, 4)}\ttheoretical: {round((end-start)/(2 * math.sqrt(3)), 4)}")

    Wmax = 0
    for i in range(k):
        crit = abs(np.count_nonzero((0 < X) & (X < (i+1)*h))/n - i * p)
        if crit > Wmax:
            Wmax = crit
    print(f"Criterion {round(Wmax, 4)} fit: {Wmax < Crit}")

    a = a/5
    c = c/2

plt.hist(X, bins=k, color="gold", edgecolor="black")
plt.ylabel('Count')
plt.xlabel('X')
plt.show()

plt.hist(X, density=True, bins=k, color="gold", edgecolor="black")
plt.ylabel('Frequency')
plt.xlabel('X')
plt.show()
