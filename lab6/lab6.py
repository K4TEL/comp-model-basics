import numpy as np
from sklearn.preprocessing import minmax_scale
import pandas as pd

x1_min, x1_max = 10/6, 2.0
x2_min, x2_max = 0.2, 0.7

n = 4  # кількість точок
c = 4  # кількість коефіцієнтів
m = 3  # кількість випробувань

norm_factors = np.array([[1, 1],
                         [1, -1],
                         [-1, 1],
                         [-1, -1]])

# number of generated entities = [500, 1000, 1400]
# Y - avg count of entities in Queue1
y_values = np.array([[0.222, 0.614, 0.722],
                     [0.0, 0.0, 0.0],
                     [0.536, 2.153, 2.966],
                     [0, 0.371, 0.548]])

factor_cols = ["X0", "X1", "X2", "X1*X2"]
val_cols = ["Y1", "Y2", "Y3"]

# інтеракції між факторами
def interact_factors(factors):
    full_factors = np.zeros((n, c))
    full_factors[:, 0] = 1
    full_factors[:, 1:-1] = factors
    full_factors[:, -1] = factors[:, 0] * factors[:, 1]
    return full_factors


# отримання натуральних факторів з нормованих
def natural(norm_factors):
    natural = np.zeros_like(norm_factors, dtype=float)
    natural[:, 0][norm_factors[:, 0] == 1] = x1_max
    natural[:, 1][norm_factors[:, 1] == 1] = x2_max
    natural[:, 0][norm_factors[:, 0] == -1] = x1_min
    natural[:, 1][norm_factors[:, 1] == -1] = x2_min
    return natural


# регресія за факторами та коефцієнтами
def regression(matrix, coefs, n, c):
    def func(factors, coefs, c):
        y = coefs[0]
        for i in range(c):
            y += coefs[i+1] * factors[i+1]
        return y
    values = np.zeros(n)
    for f in range(n):
        values[f] = func(matrix[f], coefs, c)
    return values


# коефіцієнти з вирішення СЛР
def solve_coef(factors, values):
    A = np.zeros((c, c))
    x_mean = np.mean(factors, axis=0)
    for i in range(1, c):
        for j in range(1, c):
            A[j, i] = np.mean(factors[:, i] * factors[:, j])
    A[0, :] = x_mean
    A[:, 0] = x_mean
    B = np.mean(np.tile(values, (c, 1)).T * factors, axis=0)
    coef = np.linalg.solve(A, B)
    return coef


# коефіцієнти з нормалізованих факторів та значень
def solve_norm_coef(factors, values):
    C = np.zeros(c)
    for i in range(c):
        C[i] = np.sum(values * factors[:, i]) / n
    return C


def print_regression(coefs):
    formula = "Y = "
    for i in range(c):
        formula += f"{round(coefs[i], 2)}*X{i}"
        if i != c-1:
            formula += " + "
    print(formula)

print(f"X1 min: {x1_min}\tX1 max: {x1_max}")
print(f"X2 min: {x2_min}\tX2 max: {x2_max}")

# формування факторів
factors = natural(norm_factors)
full_factors = interact_factors(factors)

# Перевірка однорідності дисперсії за критерієм Кохрена
# отримання оптимальної кількості випробувань для однієї комбінації факторів
Gt = [6.798, 5.157]  # N = 8, m = [2.. 3]
for i in range(len(Gt)):
    y_val = y_values[:, 1-i:]
    std_y = np.std(y_val, axis=1)**2
    print(f"Max Y dispersion: {std_y.max()}")
    Gp = std_y.max()/std_y.mean()
    print(f"m = {i+2} Gp: {Gp}\tGt: {Gt[i]}")
    if Gp < Gt[i]:
        m = i + 2  # кількість випробувань
        val_cols = val_cols[:m]
        f1, f2 = m-1, n
        print(f"Kohren criterion: {m} tries are enough")
        break

print(f"Mean Y dispersion: {std_y.mean()}")

# перевірка за критеріїм стюдента
coefs_value = np.zeros(4)
for i in range(c):
    coefs_value[i] = np.mean(y_val.mean(axis=1) * full_factors[:, i])
stud_crit = np.abs(coefs_value) / np.sqrt(std_y.mean()/(n*m))
ts = [2.306, 2.12]  # f3 = [8, 16]
sig_ind = np.argwhere(stud_crit > ts[m-2])
d = len(sig_ind.flatten())
print(f"All coefs are significant: {d == c}\t{sig_ind.flatten()}")

# формування планів
plan_cols = factor_cols + val_cols + ["Y_mean", "Y_disp"]
y_mean = y_val.mean(axis=1)

natural_plan = pd.DataFrame(columns=plan_cols)
natural_plan[factor_cols] = full_factors
natural_plan[val_cols] = y_val
natural_plan["Y_mean"] = y_mean.reshape(n, 1)
natural_plan["Y_disp"] = std_y

norm_y = minmax_scale(y_val.reshape(n*m), feature_range=(-1,1)).reshape(n ,m)
norm_factors_full = interact_factors(norm_factors)
y_norm_mean = norm_y.mean(axis=1)
std_norm_y = np.std(norm_y, axis=1)**2

norm_plan = pd.DataFrame(columns=plan_cols)
norm_plan[factor_cols] = norm_factors_full
norm_plan[val_cols] = norm_y
norm_plan["Y_mean"] = y_norm_mean.reshape(n, 1)
norm_plan["Y_disp"] = std_norm_y

# перевірка критерію Фішера
d = d-1 if d == n else d
Ft = 5.3

# отримання коефіцієнтів
coefs = solve_coef(full_factors, y_mean)
print("Natural coefs", coefs)
reg_val = regression(full_factors, coefs, n, m)
se = np.square(np.subtract(y_mean, reg_val))
print(f"MSE: {np.mean(se)}")

natural_plan["Y_reg"] = reg_val
print(natural_plan)
print_regression(coefs)

disp = m/(n - d) * np.sum(se)
Fp = disp / std_y.mean()
print(f"Natural Fisher crit: {Fp < Ft}\t{Fp}\t{Ft}")

# отримання нормалізованих коефіцієнтів
norm_coefs = solve_norm_coef(norm_factors_full, y_norm_mean)
print("Normalized coefs", norm_coefs)
reg_norm_val = regression(norm_factors_full, norm_coefs, n, m)
se_norm = np.square(np.subtract(y_norm_mean, reg_norm_val))
print(f"MSE: {np.mean(se_norm)}")

norm_plan["Y_reg"] = reg_norm_val
print(norm_plan)
print_regression(norm_coefs)

disp_norm = m/(n - d) * np.sum(se_norm)
Fp_norm = disp_norm / std_norm_y.mean()
print(f"Norm Fisher crit: {Fp_norm < Ft}\t{Fp_norm}\t{Ft}")
