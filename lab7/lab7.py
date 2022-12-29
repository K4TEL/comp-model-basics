import numpy as np
import matplotlib.pyplot as plt
import itertools

n = 16

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


def reg_crit(Y, Y_reg):
    return np.sum(np.square(np.subtract(Y, Y_reg))) / np.sum(np.square(Y))


def ms_crit(Yt, Ya, Yb):
    yb = np.zeros_like(Ya)
    yb[:len(Yb)] = Yb
    return 3 * np.sum(np.square(np.subtract(Ya, yb))) / np.sum(np.square(Yt))


def data_split(X, Y, test_step):
    X_test, y_test = X[::test_step], Y[::test_step]
    X_train, y_train = np.delete(X, slice(None, None, test_step)), np.delete(Y, slice(None, None, test_step))
    print(f"TRAIN SIZE {len(X_train)}\n\tX train {X_train}\n\tY train {y_train}")
    print(f"TEST SIZE {len(X_test)}\n\tX test {X_test}\n\tY test {y_test}")
    return X_train, X_test, y_train, y_test


def print_regression(coefs, c):
    formula = "Y = "
    for i in range(c):
        if coefs[i] != 0:
            formula += f"{round(coefs[i], 2)}/X^{i} + "

    formula = formula[:-2]
    print(formula)


def get_model(X, Y):
    split_r_crits, split_ms_crits, split_models = {}, {}, {}
    for test_step in range(2, 7):
        X_train, X_test, y_train, y_test = data_split(X, Y, test_step)
        t = len(X_train)

        C = solve_sle(X_train, y_train, t)
        print(f"Unmasked model coefs: {C}")
        print_regression(C, t)

        coef_mask = np.array(list(map(list, itertools.product([0, 1], repeat=t))))
        coef_mask = coef_mask[(coef_mask == 1).sum(axis=1) > 1]
        coef_mask = np.flip(coef_mask[np.argsort(coef_mask.sum(axis=1))], 0)
        print(f"Total mask iterations to check: {coef_mask.shape[0]}")

        r_crits, ms_crits, models = {}, {}, {}
        for model in range(coef_mask.shape[0]):
            C_masked = np.where(coef_mask[model], C, 0)
            Y_reg = func(X_test, C_masked)
            R = reg_crit(y_test, Y_reg)

            if len(r_crits.values()) > 1 and min(r_crits.values()) > R:
                print(f"R {model}\tmask {coef_mask[model]} - crit {R}")

            MS = ms_crit(Y, func(X_train, C_masked), Y_reg)
            if len(ms_crits.values()) > 1 and min(ms_crits.values()) > MS:
                print(f"MS {model}\tmask {coef_mask[model]} - crit {MS}")

            r_crits[model], ms_crits[model], models[model] = R, MS, C_masked

        best_ind_r, best_ind_ms = min(r_crits, key=r_crits.get), min(ms_crits, key=ms_crits.get)
        print(f"BEST MODEL\nR\tmask {coef_mask[best_ind_r]} - crit {r_crits[best_ind_r]}"
              f"\nMS\tmask {coef_mask[best_ind_ms]} - crit {ms_crits[best_ind_ms]}")
        split_r_crits[test_step], split_ms_crits[test_step] = r_crits[best_ind_r], ms_crits[best_ind_ms]
        split_models[test_step] = [models[best_ind_r], models[best_ind_ms]]

    best_model_ind_r, best_model_ind_ms = min(split_r_crits, key=split_r_crits.get), min(split_ms_crits, key=split_ms_crits.get)
    Mr, Mms = split_models[best_model_ind_r][0], split_models[best_model_ind_ms][1]
    print(f"BEST\nR test step {best_model_ind_r}\tmodel{Mr} - crit {split_r_crits[best_model_ind_r]}"
          f"\nMS test step {best_model_ind_ms}\tmodel{Mms} - crit {split_ms_crits[best_model_ind_ms]}")
    return best_model_ind_ms, Mms


s, m = get_model(X0, Y0)
X_train, X_test, y_train, y_test = data_split(X0, Y0, s)
t = len(X_train)
print("Final (the best) regression model: ")
print_regression(m, t)
y_reg = func(X_test, m)
r = reg_crit(y_test, y_reg)
ms = ms_crit(Y0, func(X_train, m), y_reg)
print(f"Regression crit: {r}")
print(f"Minimal shift crit: {ms}")

plt.plot(X0, func(X0, m), label=f"Approx")
plt.plot(X0, Y0, label="Original")
plt.xlabel('X')
plt.ylabel('Y')
plt.title(f'Function graphs')
plt.legend()
plt.show()
