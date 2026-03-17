import numpy as np

def ben_functions(X, FunIndex, Dim):

    X = np.array(X)
    if FunIndex == 1:  # Sphere
        Fit = np.sum(X ** 2)
    elif FunIndex == 2:  # Schwefel 2.22
        Fit = np.sum(np.abs(X)) + np.prod(np.abs(X))
    elif FunIndex == 3:  # Schwefel 1.2
        Fit = 0
        for i in range(1, Dim + 1):
            Fit += np.sum(X[:i]) ** 2
    elif FunIndex == 4:  # Schwefel 2.21
        Fit = np.max(np.abs(X))
    elif FunIndex == 5:  # Rosenbrock
        Fit = np.sum(100 * (X[1:] - X[:-1] ** 2) ** 2 + (X[:-1] - 1) ** 2)
    elif FunIndex == 6:  # Step
        Fit = np.sum(np.floor(X + 0.5) ** 2)
    elif FunIndex == 7:  # Quartic with noise
        Fit = np.sum(np.arange(1, Dim + 1) * (X ** 4)) + np.random.rand()
    elif FunIndex == 8:  # Schwefel
        Fit = np.sum(-X * np.sin(np.sqrt(np.abs(X))))
    elif FunIndex == 9:  # Rastrigin
        Fit = np.sum(X ** 2 - 10 * np.cos(2 * np.pi * X)) + 10 * Dim
    elif FunIndex == 10:  # Ackley
        Fit = -20 * np.exp(-0.2 * np.sqrt(np.sum(X ** 2) / Dim)) - \
              np.exp(np.sum(np.cos(2 * np.pi * X)) / Dim) + 20 + np.exp(1)
    elif FunIndex == 11:  # Griewank
        i = np.arange(1, Dim + 1)
        Fit = np.sum(X ** 2) / 4000 - np.prod(np.cos(X / np.sqrt(i))) + 1
    elif FunIndex == 12:  # Penalized 1
        a = 10
        k = 100
        m = 4
        y = 1 + (X + 1) / 4
        term1 = (np.pi / Dim) * (10 * (np.sin(np.pi * y[0])) ** 2 +
                                  np.sum((y[:-1] - 1) ** 2 * (1 + 10 * (np.sin(np.pi * y[1:])) ** 2)) +
                                  (y[-1] - 1) ** 2)
        penalty = np.sum(k * ((X - a) ** m) * (X > a) + k * ((-X - a) ** m) * (X < -a))
        Fit = term1 + penalty
    elif FunIndex == 13:  # Penalized 2
        a = 10
        k = 100
        m = 4
        term1 = 0.1 * ((np.sin(3 * np.pi * X[0])) ** 2 +
                       np.sum((X[:-1] - 1) ** 2 * (1 + (np.sin(3 * np.pi * X[1:])) ** 2)) +
                       (X[-1] - 1) ** 2 * (1 + (np.sin(2 * np.pi * X[-1])) ** 2))
        penalty = np.sum(k * ((X - a) ** m) * (X > a) + k * ((-X - a) ** m) * (X < -a))
        Fit = term1 + penalty
    elif FunIndex == 14:  # Foxholes
        a = np.array([[-32, -16, 0, 16, 32, -32, -16, 0, 16, 32, -32, -16, 0, 16, 32, -32, -16, 0, 16, 32, -32, -16, 0, 16, 32],
                      [-32, -32, -32, -32, -32, -16, -16, -16, -16, -16, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 32, 32, 32, 32, 32]])
        b = np.zeros(25)
        for j in range(25):
            b[j] = np.sum((X - a[:, j]) ** 6)
        Fit = 1 / (1 / 500 + np.sum(1.0 / (np.arange(1, 26) + b)))
    elif FunIndex == 15:  # Kowalik
        a = np.array([0.1957, 0.1947, 0.1735, 0.16, 0.0844, 0.0627, 0.0456, 0.0342, 0.0323, 0.0235, 0.0246])
        b = 1.0 / np.array([0.25, 0.5, 1, 2, 4, 6, 8, 10, 12, 14, 16])
        numerator = X[0] * (b**2 + X[1] * b)
        denominator = b**2 + X[2] * b + X[3]
        Fit = np.sum((a - numerator / denominator) ** 2)
    elif FunIndex == 16:  # Six Hump Camel
        Fit = 4 * X[0] ** 2 - 2.1 * X[0] ** 4 + X[0] ** 6 / 3 + X[0] * X[1] - 4 * X[1] ** 2 + 4 * X[1] ** 4
    elif FunIndex == 17:  # Branin
        Fit = (X[1] - (5.1 / (4 * np.pi ** 2)) * X[0] ** 2 + (5 / np.pi) * X[0] - 6) ** 2 + \
              10 * (1 - 1 / (8 * np.pi)) * np.cos(X[0]) + 10
    elif FunIndex == 18:  # Goldstein-Price
        Fit = (1 + (X[0] + X[1] + 1) ** 2 * (19 - 14 * X[0] + 3 * X[0] ** 2 - 14 * X[1] + 6 * X[0] * X[1] + 3 * X[1] ** 2)) * \
              (30 + (2 * X[0] - 3 * X[1]) ** 2 * (18 - 32 * X[0] + 12 * X[0] ** 2 + 48 * X[1] - 36 * X[0] * X[1] + 27 * X[1] ** 2))
    elif FunIndex == 19:  # Hartman 3
        a = np.array([[3, 10, 30],
                      [0.1, 10, 35],
                      [3, 10, 30],
                      [0.1, 10, 35]])
        c = np.array([1, 1.2, 3, 3.2])
        p = np.array([[0.3689, 0.117, 0.2673],
                      [0.4699, 0.4387, 0.747],
                      [0.1091, 0.8732, 0.5547],
                      [0.03815, 0.5743, 0.8828]])
        Fit = 0
        for i in range(4):
            Fit -= c[i] * np.exp(-np.sum(a[i] * (X - p[i]) ** 2))
    elif FunIndex == 20:  # Hartman 6
        af = np.array([[10, 3, 17, 3.5, 1.7, 8],
                       [0.05, 10, 17, 0.1, 8, 14],
                       [3, 3.5, 1.7, 10, 17, 8],
                       [17, 8, 0.05, 10, 0.1, 14]])
        cf = np.array([1, 1.2, 3, 3.2])
        pf = np.array([[0.1312, 0.1696, 0.5569, 0.0124, 0.8283, 0.5886],
                       [0.2329, 0.4135, 0.8307, 0.3736, 0.1004, 0.9991],
                       [0.2348, 0.1415, 0.3522, 0.2883, 0.3047, 0.6650],
                       [0.4047, 0.8828, 0.8732, 0.5743, 0.1091, 0.0381]])
        Fit = 0
        for i in range(4):
            Fit -= cf[i] * np.exp(-np.sum(af[i] * (X - pf[i]) ** 2))
    elif FunIndex == 21:  # Shekel 5
        a = np.array([[4, 4, 4, 4],
                      [1, 1, 1, 1],
                      [8, 8, 8, 8],
                      [6, 6, 6, 6],
                      [3, 7, 3, 7],
                      [2, 9, 2, 9],
                      [5, 5, 3, 3],
                      [8, 1, 8, 1],
                      [6, 2, 6, 2],
                      [7, 3.6, 7, 3.6]])
        c = np.array([0.1, 0.2, 0.2, 0.4, 0.4, 0.6, 0.3, 0.7, 0.5, 0.5])
        Fit = 0
        for i in range(5):
            diff = X - a[i]
            Fit -= 1 / (np.dot(diff, diff) + c[i])
    elif FunIndex == 22:  # Shekel 7
        a = np.array([[4, 4, 4, 4],
                      [1, 1, 1, 1],
                      [8, 8, 8, 8],
                      [6, 6, 6, 6],
                      [3, 7, 3, 7],
                      [2, 9, 2, 9],
                      [5, 5, 3, 3],
                      [8, 1, 8, 1],
                      [6, 2, 6, 2],
                      [7, 3.6, 7, 3.6]])
        c = np.array([0.1, 0.2, 0.2, 0.4, 0.4, 0.6, 0.3, 0.7, 0.5, 0.5])
        Fit = 0
        for i in range(7):
            diff = X - a[i]
            Fit -= 1 / (np.dot(diff, diff) + c[i])
    else:  # Shekel 10
        a = np.array([[4, 4, 4, 4],
                      [1, 1, 1, 1],
                      [8, 8, 8, 8],
                      [6, 6, 6, 6],
                      [3, 7, 3, 7],
                      [2, 9, 2, 9],
                      [5, 5, 3, 3],
                      [8, 1, 8, 1],
                      [6, 2, 6, 2],
                      [7, 3.6, 7, 3.6]])
        c = np.array([0.1, 0.2, 0.2, 0.4, 0.4, 0.6, 0.3, 0.7, 0.5, 0.5])
        Fit = 0
        for i in range(10):
            diff = X - a[i]
            Fit -= 1 / (np.dot(diff, diff) + c[i])
    return Fit