import numpy as np

def space_bound(X, Up, Low):

    X = np.array(X)
    Low = np.array(Low)
    Up = np.array(Up)
    mask = (X > Up) | (X < Low)
    if np.any(mask):
        rand_vals = np.random.rand(len(X)) * (Up - Low) + Low
        X = np.where(mask, rand_vals, X)
    return X