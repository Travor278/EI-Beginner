import numpy as np
def computeCost(X,t,theta):
    inner = np.power(((X * theta.T) - t), 2)
    return np.sum(inner) / (2 * len(X))