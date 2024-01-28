import numpy as np


def pairwise_diff(A, B):
    """
    >>> from numpy.random import default_rng
    >>> rnd = default_rng(0)
    >>> X = rnd.random(size=(3, 2))
    >>> X
    array([[0.63696169, 0.26978671],
           [0.04097352, 0.01652764],
           [0.81327024, 0.91275558]])
    >>> pairwise_diff(X[:1,:], X)
    array([[ 0.63696169,  0.04097352,  0.25325908],
           [ 0.63696169,  0.81327024, -0.64296886],
           [ 0.04097352,  0.63696169, -0.25325908],
           [ 0.04097352,  0.81327024, -0.89622794],
           [ 0.81327024,  0.63696169,  0.64296886],
           [ 0.81327024,  0.04097352,  0.89622794]])
    """
    A = A[:, np.newaxis, :]
    B = B[np.newaxis, :, :]
    D = A - B
    D = D.reshape(-1, D.shape[2])
    return D


def pairwise_hstack(A, B, handle_last_as_y=False):
    """
    >>> from numpy.random import default_rng
    >>> rnd = default_rng(0)
    >>> X = rnd.random(size=(3, 2))
    >>> X
    array([[0.63696169, 0.26978671],
           [0.04097352, 0.01652764],
           [0.81327024, 0.91275558]])
    >>> pairwise_hstack(X, X)
    array([[ 0.63696169,  0.04097352,  0.25325908],
           [ 0.63696169,  0.81327024, -0.64296886],
           [ 0.04097352,  0.63696169, -0.25325908],
           [ 0.04097352,  0.81327024, -0.89622794],
           [ 0.81327024,  0.63696169,  0.64296886],
           [ 0.81327024,  0.04097352,  0.89622794]])
    """
    tA = np.tile(A[:, np.newaxis, :], [B.shape[0], 1]).reshape(-1, A.shape[1])
    tB = np.tile(B[:, :], [A.shape[0], 1])
    if handle_last_as_y:
        D = tA[:, -1:] - tB[:, -1:]
        return np.hstack((tA[:, :-1], tB[:, :-1], D))
    else:
        return np.hstack((tA, tB))
