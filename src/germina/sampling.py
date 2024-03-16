from itertools import islice

import numpy as np
from numpy import ndarray
from sklearn.model_selection import ParameterSampler

from germina.aux import get_algspace


def pairwise_sample(X, n, seed):
    """
    >>> import numpy as np
    >>> pairwise_sample(np.array([1,2,3]), 5, 0)
    array([[3, 2],
           [1, 2],
           [2, 3],
           [1, 3],
           [2, 1]])

    :param X:
    :param n:  # of pairs to sample
    :param seed:
    :return:
    """
    if not isinstance(X, ndarray):
        X = np.array(X)
    N = len(X)
    x = np.arange(0, N)
    y = np.arange(0, N)
    X_, Y_ = np.meshgrid(x, y)
    res = np.vstack([X_.ravel(), Y_.ravel()]).T
    res = res[res[:, 0] != res[:, 1]]
    rnd = np.random.default_rng(seed)
    rnd.shuffle(res)
    return X[res[:n]]


def create_search_space(algname, n, start=None, end=None, seed=0, aslist=False):
    search_space = get_algspace(algname)
    sampler = islice(ParameterSampler(search_space, n, random_state=seed), start, end)
    return list(sampler) if aslist else sampler
