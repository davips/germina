import numpy as np
from numpy import ndarray


def pairwise_sample(numbers, s, seed):
    """
    >>> import numpy as np
    >>> pairwise_sample(np.array([1,2,3]), 5, 0)
    array([[3, 2],
           [1, 2],
           [2, 3],
           [1, 3],
           [2, 1]])

    :param numbers:
    :param s:  # of pairs to sample
    :param seed:
    :return:
    """
    if not isinstance(numbers, ndarray):
        numbers = np.array(numbers)
    n = len(numbers)
    x = np.arange(0, n)
    y = np.arange(0, n)
    X, Y = np.meshgrid(x, y)
    res = np.vstack([X.ravel(), Y.ravel()]).T
    res = res[res[:, 0] != res[:, 1]]
    rnd = np.random.default_rng(seed)
    rnd.shuffle(res)
    return numbers[res[:s]]
