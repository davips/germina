from pandas import DataFrame
from scipy.stats import mode


def join(df: DataFrame, index, other, join):
    if df.index.name != index:
        df.set_index(index, inplace=True)
    if other.index.name != index:
        other.set_index(index, inplace=True)
    res = df.join([other], how=join)
    return res


def ensemble_predict(*predictions):
    """
    >>> ensemble_predict([1,9,5],[1,2,4],[1,9,4],[1,0,4])
    array([1, 9, 4])
    """
    return mode(predictions)[0]
