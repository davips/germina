from pandas import DataFrame


def join(df: DataFrame, index, other, join):
    if df.index.name != index:
        df.set_index(index, inplace=True)
    if other.index.name != index:
        other.set_index(index, inplace=True)
    res = df.join([other], how=join)
    return res
