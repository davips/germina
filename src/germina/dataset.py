from pandas import DataFrame


def join(df: DataFrame, index, other):
    if df.index.name != index:
        df.set_index(index, inplace=True)
    if other.index.name != index:
        other.set_index(index, inplace=True)
    res = df.join([other], how="outer")
    return res
