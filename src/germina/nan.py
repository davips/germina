import numpy as np
import pandas as pd


def hasNaN(df, debug=True):
    nans_hist = df.isna().sum()
    if debug:
        print("Removing NaNs...", df.shape, "\t\t\t\t\t", nans_hist.to_numpy().tolist())
    nans_ = sum(nans_hist)
    return nans_


def remove_nan_rows(df, debug=True):
    s = df.isna().sum(axis=1)
    df = df[s.ne(s.max()) | s.eq(0)]
    if debug:
        print("After removing worst rows:", df.shape)
    return df


def remove_nan_cols(df, keep, debug=True):
    bkp = backup_cols(df, keep)
    s = df.isna().sum(axis=0)
    df = df.loc[:, s.ne(s.max()) | s.eq(0)]
    df = recover_cols(df, bkp)
    if debug:
        print("After removing worst columns:", df.shape)
    return df


def backup_cols(df, targets):
    return {tgt: df[tgt] for tgt in targets if "-" not in tgt}


def recover_cols(df, bkp):
    for tgt, col in bkp.items():
        if tgt not in list(df.columns):
            df = pd.concat((df, col), axis=1)
    return df


def remove_cols(df, cols, keep, debug=True):
    for attr in cols:
        if attr in df:
            if attr not in keep:
                del df[attr]
        elif attr.endswith("*"):
            for a in df.columns:
                if a.startswith(attr[:-1]) and a not in keep:
                    del df[a]
    if debug:
        print("New shape:", df.shape)
    return df


def bina(df, attribute, positive_category):
    df[attribute] = (df[attribute] == positive_category).astype(int)
    return df


def loga(df, attribute):
    df[attribute] = np.log(df[attribute])
    return df
