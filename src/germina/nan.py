from pprint import pprint

import numpy as np
import pandas as pd
from pandas import DataFrame
from pandas.core.generic import NDFrame


def hasNaN(df: DataFrame, debug=True):
    nans_hist: NDFrame = df.isna().sum()
    if debug:
        dct = dict(reversed(sorted(nans_hist.items(), key=lambda x: x[1])))
        print("Shape:", df.shape, "\tmin & NaNs:", min(nans_hist.to_numpy().tolist()), dct)
    nans_ = sum(nans_hist)
    return nans_


def remove_nan_rows_cols(df: DataFrame, keep, rows_at_a_time=1, cols_at_a_time=1, debug=True):
    rows, cols = 0, 0
    rows_old, cols_old = 1, 1
    if debug:
        print("Shape:", df.shape, end="\t→\t")
    while hasNaN(df, debug=False) and (rows_old != rows or cols_old != cols):
        rows_old, cols_old = rows, cols
        for i in range(rows_at_a_time):
            df = remove_worst_nan_rows(df, debug=False)
        for i in range(cols_at_a_time):
            df = remove_worst_nan_cols(df, keep=keep, debug=False)
        rows, cols = df.shape
    if debug:
        print(df.shape)
    return df


def remove_nan_cols_rows(df: DataFrame, keep, rows_at_a_time=1, cols_at_a_time=1, debug=True):
    rows, cols = 0, 0
    rows_old, cols_old = 1, 1
    if debug:
        print("Shape:", df.shape, end="\t→\t")
    while hasNaN(df, debug=False) and (rows_old != rows or cols_old != cols):
        rows_old, cols_old = rows, cols
        for i in range(cols_at_a_time):
            df = remove_worst_nan_cols(df, keep=keep, debug=False)
        for i in range(rows_at_a_time):
            df = remove_worst_nan_rows(df, debug=False)
        rows, cols = df.shape
    if debug:
        print(df.shape)
    return df


def remove_worst_nan_rows(df: DataFrame, debug=True):
    s = df.isna().sum(axis=1)
    df = df[s.ne(s.max()) | s.eq(0)]
    if debug:
        print("After removing worst rows:", df.shape)
    return df


def remove_worst_nan_cols(df: DataFrame, keep, debug=True):
    bkp = backup_cols(df, keep)
    s = df.isna().sum(axis=0)
    df = df.loc[:, s.ne(s.max()) | s.eq(0)]
    df = recover_cols(df, bkp)
    if debug:
        print("After removing worst columns:", df.shape)
    return df


def backup_cols(df: DataFrame, keep):
    return {c: df[c] for c in keep if "-" not in c and c in df}


def recover_cols(df: DataFrame, bkp):
    for tgt, col in bkp.items():
        if tgt not in list(df.columns):
            df = pd.concat((df, col), axis=1)
    return df


def remove_cols(df: DataFrame, cols, keep, debug=True):
    for attr in cols:
        if attr in df:
            if attr not in keep:
                df = df.drop(columns=[attr])
        elif attr.endswith("*"):
            for a in df.columns:
                if a.startswith(attr[:-1]) and a not in keep:
                    df = df.drop(columns=[attr])
    if debug:
        print("New shape:", df.shape)
    return df


def bina(df: DataFrame, attribute, positive_category):
    if attribute in df:
        df[attribute] = (df[attribute] == positive_category).astype(int)
    return df


def loga(df: DataFrame, attribute):
    if attribute in df:
        df[attribute] = np.log(df[attribute])
    return df


def isworking(df):
    raise NotImplemented
    """
    Precisa pedir essa variável
    
    b06_t1: Qual sua situação de trabalho atual	
    7	Emprego fixo, com carteira de trabalho assinada
    8	Emprego fixo, sem carteira assinada
    9	Emprego temporário, com carteira de trabalho assinada
    10	Emprego sem contrato, sem carteira assinada
    11	Desempregada
    12	Autônoma
    13	Trabalhador Familiar
    14	Não trabalha
    15	Estudante
    b07_t1: Sobre sua situação de trabalho atual:	
    1	Você está em licença maternidade
    2	Esteve mas já interrompeu
    3	Não tem direito a licença maternidade mas não está trabalhando
    4	Não tem direito a licença maternidade e está trabalhando
    """
    return df


def only_abundant(df, threshold=10):
    """
    >>> df = DataFrame([[1,2,0],[1,0,0],[1,2,3],[1,0,0]])
    >>> only_abundant(df, 1).to_numpy()
    """
    return df.loc[:, np.count_nonzero(df, axis=0) > threshold]
