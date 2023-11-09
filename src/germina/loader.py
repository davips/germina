import dalex as dx
import xgboost as xgb
from datetime import datetime
from pprint import pprint

import numpy as np
import pandas as pd
from sklearn import clone
from sklearn.impute import IterativeImputer

from germina.dataset import join
from hdict import apply, _
from hdict.dataset.pandas_handling import file2df
from pandas import DataFrame
from sklearn.preprocessing import StandardScaler

from germina.runner import drop_many_by_vif, ch, sgid2estudoid, setindex


def load_from_csv(d, storages, storage_to_be_updated, path, vif, filename, field, transpose, old_indexname="id_estudo", vars=None, verbose=True):
    if verbose:
        print(datetime.now())
    d = d >> apply(file2df, path + filename + ".csv", transpose=transpose, index=True)(field)
    d = ch(d, storages, storage_to_be_updated)
    if verbose:
        print(f"Loaded '{field}' data from '{filename}.csv' ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓", d[field].shape, "↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑")
    if vars:
        d = d >> apply(lambda df, vs: df[vs], _[field], vars)(field)
        d = ch(d, storages, storage_to_be_updated)
        if verbose:
            print(f"Selected '{len(vars)} {vars}' ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓", d[field].shape, "↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑")
    d = d >> apply(setindex, _[field], old_indexname=old_indexname)(field)
    d = ch(d, storages, storage_to_be_updated)
    if vif:
        if verbose:
            print(f"Apply VIF to '{field}' ----------------------------------------------------------------------------------------------------------------------------")
        if verbose:
            print(datetime.now())
        d = drop_many_by_vif(d, field, storages, storage_to_be_updated, [], [])
        d = ch(d, storages, storage_to_be_updated)
        if verbose:
            print("after VIF ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓", d[field].shape, "↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑")
    if verbose:
        print()
    return d


def load_from_synapse(d, storages, storage_to_be_updated, path, vif, filename, field):
    print(datetime.now())
    d = d >> apply(file2df, path + filename + ".csv")(field)
    d = ch(d, storages, storage_to_be_updated)
    print(f"Loaded '{field}' Synapse data from '{filename}.csv' ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓", d[field].shape, "↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑")
    d = d >> apply(sgid2estudoid, _[field])(field)
    d = ch(d, storages, storage_to_be_updated)
    print("Fixed id ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓\n", d[field], "↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑")
    if vif:
        print(f"Apply VIF to '{field}' ----------------------------------------------------------------------------------------------------------------------------")
        print(datetime.now())
        d = drop_many_by_vif(d, field, storages, storage_to_be_updated, [], [])
        d = ch(d, storages, storage_to_be_updated)
        print("after VIF ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓", d[field].shape, "↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑")
    print()
    d = d >> apply(setindex, _[field])(field)
    d = ch(d, storages, storage_to_be_updated)
    return d


def load_from_osf(d, storages, storage_to_be_updated, path, vif, filename, vars__no_t, field, keeprows):
    print(datetime.now())
    if field == "fullosf":
        d = d >> apply(file2df, path + filename + ".csv")(field)
    else:
        if "fullosf" not in d:
            raise Exception(f"Load 'fullosf' from csv first.")
        vars = []
        for v in sorted(vars__no_t):
            for i in range(7):
                t_candidate = f"{v}_t{i}"
                if t_candidate in d.fullosf:
                    vars.append(t_candidate)
        vars.sort()
        d = d >> apply(lambda fullosf, vs: fullosf[vs], vs=vars)(field)
    if vif:
        print(f"Apply VIF to '{field}' ----------------------------------------------------------------------------------------------------------------------------")
        print(datetime.now())
        d = drop_many_by_vif(d, field, storages, storage_to_be_updated, [], keeprows)
        d = ch(d, storages, storage_to_be_updated)
        print("after VIF ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓", d[field].shape, "↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑")
    d = d >> apply(setindex, _[field])(field)
    d = ch(d, storages, storage_to_be_updated)
    print(f"Loaded '{field}' OSF data from '{filename}.csv' ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓", d[field].shape, "↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑\n")
    return d


def apply_std(d, storages, storage_to_be_updated, path, vif, field):
    print(datetime.now())
    d = d >> apply(lambda x: DataFrame(StandardScaler().fit_transform(x)), _[field])(field)
    d = d >> apply(setindex, _[field])(field)
    d = ch(d, storages, storage_to_be_updated)
    print("Scaled ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓", d[field].shape, "↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑\n")
    return d


def clean_for_dalex(d, storages, storage_to_be_updated):
    print(datetime.now())
    d = d >> apply(lambda df: df.drop(["EBF_3m"], axis=1)).X
    d = d >> apply(lambda X: [col.replace("[", "").replace("]", "").replace("<", "").replace(" ", "_") for col in X.columns]).Xcols
    d = d >> apply(lambda X, Xcols: X.rename(columns=dict(zip(X.columns, Xcols)))).X
    # d = d >> apply(lambda X: pd.get_dummies(X["delivery_mode"])["vaginal"].astype(int)).delivery_mode
    # d = d >> apply(join, df=_.X, other=_.delivery_mode).X
    d = d >> apply(lambda df: pd.get_dummies(df["EBF_3m"])["EBF"].astype(int)).y
    d = ch(d, storages, storage_to_be_updated)
    print("Scaled ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓", d.X.shape, d.y.shape, "↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑\n")
    return d


def train_xgb(params, X, y, idxtr):
    print(datetime.now(), "train_xgb")
    return xgb.train(params, xgb.DMatrix(X.iloc[idxtr], label=y.iloc[idxtr]))


def get_balance(d, storages, storage_to_be_updated):
    d = d >> apply(lambda X: X.shape).Xshape
    d = d >> apply(lambda y: y.shape).yshape
    d = d >> apply(lambda y: np.unique(y, return_counts=True))("unique_labels", "counts")
    d = d >> apply(lambda y, counts: counts / len(y)).proportions
    d = ch(d, storages, storage_to_be_updated)
    print(datetime.now(), "X, y:", d.Xshape, d.yshape, f"{d.counts=}\t{d.proportions=}")
    return d


def build_explainer(classifier, X, y, idxtr):
    print(datetime.now(), "build_explainer")
    return dx.Explainer(classifier, X.iloc[idxtr], y.iloc[idxtr])


def explain_modelparts(explainer):
    print(datetime.now(), "explain_modelparts")
    return explainer.model_parts()  # .plot(show=False).show()


def explain_predictparts(explainer, X, idxts):
    print(datetime.now(), "explain_predictparts")
    return explainer.predict_parts(X.iloc[idxts])  # .plot(min_max=[0, 1], show=False).show()


def importances(res_importances, importances, descr1, descr2, scoring, X):
    newscoring = {}
    for k, lst in res_importances[scoring].items():
        newscoring[k] = lst.copy()
    for i in importances.importances_mean.argsort()[::-1]:
        if importances.importances_mean[i] - importances.importances_std[i] > 0:
            newscoring["description"].append(f"{descr1}-{descr2}-{scoring}")
            newscoring["variable"].append(X.columns[i])
            newscoring["importance-mean"].append(importances.importances_mean[i])
            newscoring["importance-stdev"].append(importances.importances_std[i])
    cpy = res_importances.copy()
    cpy[scoring] = newscoring
    return cpy


def impute(imputation_alg, single_large, single_small):
    print("\n", datetime.now(), f"Impute missing values for single EEG small -----------------------------------------------------------------------------------------------------------")
    imputer = IterativeImputer(estimator=clone(imputation_alg)).fit(X=single_large)
    return DataFrame(imputer.transform(X=single_small), index=single_small.index, columns=single_small.columns)


def percentile_split(df: DataFrame, col=None, out=None):
    df2 = df.copy()
    if None in [col, out]:
        if col is not out:
            raise Exception(f"Both `col`, `out` must be either `None` or something else.")
        col = out = df.columns[0]
    df2.loc[df[col] < df[col].quantile(2 / 5), out] = 0
    df2.loc[df[col] > df[col].quantile(3 / 5), out] = 1
    df2.drop(df2[(df2[col] != 0) & (df2[col] != 1)].index, inplace=True)
    return df2
