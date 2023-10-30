import dalex as dx
import xgboost as xgb
from datetime import datetime
from pprint import pprint

import numpy as np
import pandas as pd
from hdict import apply, _
from hdict.dataset.pandas_handling import file2df
from pandas import DataFrame
from sklearn.preprocessing import StandardScaler

from germina.runner import drop_many_by_vif, ch, sgid2estudoid, setindex


def load_from_csv(d, storages, storage_to_be_updated, path, vif, filename, field, transpose, old_indexname="id_estudo"):
    print(datetime.now())
    d = d >> apply(file2df, path + filename + ".csv", transpose=transpose, index=True)(field)
    d = ch(d, storages, storage_to_be_updated)
    print(f"Loaded '{field}' data from '{filename}.csv' ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓", d[field].shape, "↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑")
    d = d >> apply(setindex, _[field], old_indexname=old_indexname)(field)
    d = ch(d, storages, storage_to_be_updated)
    if vif:
        print(f"Apply VIF to '{field}' ----------------------------------------------------------------------------------------------------------------------------")
        print(datetime.now())
        d = drop_many_by_vif(d, field, storages, storage_to_be_updated, [], [])
        d = ch(d, storages, storage_to_be_updated)
        print("after VIF ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓", d[field].shape, "↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑")
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
    d = d >> apply(lambda df: pd.get_dummies(df["EBF_3m"])["EBF"].astype(int)).y
    d = ch(d, storages, storage_to_be_updated)
    print("Scaled ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓", d.X.shape, d.y.shape, "↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑\n")
    return d


def train_xgb(params, X, y, idxtr):
    return xgb.train(params, xgb.DMatrix(X[idxtr], label=y[idxtr]))


def get_balance(d, storages, storage_to_be_updated):
    print("Calculate class balance -------------------------------------------------------------------------------------------------------------------------------------------------")
    print(datetime.now())
    d = d >> apply(lambda X: X.shape).Xshape
    d = d >> apply(lambda y: y.shape).yshape
    d = d >> apply(lambda y: np.unique(y, return_counts=True))("unique_labels", "counts")
    d = d >> apply(lambda y, counts: counts / len(y)).proportions
    d = ch(d, storages, storage_to_be_updated)
    print("X, y:", d.Xshape, d.yshape)
    print(f"{d.counts=}\t{d.proportions=}")
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
