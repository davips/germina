import copy

import xgboost as xgb
from datetime import datetime
from pprint import pprint

import numpy as np
import pandas as pd
from germina.nan import remove_nan_rows_cols
from sklearn import clone
from sklearn.impute import IterativeImputer
from sklearn.metrics._scorer import _ProbaScorer

from germina.dataset import join
from hdict import apply, _, field
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
        print(f"Loaded '{field}' data from '{filename}.csv' ↓↓↓↓↓↓↓", d[field].shape, "↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑")
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
    print("Fixed id ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓\n", d[field].shape, "↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑")
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
        vars.sort(kind="stable")
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


def apply_std(d, storages, storage_to_be_updated, path, vif, field, verbose=False):
    if verbose:
        print(datetime.now())
    d = d >> apply(lambda x: DataFrame(StandardScaler().fit_transform(x)), _[field])(field)
    d = d >> apply(setindex, _[field])(field)
    d = ch(d, storages, storage_to_be_updated)
    if verbose:
        print("Scaled ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓", d[field].shape, "↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑\n")
    return d


def cut(df, target_var, div=2):
    df2: DataFrame = df.copy()
    me = df2[target_var].mean()
    st = df2[target_var].std()
    if isinstance(div, str):
        hi = df2[target_var] > me + st / float(div)
        lo = df2[target_var] < me - st / float(div)
        div = 2
    else:
        hi = df2[target_var] > me + st / 2
        lo = df2[target_var] < me - st / 2
    if div == 2:
        pos = df2[target_var][hi] * 0 + 1
        neg = df2[target_var][lo] * 0
        df2[target_var] = pd.concat([neg, pos])
    else:
        df2[target_var] = df2[target_var] * 0 + 1
        df2[target_var][hi] = df2[target_var][hi] * 0 + 2
        df2[target_var][lo] = df2[target_var][lo] * 0
    return df2[target_var].dropna().astype(int)


def clean_for_dalex(d, storages, storage_to_be_updated, verbose=False, target="EBF_3m", alias="EBF", keep=[], field=None):
    if field is None:
        raise Exception(f"")
    if verbose:
        print(datetime.now())
    if alias != "EBF":
        d = d >> apply(remove_nan_rows_cols, keep=keep).df
    d = d >> apply(lambda df, tgt: df.drop([tgt], axis=1), tgt=target).X0
    d = d >> apply(lambda X0: [col.replace("[", "").replace("]", "").replace("<", "").replace(" ", "_").replace(":", "_").replace(".", "_").replace("-", "_").replace("(", "").replace(")", "").replace("{", "").replace("}", "").replace(";", "").replace(",", "") for col in X0.columns]).Xcols
    d = d >> apply(lambda X0, Xcols: X0.rename(columns=dict(zip(X0.columns, Xcols)))).X
    d["Xor"] = _.X
    d = d >> apply(lambda df, tgt: df[tgt], tgt=target).yor
    if alias == "EBF":
        d = d >> apply(lambda X: pd.get_dummies(X["delivery_mode"])["vaginal"].astype(int)).delivery_mode
        d = d >> apply(join, df=_.X, other=_.delivery_mode).X
        d = d >> apply(lambda df, tgt: pd.get_dummies(df[tgt])[alias].astype(int), tgt=target).y
    elif d.div == -1:  # pairwise regression
        d = d >> apply(lambda df, tgt: df[tgt], tgt=target).y
        d = ch(d, storages, storage_to_be_updated)
        d.apply(lambda X, y: X.loc[y.index], out=f"X")
    else:
        d = d >> apply(cut).y
        d = ch(d, storages, storage_to_be_updated)
        d.apply(lambda X, y: X.loc[y.index], out=f"X")
    d = ch(d, storages, storage_to_be_updated)
    d.apply(lambda df, alias, y: df[alias].loc[y.index], alias=alias, out=f"yor_{field}_{alias}")
    if verbose:
        print("Cleaned ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓", d.X.shape, d.y.shape, "↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑\n")
    return d


def train_xgb(params, X, y, idxtr):
    print(datetime.now(), "train_xgb")
    return xgb.train(params, xgb.DMatrix(X.iloc[idxtr], label=y.iloc[idxtr]))


def get_balance(d, storages, storage_to_be_updated, verbose=False):
    d = d >> apply(lambda X: X.shape).Xshape
    d = d >> apply(lambda y: y.shape).yshape
    d = d >> apply(lambda y: np.unique(y, return_counts=True))("unique_labels", "counts")
    d = d >> apply(lambda y, counts: counts / len(y)).proportions
    d = ch(d, storages, storage_to_be_updated)
    if verbose:
        print(datetime.now(), "X, y:", d.Xshape, d.yshape, f"{d.counts=}\t{d.proportions=}")
    return d


def importances(res_importances, importances, descr1, descr2, scoring, X):
    newscoring = {}
    for k, lst in res_importances[scoring].items():
        newscoring[k] = lst.copy()
    for i in importances.importances_mean.argsort(kind="stable")[::-1]:
        if importances.importances_mean[i] - importances.importances_std[i] > 0:
            newscoring["description"].append(f"{descr1}-{descr2}-{scoring}")
            newscoring["variable"].append(X.columns[i])
            newscoring["importance_mean"].append(importances.importances_mean[i])
            newscoring["importance_std"].append(importances.importances_std[i])
    cpy = res_importances.copy()
    cpy[scoring] = newscoring
    return cpy


def importances2(res_importances, contribs_accumulator, values_accumulator, descr1, descr2, scoring):
    """Roda 1 vez por cenário (descr1, descr2, scoring)"""
    dctmean, dctstd, dctpval, dctvaluescontribs = {}, {}, {}, {}
    for variable, v in contribs_accumulator.items():
        dctmean[variable] = np.mean(v)
        dctstd[variable] = np.std(v)
        dctpval[variable] = (np.sum(np.array(v) >= 0) + 1.0) / (len(contribs_accumulator) + 1)
        dctvaluescontribs[variable] = list(zip(values_accumulator[variable], contribs_accumulator[variable]))

    newscoring = {}
    for variable, lst in res_importances[scoring].items():  # copia anterior para acrescentar novo cenário no loop abaixo
        newscoring[variable] = lst.copy()
    for (variable, m), s, pval, valuescontribs in zip(dctmean.items(), dctstd.values(), dctpval.values(), dctvaluescontribs.values()):  # uma volta para cada bebê (LOO)
        newscoring["description"].append(f"{descr1}-{descr2}-{scoring}")
        newscoring["variable"].append(variable)
        newscoring["shap_mean"].append(m)
        newscoring["shap_std"].append(s)
        newscoring["shap_p-value"].append(pval)
        newscoring["values_shaps"][variable].append(valuescontribs)
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


def aaa(predictparts, contribs_accumulator):
    contribs = dict(zip(predictparts.result["variable"], predictparts.result["contribution"]))
    contribs = {k.split(" ")[0]: v for k, v in contribs.items()}
    if contribs_accumulator is None:
        contribs_accumulator = {k: [v] for k, v in contribs.items()}
    else:
        for k, v in contribs.items():
            contribs_accumulator[k].append(v)
    return contribs_accumulator


def bbb(predictparts, values_accumulator):
    values = {name_val.split(" = ")[0]: float(name_val.split(" = ")[1:][0]) for name_val in predictparts.result["variable"]}
    if values_accumulator is None:
        values_accumulator = {k: [v] for k, v in values.items()}
    else:
        for k, v in values.items():
            values_accumulator[k].append(v)
    return values_accumulator


def start_reses(res, measure, res_importances):
    res = res.copy()
    res_importances = res_importances.copy()
    # res[measure] = {"description": [], "score": [], "p-value": []}
    res[measure] = {"description": [], "alg": [], "score": [], "p-value": []}
    res_importances[measure] = {"description": [], "variable": [], "shap_mean": [], "shap_std": [], "shap_p-value": [], "values_shaps": {}}
    return res, res_importances


def ccc(scoring, res, field, alg_name, d_score, target_var, d_pval):
    if isinstance(scoring, _ProbaScorer):
        scoring = "average_precision_score"
    res = copy.deepcopy(res)
    res[scoring]["description"].append(f"{field}-{target_var}-{scoring}")
    res[scoring]["alg"].append(alg_name)
    res[scoring]["score"].append(d_score)
    res[scoring]["p-value"].append(d_pval)
    print(f"{scoring:20} (p-value):\t{d_score:.4f} ({d_pval:.4f})", flush=True)
    return res
