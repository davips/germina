from itertools import repeat

import numpy as np
from hdict import hdict, _
from lightgbm import LGBMClassifier as LGBMc, LGBMRegressor as LGBMr
from mlxtend.feature_selection import SequentialFeatureSelector as sfs
from pandas import DataFrame
from shelchemy.scheduler import Scheduler
from sklearn.ensemble import ExtraTreesClassifier as ETc, ExtraTreesRegressor as ETr
from sklearn.ensemble import RandomForestClassifier as RFc, RandomForestRegressor as RFr
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from xgboost import XGBClassifier as XGBc, XGBRegressor as XGBr

from germina.pairwise import pairwise_diff, pairwise_hstack
from germina.runner import ch

__ = enable_iterative_imputer


def interpolate_for_classification(targets, conditions):
    """
    :param targets:
        sorted
    :param conditions:
        `1` means the resulting value should be greater than the corresponding target.
        `0` means the resulting value should be equal than the corresponding target. (`0` is not usually needed)
        `-1` means the resulting value should be lesser than the corresponding target.
    :return:

    # >>> tgts = np.array([77,88,81,84,88,90,95,100,103,105,110,112,115,120])
    # >>> conds = np.array([1,1,-1,1,1,-1,1,-1,-1,1,-1,-1,1,-1])
    # >>> interpolate(tgts, conds)
    93.25
    """
    first = 2 * targets[0] - targets[1]
    last = 2 * targets[-1] - targets[-2]
    targets = np.hstack([np.array([first]), targets, np.array([last])])
    conditions = np.hstack([np.array([1]), conditions, np.array([-1])])
    acc = np.cumsum(conditions)
    mx_mask = acc == np.max(acc)
    mx_idxs = np.flatnonzero(mx_mask)
    neighbors_before = targets[mx_idxs]
    neighbors_after = targets[mx_idxs + 1]
    candidates = (neighbors_before + neighbors_after) / 2
    return np.mean(candidates)


def interpolate_for_regression(targets, conditions):
    candidates = targets + conditions
    return np.mean(candidates)


def imputer(alg, n_estimators, seed, jobs):
    if alg == "lgbm":
        return IterativeImputer(LGBMr(n_estimators=n_estimators, random_state=seed, n_jobs=jobs, deterministic=True, force_row_wise=True), random_state=seed)
    elif alg.endswith("knn"):
        return IterativeImputer(Pipeline(steps=[('scaler', StandardScaler()), ('knn', KNeighborsRegressor(n_neighbors=n_estimators, n_jobs=jobs))]), random_state=seed)
    else:
        raise Exception(f"Unknown {alg=}")


def imputation(Xy_tr, babya, babyb, alg_imp, n_estimators_imp, seed, jobs):
    print("\timputing", end="", flush=True)
    imp = imputer(alg_imp, n_estimators_imp, seed, jobs)
    Xy_tr = imp.fit_transform(Xy_tr)  # First, fit using label info.
    imp.fit(Xy_tr[:, :-1])  # Then fit without labels to be able to make a model compatible with the test instance.
    babyxa = imp.transform(babya[:, :-1])
    babyxb = imp.transform(babyb[:, :-1])
    babya[:, :-1] = babyxa
    babyb[:, :-1] = babyxb
    return Xy_tr, babya, babyb


def selector(forward, alg, n_estimators, k_features, k_folds, seed, jobs):
    if alg == "lgbm":
        return sfs(LGBMr(n_estimators=n_estimators, random_state=seed, n_jobs=1, deterministic=True, force_row_wise=True), k_features=k_features, forward=forward, verbose=0, cv=k_folds, n_jobs=jobs, scoring='r2')
    elif alg == "knn":
        return sfs(Pipeline(steps=[('scaler', StandardScaler()), ('knn', KNeighborsRegressor(n_neighbors=n_estimators, n_jobs=1))]), k_features=k_features, forward=forward, verbose=0, cv=k_folds, n_jobs=jobs, scoring='r2')
    else:
        raise Exception(f"Unknown {alg=}")


def fselection(Xy_tr, babya, babyb, alg_fsel, n_estimators_fsel, forward_fsel, k_features_fsel, k_folds_fsel, seed, jobs):
    print("\tselecting", end="", flush=True)
    sel = selector(forward_fsel, alg_fsel, n_estimators_fsel, k_features_fsel, k_folds_fsel, seed, jobs)
    X_tr = sel.fit_transform(Xy_tr[:, :-1], Xy_tr[:, -1])
    babyxa = sel.transform(babya[:, :-1])
    babyxb = sel.transform(babyb[:, :-1])
    babya = np.hstack([babyxa, babya[:, -1:]])
    babyb = np.hstack([babyxb, babyb[:, -1:]])
    Xy_tr = np.hstack([X_tr, Xy_tr[:, -1:]])
    return Xy_tr, babya, babyb


def predictors(alg, n_estimators, seed, jobs):
    if alg == "rf":
        algclass_c = RFc(n_estimators=n_estimators, random_state=seed, n_jobs=jobs)
        algclass_r = RFr(n_estimators=n_estimators, random_state=seed, n_jobs=jobs)
    elif alg == "lgbm":
        algclass_c = LGBMc(n_estimators=n_estimators, random_state=seed, n_jobs=jobs, deterministic=True, force_row_wise=True)
        algclass_r = LGBMr(n_estimators=n_estimators, random_state=seed, n_jobs=jobs, deterministic=True, force_row_wise=True)
    elif alg == "et":
        algclass_c = ETc(n_estimators=n_estimators, random_state=seed, n_jobs=jobs)
        algclass_r = ETr(n_estimators=n_estimators, random_state=seed, n_jobs=jobs)
    elif alg == "xg":
        algclass_c = XGBc(n_estimators=n_estimators, random_state=seed, n_jobs=jobs)
        algclass_r = XGBr(n_estimators=n_estimators, random_state=seed, n_jobs=jobs)
    elif alg == "cart":
        algclass_c = DecisionTreeClassifier(random_state=seed, n_jobs=jobs)
        algclass_r = DecisionTreeRegressor(random_state=seed, n_jobs=jobs)
    elif alg == "knn":
        algclass_c = Pipeline(steps=[('scaler', StandardScaler()), ('knn', KNeighborsClassifier(n_neighbors=n_estimators, n_jobs=jobs))])
        algclass_r = Pipeline(steps=[('scaler', StandardScaler()), ('knn', KNeighborsRegressor(n_neighbors=n_estimators, n_jobs=jobs))])
    else:
        raise Exception(f"Unknown {alg=}. Options: rf,lgbm,et,xg,cart,knn")
    return algclass_c, algclass_r


# Setting n_nearest_features << n_features, skip_complete=True or increasing tol can help to reduce its computational cost.

def train_c(pairs_X_tr, pairs_y_tr_c, Xts, alg_train, n_estimators_train, seed, jobs):
    print("\ttrainingC", end="", flush=True)
    alg_c = predictors(alg_train, n_estimators_train, seed, jobs)[0]
    alg_c.fit(pairs_X_tr, pairs_y_tr_c)
    predicted_c = alg_c.predict(Xts)
    predictedprobas_c = alg_c.predict_proba(Xts)
    return predicted_c, predictedprobas_c


def train_r(pairs_X_tr, pairs_y_tr_r, alg_train, n_estimators_train, seed, jobs):
    print("\ttrainingR", end="", flush=True)
    alg_r = predictors(alg_train, n_estimators_train, seed, jobs)[1]
    alg_r.fit(pairs_X_tr, pairs_y_tr_r)
    return alg_r


def contrib2prediction(contrib):
    class_index = np.argmax(contrib, axis=1)
    return LabelEncoder().inverse_transform(class_index)


def loo(df: DataFrame, permutation: int, pairwise: str, threshold: float,
        alg, n_estimators,
        n_estimators_imp,
        n_estimators_fsel, forward_fsel, k_features_fsel, k_folds_fsel,
        db, storages: dict, sched: bool,
        seed, jobs: int):
    """
    Perform LOO on both a classifier and a regressor.

    Make two types of predictions for each model: binary, continuous.

    :param df:          Sample including target variable. Last column is the target variable.
    :param pairwise:    pairwise type: by `concatenation`, `difference`, or `none`
    :param threshold:   minimal distance between labels to make a difference between `high` and `low`
                        pairs with distance lesser than `threshold` will be discarded
    :param rejection_threshold__inpct: The model will refuse to answer when predicted value is within `center +- rejection_threshold`.
    :param db:
    :param storages:
    :param sched:
    :param jobs: # of "threads"
    :return:

    (https://docutils.sourceforge.io/docs/ref/rst/restructuredtext.html)
    """
    if k_features_fsel >= df.shape[1] - 1:
        n_estimators_fsel = 0
        forward_fsel = False
        k_features_fsel = 0
        k_folds_fsel = 0

    if pairwise not in {"none", "concatenation", "difference"}:
        raise Exception(f"Not implemented for {pairwise=}")

    df = df.sample(frac=1, random_state=seed)

    # helper functions
    # filter = lambda tmp, thr, me: (tmp[:, -1] < 0) | (tmp[:, -1] / me >= thr)
    # filter = lambda tmp, thr, me: abs(tmp[:, -1] / me) >= thr
    filter = lambda tmp, thr: abs(tmp[:, -1]) >= thr
    if pairwise == "difference":
        pairs = lambda a, b: pairwise_diff(a, b)
        pairs_ts = lambda a, b: pairwise_diff(a, b)
    elif pairwise == "concatenation":
        pairs = lambda a, b: pairwise_hstack(a, b, handle_last_as_y=True)
        pairs_ts = lambda a, b: pairwise_hstack(a, b)
    else:
        raise Exception(f"Not implemented for {pairwise=}")

    if df.isna().sum().sum() == 0:
        n_estimators_imp = 0
    print(df.shape, "<<<<<<<<<<<<<<<<<<<<")

    # LOO
    d = hdict(df=df, alg_train=alg, n_estimators_train=n_estimators,
              alg_imp=alg, n_estimators_imp=n_estimators_imp,
              alg_fsel=alg, n_estimators_fsel=n_estimators_fsel, forward_fsel=forward_fsel, k_features_fsel=k_features_fsel, k_folds_fsel=k_folds_fsel,
              seed=seed, _jobs_=jobs)
    hits_c, hits_r = {0: 0, 1: 0}, {0: 0, 1: 0}
    tot, tot_c = {0: 0, 1: 0}, {0: 0, 1: 0}
    y, y_c, z_lst_c, shap_c = [], [], [], []
    ansi = d.hosh.ansi
    odd = df.index[1::2]
    even = df.index[::2]
    paired = zip(odd, even)
    tasks = zip(repeat(threshold), repeat(pairwise), repeat(d.id), repeat(permutation), paired)
    bacc_c = 0
    for c, (ths, pw, id, per, (idxa, idxb)) in enumerate((Scheduler(db, timeout=60) << tasks) if sched else tasks):
        if not sched:
            print(f"\r Permutation: {permutation:8}\t\t{ansi} pair {idxa, idxb}: {c:3} {100 * c / len(df):8.5f}% {bacc_c:5.3f}          ", end="", flush=True)

        # prepare current pair of babies and training set
        babydfa = df.loc[[idxa], :]
        babydfb = df.loc[[idxb], :]
        baby_ya = babydfa.iloc[0, -1:]
        baby_yb = babydfb.iloc[0, -1:]
        if baby_ya.isna().sum().sum() > 0 or baby_yb.isna().sum().sum() > 0:
            continue  # skip NaN labels
        baby_ya = baby_ya.to_numpy()
        baby_yb = baby_yb.to_numpy()
        babya = babydfa.to_numpy()
        babyb = babydfb.to_numpy()
        Xy_tr = df.drop([idxa, idxb], axis="rows")

        # missing value imputation
        if n_estimators_imp > 0:
            d.apply(imputation, Xy_tr, babya, babyb, jobs=_._jobs_, out="result_imput")
            d = ch(d, storages)
            if not sched:
                print(f"\r Permutation: {permutation:8}\t\t{ansi} pair {idxa, idxb}: {c:3} {100 * c / len(df):8.5f}% {bacc_c:5.3f}          ", end="", flush=True)
            Xy_tr, babya, babyb = d.result_imput
        else:
            Xy_tr = Xy_tr.to_numpy()

        # feature selection
        if k_features_fsel > 0:
            d.apply(fselection, Xy_tr, babya, babyb, jobs=_._jobs_, out="result_fsel")
            d = ch(d, storages)
            if not sched:
                print(f"\r Permutation: {permutation:8}\t\t{ansi} pair {idxa, idxb}: {c:3} {100 * c / len(df):8.5f}% {bacc_c:5.3f}          ", end="", flush=True)
            Xy_tr, babya, babyb = d.result_fsel
        babyxa = babya[:, :-1]
        babyxb = babyb[:, :-1]

        # pairwise transformation
        # training set
        # me = np.mean(Xy_tr[:, -1])
        tmp = pairs(Xy_tr, Xy_tr)
        pairs_Xy_tr = tmp[filter(tmp, threshold)]  # exclui pares com alvos próximos
        Xtr = pairs_Xy_tr[:, :-1]
        ytr_c = (pairs_Xy_tr[:, -1] >= 0).astype(int)
        # print(sum(ytr_c.tolist()), len(ytr_c.tolist()))
        # test set
        Xts = pairs_ts(babyxa, babyxb)

        # training
        d.apply(train_c, Xtr, ytr_c, Xts, jobs=_._jobs_, out="result_train_c")
        d = ch(d, storages)
        if not sched:
            print(f"\r Permutation: {permutation:8}\t\t{ansi} pair {idxa, idxb}: {c:3} {100 * c / len(df):8.5f}% {bacc_c:5.3f}          ", end="", flush=True)

        if sched:
            continue

        # prediction
        predicted_c, predictedprobas_c = d.result_train_c
        predicted_c = predicted_c[0]

        # evaluate
        expected = int(baby_ya[0] >= baby_yb[0])
        tot[expected] += 1
        z_lst_c.append(predicted_c)
        hits_c[expected] += int(expected == predicted_c)
        tot_c[expected] += 1

        # temporary accuracy
        if tot_c[0] * tot_c[1] > 0:
            acc0 = hits_c[0] / tot_c[0]
            acc1 = hits_c[1] / tot_c[1]
            bacc_c = (acc0 + acc1) / 2

        # SHAP
        if False and permutation == 0:
            # print(contrib2prediction()
            # shap_c.append(alg_c.predict(Xts, pred_contrib=True).tolist())
            # shap_r.append(alg_r.predict(Xts, pred_contrib=True).tolist())

            shap_c = alg_c.predict(Xts, pred_contrib=True)
            # shap_r = alg_r.predict(Xts, pred_contrib=True)
            print()
            print()
            print("____________________________________________")
            print()
            print(Xts.shape)
            print()
            print("+++++++++++++++++++++++++++++++++++++")
            print()
            print(DataFrame(shap_c))
            print()
            print("-------------------------------")
            print()
            print(DataFrame(shap_r))
            print()
            # 1 - transforma em toshaps (um por bebe de treino, pois parearam com o bebe de teste pra criar o teste pareado)
            # ...
            exit()

    if sched:
        return

    # classification
    if tot[0] == 0 or tot[1] == 0:
        print(f"Resulted in class total with zero value: {tot=}")
        bacc_c = -1
    elif tot_c[1] == 0:
        print(f"Resulted in class total with zero value: {tot_c=}")
        bacc_c = -1
    else:
        acc0 = hits_c[0] / tot_c[0]
        acc1 = hits_c[1] / tot_c[1]
        bacc_c = (acc0 + acc1) / 2

    # regression
    return d, bacc_c, hits_c, tot, tot_c, shap_c
