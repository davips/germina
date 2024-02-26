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


def imputation(Xy_tr, baby, alg_imp, n_estimators_imp, seed, jobs):
    print("\timputing", end="", flush=True)
    imp = imputer(alg_imp, n_estimators_imp, seed, jobs)
    Xy_tr = imp.fit_transform(Xy_tr)  # First, fit using label info.
    imp.fit(Xy_tr[:, :-1])  # Then fit without labels to be able to make a model compatible with the test instance.
    babyx = imp.transform(baby[:, :-1])
    baby[:, :-1] = babyx
    return Xy_tr, baby


def selector(forward, alg, n_estimators, k_features, k_folds, seed, jobs):
    if alg == "lgbm":
        return sfs(LGBMr(n_estimators=n_estimators, random_state=seed, n_jobs=1, deterministic=True, force_row_wise=True), k_features=k_features, forward=forward, verbose=0, cv=k_folds, n_jobs=jobs, scoring='r2')
    elif alg == "knn":
        return sfs(Pipeline(steps=[('scaler', StandardScaler()), ('knn', KNeighborsRegressor(n_neighbors=n_estimators, n_jobs=1))]), k_features=k_features, forward=forward, verbose=0, cv=k_folds, n_jobs=jobs, scoring='r2')
    else:
        raise Exception(f"Unknown {alg=}")


def fselection(Xy_tr, baby, alg_fsel, n_estimators_fsel, forward_fsel, k_features_fsel, k_folds_fsel, seed, jobs):
    print("\tselecting", end="", flush=True)
    sel = selector(forward_fsel, alg_fsel, n_estimators_fsel, k_features_fsel, k_folds_fsel, seed, jobs)
    X_tr = sel.fit_transform(Xy_tr[:, :-1], Xy_tr[:, -1])
    babyx = sel.transform(baby[:, :-1])
    baby = np.hstack([babyx, baby[:, -1:]])
    Xy_tr = np.hstack([X_tr, Xy_tr[:, -1:]])
    return Xy_tr, baby


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

def train_c(pairs_X_tr, pairs_y_tr_c, alg_train, n_estimators_train, seed, jobs):
    print("\ttrainingC", end="", flush=True)
    alg_c = predictors(alg_train, n_estimators_train, seed, jobs)[0]
    alg_c.fit(pairs_X_tr, pairs_y_tr_c)
    return alg_c


def train_r(pairs_X_tr, pairs_y_tr_r, alg_train, n_estimators_train, seed, jobs):
    print("\ttrainingR", end="", flush=True)
    alg_r = predictors(alg_train, n_estimators_train, seed, jobs)[1]
    alg_r.fit(pairs_X_tr, pairs_y_tr_r)
    return alg_r


def contrib2prediction(contrib):
    class_index = np.argmax(contrib, axis=1)
    return LabelEncoder().inverse_transform(class_index)


def loo(df: DataFrame, permutation: int, pairwise: str, threshold: float, rejection_threshold__inpct: float, extreme_pairing_onprediction: float,
        alg, n_estimators,
        n_estimators_imp,
        n_estimators_fsel, forward_fsel, k_features_fsel, k_folds_fsel,
        db, storages: dict, sched: bool,
        seed, jobs: int, pct: bool, center: float):
    """
    Perform LOO on both a classifier and a regressor.

    Make two types of predictions for each model: binary, continuous.

    :param df:          Sample including target variable. Last column is the target variable.
    :param pairwise:    pairwise type: by `concatenation`, `difference`, or `none`
    :param threshold:   minimal distance between labels to make a difference between `high` and `low`
                        pairs with distance lesser than `threshold` will be discarded
                        TODO: option for relative distance `concatenation%`, `difference%`
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

    if pairwise not in {"none", "concatenation", "difference", "concatenation%", "difference%"}:  # TODO: %
        raise Exception(f"Not implemented for {pairwise=}")

    # helper functions
    handle_last_as_y = "%" if pct else True
    filter = lambda tmp: abs(tmp[:, -1]) >= threshold
    if pairwise == "difference":
        pairs = lambda a, b: pairwise_diff(a, b, pct=handle_last_as_y == "%")
        pairs_ts = lambda a, b: pairwise_diff(a, b)
    elif pairwise == "concatenation":
        pairs = lambda a, b: pairwise_hstack(a, b, handle_last_as_y=handle_last_as_y)
        pairs_ts = lambda a, b: pairwise_hstack(a, b)
    elif pairwise == "none":
        if pct:
            df = df.loc[abs(df.iloc[:, -1] / center - 1) >= threshold]
        else:
            df = df.loc[abs(df.iloc[:, -1] - center) >= threshold]
        pairwise = False
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
    tot, tot_c, tot_r = {0: 0, 1: 0}, {0: 0, 1: 0}, {0: 0, 1: 0}
    y, y_c, y_r, z_lst_c, z_lst_r, shap_c, shap_r = [], [], [], [], [], [], []
    ansi = d.hosh.ansi
    tasks = zip(repeat([threshold, rejection_threshold__inpct]), repeat(pairwise), repeat(d.id), repeat(permutation), df.index)
    for c, (ths, pw, id, per, idx) in enumerate((Scheduler(db, timeout=60) << tasks) if sched else tasks):
        if not sched:
            print(f"\r Permutation: {permutation:8}\t\t{ansi} baby {idx}: {c:3} {100 * c / len(df):8.5f}%             ", end="", flush=True)

        # prepare current baby and training set
        babydf = df.loc[[idx], :]
        # baby_x = babydf.iloc[:, :-1]
        baby_y = babydf.iloc[0, -1:]
        if baby_y.isna().sum().sum() > 0:
            continue  # skip NaN labels
        baby_y = baby_y.to_numpy()
        baby = babydf.to_numpy()
        babyx = baby[:, :-1]
        Xy_tr = df.drop(idx, axis="rows")

        # missing value imputation
        if n_estimators_imp > 0:
            # Xy_tr, baby = imputation(Xy_tr, baby, alg, n_estimators_fsel_imput, seed, jobs)
            d.apply(imputation, Xy_tr, baby, jobs=_._jobs_, out="result_imput")
            d = ch(d, storages)
            if not sched:
                print(f"\r Permutation: {permutation:8}\t\t{ansi} baby {idx}: {c:3} {100 * c / len(df):8.5f}%             ", end="", flush=True)
            Xy_tr, baby = d.result_imput
        else:
            Xy_tr = Xy_tr.to_numpy()

        # feature selection
        if k_features_fsel > 0:
            # Xy_tr, baby = fselection(Xy_tr, baby, alg, n_estimators_fsel_imput, forward_fsel, k_features_fsel, k_folds_fsel, seed, jobs)
            d.apply(fselection, Xy_tr, baby, jobs=_._jobs_, out="result_fsel")
            d = ch(d, storages)
            if not sched:
                print(f"\r Permutation: {permutation:8}\t\t{ansi} baby {idx}: {c:3} {100 * c / len(df):8.5f}%             ", end="", flush=True)
            Xy_tr, baby = d.result_fsel

        if pairwise:  # pairwise transformation
            # training set
            tmp = pairs(Xy_tr, Xy_tr)
            pairs_Xy_tr = tmp[filter(tmp)]
            Xtr = pairs_Xy_tr[:, :-1]
            ytr_c = (pairs_Xy_tr[:, -1] >= 0).astype(int)
            ytr_r = pairs_Xy_tr[:, -1]
            # test set
            if extreme_pairing_onprediction:
                if pct:
                    Xy_tr = Xy_tr[abs(Xy_tr[:, -1] / center - 1) >= threshold]
                else:
                    Xy_tr = Xy_tr[abs(Xy_tr[:, -1] - center) >= threshold]

            Xts = pairs_ts(babyx, Xy_tr[:, :-1])
            # Xts_a = pairs_ts(babyx, Xy_tr[:, :-1])
            # Xts_b = pairs_ts(Xy_tr[:, :-1], babyx)
            # Xts = np.vstack([Xts_a, Xts_b])

            # true values for pairs (they are irrelevant):
            # yts_c = (Xy_ts[:, -1] >= 0).astype(int)
            # yts_r = Xy_ts[:, -1]
        else:
            Xtr = Xy_tr[:, :-1]
            ytr_c = (Xy_tr[:, -1] >= center).astype(int)
            ytr_r = Xy_tr[:, -1]
            # test set
            Xts = baby[:, :-1]

        # training
        d.apply(train_c, Xtr, ytr_c, jobs=_._jobs_, out="result_train_c")
        d = ch(d, storages)
        if not sched:
            print(f"\r Permutation: {permutation:8}\t\t{ansi} baby {idx}: {c:3} {100 * c / len(df):8.5f}%             ", end="", flush=True)
        # d.apply(train_r, Xtr, ytr_r, jobs=_._jobs_, out="result_train_r")
        # d = ch(d, storages)
        # alg_c, alg_r = d.result_train_c, d.result_train_r
        alg_c = d.result_train_c

        if sched:
            continue

        # prediction
        zts_c = alg_c.predict(Xts)
        # zts_r = alg_r.predict(Xts)

        if pairwise:
            # interpolation
            conditions = 2 * zts_c - 1

            # conditions[len(Xy_tr):] = -conditions[len(Xy_tr):]
            # targets = np.hstack([Xy_tr[:, -1], Xy_tr[:, -1]])
            targets = Xy_tr[:, -1]

            baby_z_c = interpolate_for_classification(targets, conditions)
            # baby_z_r = interpolate_for_regression(targets, conditions=zts_r)
        else:
            baby_z_c = zts_c[0]
            # baby_z_r = zts_r[0]

        # evaluate on accepted instances
        expected = int(baby_y[0] >= center)
        tot[expected] += 1
        y.append(baby_y[0])
        if abs(baby_z_c / center - 1) >= rejection_threshold__inpct:
            y_c.append(baby_y[0])
            z_lst_c.append(baby_z_c)
            predicted_c = int(baby_z_c >= center)
            hits_c[expected] += int(expected == predicted_c)
            tot_c[expected] += 1
        # if abs(baby_z_r - center) >= rejection_threshold:
        #     y_r.append(baby_y[0])
        #     z_lst_r.append(baby_z_r)
        #     predicted_r = int(baby_z_r >= center)
        #     hits_r[expected] += int(expected == predicted_r)
        #     tot_r[expected] += 1

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
        print(f"Possible problems with `center`. Resulted in class total with zero value: {tot=} {center=}")
        rj_c = rj_r = bacc_c = bacc_r = -1
    # elif tot_r[0] == 0 or tot_c[1] == 0:
    elif tot_c[1] == 0:
        print(f"Possible problems with `rejection_threshold`. Resulted in class total with zero value: {tot_c=} {tot_r=} {rejection_threshold__inpct=}")
        rj_c = rj_r = bacc_c = bacc_r = -1
    else:
        rj_c = (len(y) - len(y_c)) / len(y)
        # rj_r = (len(y) - len(y_r)) / len(y)
        acc0 = hits_c[0] / tot_c[0]
        acc1 = hits_c[1] / tot_c[1]
        bacc_c = (acc0 + acc1) / 2
        # acc0 = hits_r[0] / tot_r[0]
        # acc1 = hits_r[1] / tot_r[1]
        # bacc_r = (acc0 + acc1) / 2

    # regression
    z_c = np.array(z_lst_c)
    # z_r = np.array(z_lst_r)
    r2_c = r2_score(y_c, z_c)
    # r2_r = r2_score(y_r, z_r)
    bacc_r = rj_r = r2_r = -1
    # print()
    # print(y)
    return d, bacc_c, bacc_r, r2_c, r2_r, hits_c, hits_r, tot, tot_c, tot_r, rj_c, rj_r, shap_c, shap_r
