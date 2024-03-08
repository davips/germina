import warnings
from itertools import repeat

import numpy as np
from lange import ap
from lightgbm import LGBMClassifier as LGBMc
from lightgbm import LGBMRegressor as LGBMr
from pairwiseprediction.classifier import PairwiseClassifier
from pandas import DataFrame
from scipy.stats import poisson, uniform
from shelchemy.scheduler import Scheduler
from sklearn.ensemble import ExtraTreesClassifier as ETc
from sklearn.ensemble import RandomForestClassifier as RFc
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve, auc
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier as XGBc

from germina.loo import fselection
from germina.runner import ch
from germina.sampling import pairwise_sample
from germina.shaps import SHAPs

warnings.filterwarnings('ignore')

__ = enable_iterative_imputer


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


def predictors(alg, n_estimators, seed, jobs):
    match alg:
        case "rf":
            return RFc, {"n_estimators": n_estimators, "random_state": seed, "n_jobs": jobs}
        case "lgbm":
            return LGBMc, {"deterministic": True, "force_row_wise": True, "random_state": seed, "n_jobs": jobs}
        case "et":
            return ETc, {"n_estimators": n_estimators, "random_state": seed, "n_jobs": jobs}
        case "xg":
            return XGBc, {"n_estimators": n_estimators, "random_state": seed, "n_jobs": jobs}
        case "cart":
            return DecisionTreeClassifier, {"random_state": seed}
        case x:
            if x.startswith("ocart"):
                param_dist = {
                    'criterion': ['gini', 'entropy'],
                    'max_depth': poisson(mu=4, loc=2),
                    'min_samples_split': uniform(),
                    'max_leaf_nodes': poisson(mu=12, loc=3)
                }
                print(x)
                n_iter = int(x.split("-")[1])
                cv = int(x.split("-")[2])
                clf = DecisionTreeClassifier()
                return RandomizedSearchCV, {"pre_dispatch": "n_jobs//2", "cv": cv, "n_jobs": jobs, "estimator": clf, "param_distributions": param_dist, "n_iter": n_iter, "random_state": seed, "scoring": "balanced_accuracy"}
            else:
                raise Exception(f"Unknown {alg=}. Options: rf,lgbm,et,xg,cart,ocart-*")


def trainpredict_c(Xwtr, Xwts,
                   alg_train, pairing_style, threshold, proportion, center, only_relevant_pairs_on_prediction,
                   n_estimators_train, seed, jobs):
    print("\ttrainingC", end="", flush=True)
    predictors_, kwargs = predictors(alg_train, n_estimators_train, seed, jobs)
    alg_c = PairwiseClassifier(predictors_,
                               pairing_style, threshold, proportion, center, only_relevant_pairs_on_prediction, **kwargs)
    alg_c.fit(Xwtr)
    predicted_c = alg_c.predict(Xwts, paired_rows=True)[::2]
    predictedprobas_c = alg_c.predict_proba(Xwts, paired_rows=True)[::2]
    return predicted_c, predictedprobas_c


def trainpredictshap(Xwtr, Xwts,
                     alg_train, pairing_style, threshold, proportion, center, only_relevant_pairs_on_prediction,
                     n_estimators_train, columns, seed, jobs):
    print("\ttrainingC", end="", flush=True)
    predictors_, kwargs = predictors(alg_train, n_estimators_train, seed, jobs)
    alg_c = PairwiseClassifier(predictors_,
                               pairing_style, threshold, proportion, center, only_relevant_pairs_on_prediction, **kwargs)
    alg_c.fit(Xwtr)
    print("\tpredictingC", end="", flush=True)
    predicted_labels = alg_c.predict(Xwts, paired_rows=True)[::2]
    predicted_probas = alg_c.predict_proba(Xwts, paired_rows=True)[::2]
    print("\tcalculating SHAP", end="", flush=True)
    shap = alg_c.shap(Xwts[0], Xwts[1], columns, seed)
    return predicted_labels, predicted_probas, shap


def loo(df: DataFrame, permutation: int, pairwise: str, threshold: float,
        alg, n_estimators,
        n_estimators_imp,
        n_estimators_fsel, forward_fsel, k_features_fsel, k_folds_fsel,
        db, storages: dict, sched: bool, shap: bool,
        nsamp, seed, jobs: int):
    """
    Perform Leave-2-Out on a pairwise classifier.

    :param nsamp:
    :param shap:
    :param df:          Sample including target variable. Last column is the target variable.
    :param permutation: A number for this run.
    :param pairwise:    Pairwise type: by `concatenation`, `difference`, or `none`.
    :param threshold:   Minimal distance between labels to make a difference between `high` and `low` pairs with distance lesser than `threshold` will be discarded.
    :param alg:
    :param n_estimators:
    :param n_estimators_imp:
    :param n_estimators_fsel:
    :param forward_fsel:
    :param k_features_fsel:
    :param k_folds_fsel:
    :param db:
    :param storages:
    :param sched:
    :param seed:
    :param jobs: # of "threads"
    :return:

    (https://docutils.sourceforge.io/docs/ref/rst/restructuredtext.html)
    """
    if "cart" in alg:
        n_estimators = 0
    if k_features_fsel >= df.shape[1] - 1:
        n_estimators_fsel = 0
        forward_fsel = False
        k_features_fsel = 0
        k_folds_fsel = 0
    if pairwise not in {"concatenation", "difference"}:
        raise Exception(f"Not implemented for {pairwise=}")
    df = df.sample(frac=1, random_state=seed)
    if df.isna().sum().sum() == 0:
        n_estimators_imp = 0
    print(df.shape, "<<<<<<<<<<<<<<<<<<<<")

    # LOO
    from hdict import hdict, _
    d = hdict(df=df, alg_train=alg, pairing_style=pairwise, n_estimators_train=n_estimators, center=None, only_relevant_pairs_on_prediction=False, threshold=threshold, proportion=False,
              alg_imp=alg, n_estimators_imp=n_estimators_imp,
              alg_fsel=alg, n_estimators_fsel=n_estimators_fsel, forward_fsel=forward_fsel, k_features_fsel=k_features_fsel, k_folds_fsel=k_folds_fsel,
              columns=df.columns.tolist()[:-1],
              seed=seed, _jobs_=jobs)
    hits_c, hits_r = {0: 0, 1: 0}, {0: 0, 1: 0}
    tot, tot_c, errors = {0: 0, 1: 0}, {0: 0, 1: 0}, {0: [], 1: []}
    y, p, z_lst_c = [], [], []
    shaps = SHAPs()
    ansi = d.hosh.ansi
    pairs = pairwise_sample(df.index, nsamp, seed)
    # pairs1 = zip(df.index[::2], df.index[1::2])
    # pairs2 = zip(df.index[1::2], df.index[2::2])
    # pairs = chain(pairs1, pairs2, *pairsx)
    tasks = zip(repeat(alg), repeat(pairwise), repeat(threshold), repeat(d.id), repeat(permutation), pairs)
    bacc_c = 0
    for c, (alg0, pw0, thr0, id0, perm0, (idxa, idxb)) in enumerate((Scheduler(db, timeout=60) << tasks) if sched else tasks):
        if not sched:
            print(f"\r Permutation: {permutation:8}\t\t{ansi} pair {idxa, idxb}: {c:3} {100 * c / len(pairs):8.5f}% {bacc_c:5.3f}          ", end="", flush=True)

        # prepare current pair of babies and training set
        babydfa = df.loc[[idxa], :]
        babydfb = df.loc[[idxb], :]
        baby_ya = babydfa.iloc[0, -1:]
        baby_yb = babydfb.iloc[0, -1:]
        # TODO remove babies with NaN labels in training set?
        if baby_ya.isna().sum().sum() > 0 or baby_yb.isna().sum().sum() > 0:
            continue  # skip NaN labels from testing set
        baby_ya = baby_ya.to_numpy()
        baby_yb = baby_yb.to_numpy()
        babya = babydfa.to_numpy()
        babyb = babydfb.to_numpy()

        Xw_tr = df.drop([idxa, idxb], axis="rows")
        # missing value imputation
        if n_estimators_imp > 0:
            # noinspection PyTypeChecker
            d.apply(imputation, Xw_tr, babya, babyb, jobs=_._jobs_, out="result_imput")
            d = ch(d, storages)
            if not sched:
                print(f"\r Permutation: {permutation:8}\t\t{ansi} pair {idxa, idxb}: {c:3} {100 * c / len(pairs):8.5f}% {bacc_c:5.3f}          ", end="", flush=True)
            Xw_tr, babya, babyb = d.result_imput
        else:
            Xw_tr = Xw_tr.to_numpy()

        # feature selection
        if k_features_fsel > 0:
            # noinspection PyTypeChecker
            d.apply(fselection, Xw_tr, babya, babyb, jobs=_._jobs_, out="result_fsel")
            d = ch(d, storages)
            if not sched:
                print(f"\r Permutation: {permutation:8}\t\t{ansi} pair {idxa, idxb}: {c:3} {100 * c / len(pairs):8.5f}% {bacc_c:5.3f}          ", end="", flush=True)
            Xw_tr, babya, babyb = d.result_fsel
        # babyxa = babya[:, :-1]
        # babyxb = babyb[:, :-1]

        # training
        Xw_ts = np.vstack([babya, babyb])
        if shap and permutation == 0:
            # noinspection PyTypeChecker
            d.apply(trainpredictshap, Xw_tr, Xw_ts, jobs=_._jobs_, out="result_train_c")
        else:
            # noinspection PyTypeChecker
            d.apply(trainpredict_c, Xw_tr, Xw_ts, jobs=_._jobs_, out="result_train_c")
        d = ch(d, storages)
        if not sched:
            print(f"\r Permutation: {permutation:8}\t\t{ansi} pair {idxa, idxb}: {c:3} {100 * c / len(pairs):8.5f}% {bacc_c:5.3f}          ", end="", flush=True)

        if sched:
            continue

        # prediction
        ret = d.result_train_c
        if len(ret) == 3:
            predicted_c, probas_c, shp = ret
        else:
            predicted_c, probas_c = ret
            shp = None
        predicted_c = predicted_c[0]

        # accumulate
        expected = int(baby_ya[0] >= baby_yb[0])
        y.append(expected)
        tot[expected] += 1
        z_lst_c.append(predicted_c)
        p.append(probas_c[0, 1])
        if shap and permutation == 0:
            shaps.add(babya, babyb, shp)
        hits_c[expected] += int(expected == predicted_c)
        tot_c[expected] += 1

        # errors
        if expected == predicted_c:
            errors[expected].append((babydfa, babydfb))

        # temporary accuracy
        if tot_c[0] * tot_c[1] > 0:
            acc0 = hits_c[0] / tot_c[0]
            acc1 = hits_c[1] / tot_c[1]
            bacc_c = (acc0 + acc1) / 2

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
        bacc_c = round((acc0 + acc1) / 2, 2)

    # precision_recall_curve
    aps = round(average_precision_score(y, p), 2) if bacc_c > 0 else None
    pr, rc = precision_recall_curve(y, p)[:2]
    auprc = round(auc(rc, pr), 2) if bacc_c > 0 else None

    return d, bacc_c, hits_c, tot, errors, aps, auprc, shaps
