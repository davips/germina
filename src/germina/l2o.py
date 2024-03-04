from itertools import repeat, chain

import numpy as np
from lightgbm import LGBMClassifier as LGBMc
from pairwiseprediction.classifier import PairwiseClassifier
from pandas import DataFrame
from shelchemy.scheduler import Scheduler
from sklearn.ensemble import ExtraTreesClassifier as ETc
from sklearn.ensemble import RandomForestClassifier as RFc
from sklearn.experimental import enable_iterative_imputer
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier as XGBc

from germina.loo import imputation, fselection
from germina.runner import ch

__ = enable_iterative_imputer


def predictors(alg):
    match alg:
        case "rf":
            return RFc
        case "lgbm":
            return LGBMc
        case "et":
            return ETc
        case "xg":
            return XGBc
        case "cart":
            return DecisionTreeClassifier
        case _:
            raise Exception(f"Unknown {alg=}. Options: rf,lgbm,et,xg,cart")


def trainpredict_c(Xwtr, Xwts,
                   alg_train, pairing_style, threshold, proportion, center, only_relevant_pairs_on_prediction,
                   n_estimators_train, seed, jobs):
    print("\ttrainingC", end="", flush=True)
    kwargs = dict(n_estimators=n_estimators_train, random_state=seed, n_jobs=jobs)
    if alg_train == "lgbm":
        kwargs["deterministic"] = kwargs["force_row_wise"] = True
    alg_c = PairwiseClassifier(predictors(alg_train),
                               pairing_style, threshold, proportion, center, only_relevant_pairs_on_prediction, **kwargs)
    alg_c.fit(Xwtr)
    predicted_c = alg_c.predict(Xwts, paired_rows=True)[::2]
    predictedprobas_c = alg_c.predict_proba(Xwts, paired_rows=True)[::2]
    return predicted_c, predictedprobas_c


def loo(df: DataFrame, permutation: int, pairwise: str, threshold: float,
        alg, n_estimators,
        n_estimators_imp,
        n_estimators_fsel, forward_fsel, k_features_fsel, k_folds_fsel,
        db, storages: dict, sched: bool,
        seed, jobs: int):
    """
    Perform Leave-2-Out on a pairwise classifier.

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
              seed=seed, _jobs_=jobs)
    hits_c, hits_r = {0: 0, 1: 0}, {0: 0, 1: 0}
    tot, tot_c = {0: 0, 1: 0}, {0: 0, 1: 0}
    y, y_c, z_lst_c, shap_c = [], [], [], []
    ansi = d.hosh.ansi
    pairs1 = zip(df.index[::2], df.index[1::2])
    pairs2 = zip(df.index[1::2], df.index[2::2])
    pairs = chain(pairs1, pairs2)
    tasks = zip(repeat(alg), repeat(pairwise), repeat(threshold), repeat(d.id), repeat(permutation), pairs)
    bacc_c = 0
    targetvar = df.columns[-1]
    for c, (alg0, pw0, thr0, id0, perm0, (idxa, idxb)) in enumerate((Scheduler(db, timeout=60) << tasks) if sched else tasks):
        if not sched:
            print(f"\r Permutation: {permutation:8}\t\t{ansi} pair {idxa, idxb}: {c:3} {50 * c / len(df):8.5f}% {bacc_c:5.3f}          ", end="", flush=True)

        # prepare current pair of babies and training set
        babydfa = df.loc[[idxa], :]
        babydfb = df.loc[[idxb], :]
        baby_ya = babydfa.iloc[0, -1:]
        baby_yb = babydfb.iloc[0, -1:]
        # TODO remove babies with NaN labels in training set
        if baby_ya.isna().sum().sum() > 0 or baby_yb.isna().sum().sum() > 0:
            continue  # skip NaN labels from testing set
        baby_ya = baby_ya.to_numpy()
        baby_yb = baby_yb.to_numpy()
        babya = babydfa.to_numpy()
        babyb = babydfb.to_numpy()

        Xw_tr = df.drop([idxa, idxb], axis="rows")
        # missing value imputation
        if n_estimators_imp > 0:
            d.apply(imputation, Xw_tr, babya, babyb, jobs=_._jobs_, out="result_imput")
            d = ch(d, storages)
            if not sched:
                print(f"\r Permutation: {permutation:8}\t\t{ansi} pair {idxa, idxb}: {c:3} {50 * c / len(df):8.5f}% {bacc_c:5.3f}          ", end="", flush=True)
            Xw_tr, babya, babyb = d.result_imput
        else:
            Xw_tr = Xw_tr.to_numpy()

        # feature selection
        if k_features_fsel > 0:
            d.apply(fselection, Xw_tr, babya, babyb, jobs=_._jobs_, out="result_fsel")
            d = ch(d, storages)
            if not sched:
                print(f"\r Permutation: {permutation:8}\t\t{ansi} pair {idxa, idxb}: {c:3} {50 * c / len(df):8.5f}% {bacc_c:5.3f}          ", end="", flush=True)
            Xw_tr, babya, babyb = d.result_fsel
        babyxa = babya[:, :-1]
        babyxb = babyb[:, :-1]

        # training  Xwtr, Xwts,
        #           alg_train, pairing_style, threshold, proportion, center, only_relevant_pairs_on_prediction,
        #           n_estimators_train, seed, jobs
        Xw_ts = np.vstack([babya, babyb])
        d.apply(trainpredict_c, Xw_tr, Xw_ts, jobs=_._jobs_, out="result_train_c")
        d = ch(d, storages)
        if not sched:
            print(f"\r Permutation: {permutation:8}\t\t{ansi} pair {idxa, idxb}: {c:3} {50 * c / len(df):8.5f}% {bacc_c:5.3f}          ", end="", flush=True)

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

            # shap_c = alg_c.predict(Xts, pred_contrib=True)
            # shap_r = alg_r.predict(Xts, pred_contrib=True)
            print()
            print()
            print("____________________________________________")
            print()
            # print(Xts.shape)
            print()
            print("+++++++++++++++++++++++++++++++++++++")
            print()
            print(DataFrame(shap_c))
            print()
            print("-------------------------------")
            print()
            # print(DataFrame(shap_r))
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
