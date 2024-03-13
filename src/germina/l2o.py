import warnings
from itertools import repeat

import numpy as np
from pandas import DataFrame
from shelchemy.scheduler import Scheduler
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve, auc

from germina.aux import imputation, trainpredict_optimized
from germina.loo import fselection
from germina.runner import ch
from germina.sampling import pairwise_sample
from germina.shaps import SHAPs

warnings.filterwarnings('ignore')


def loo(df: DataFrame, permutation: int, pairwise: str, threshold: float,
        alg, n_estimators, tries, kfolds,
        n_estimators_imp,
        n_estimators_fsel, forward_fsel, k_features_fsel, k_folds_fsel,
        db, storages: dict, sched: bool, shap: bool, opt: bool,
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
              tries=tries, kfolds=kfolds, opt=opt, shap=shap,
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
        if opt:
            # noinspection PyTypeChecker
            d.apply(trainpredict_optimized, Xw_tr, Xw_ts, jobs=_._jobs_, out="result_train")
        else:
            raise Exception(f"")
            # d.apply(trainpredict_c, Xw_tr, Xw_ts, jobs=_._jobs_, out="result_train")
        d = ch(d, storages)
        if not sched:
            print(f"\r Permutation: {permutation:8}\t\t{ansi} pair {idxa, idxb}: {c:3} {100 * c / len(pairs):8.5f}% {bacc_c:5.3f}          ", end="", flush=True)

        if shap and permutation == 0:
            # noinspection PyTypeChecker
            opt_results = d.result_train["opt_results"]
            raise Exception(f"not ready")

        if sched:
            continue

        # prediction
        predicted_c, probas_c = d.result_train["pred"], d.result_train["prob"]

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
