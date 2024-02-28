from itertools import repeat

from hdict import hdict, _
from pandas import DataFrame
from shelchemy.scheduler import Scheduler
from sklearn.experimental import enable_iterative_imputer

from germina.loo import imputation, fselection, train_c
from germina.pairwise import pairwise_diff, pairwise_hstack
from germina.runner import ch

__ = enable_iterative_imputer


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
    targetvar = df.columns[-1]
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

            # shap_c = alg_c.predict(Xts, pred_contrib=True)
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
