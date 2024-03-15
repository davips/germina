from itertools import repeat
from sys import argv

import numpy as np
import pandas as pd
from argvsucks import handle_command_line
from hdict import hdict
from lange import ap
from pandas import read_csv
from scipy.stats import poisson, uniform
from shelchemy import sopen
from shelchemy.scheduler import Scheduler
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve, auc
from sklearn.tree import plot_tree
from sympy.physics.control.control_plots import plt

from germina.aux import fit, fitpredict
from germina.config import local_cache_uri, remote_cache_uri, near_cache_uri, schedule_uri
from germina.runner import ch
from germina.sampling import pairwise_sample
from germina.shaps import SHAPs
from germina.trees import tree_optimized_dv

center, threshold = None, 7.5
dct = handle_command_line(argv, font=12, alg=str, demo=False, delta=20, noage=False, sched=False, targetvar=str, jobs=int, seed=0, prefix=str, suffix=str, sps=list, nsamp=int, shap=False, tree=False)
print(dct)
font, alg, demo, delta, noage, sched, targetvar, jobs, seed, prefix, suffix, sps, nsamp, shap, tree = dct["font"], dct["alg"], dct["demo"], dct["delta"], dct["noage"], dct["sched"], dct["targetvar"], dct["jobs"], dct["seed"], dct["prefix"], dct["suffix"], dct["sps"], dct["nsamp"], dct["shap"], dct["tree"]
rnd = np.random.default_rng(0)
trials = int(alg.split("-")[1])
kfolds0 = int(alg.split("-")[2])
if alg.startswith("cartr"):
    search_space = {"criterion": ["absolute_error", "squared_error", "poisson", "friedman_mse"], "max_depth": poisson(mu=5, loc=2), "min_impurity_decrease": uniform(0, 0.01), "max_leaf_nodes": poisson(mu=20, loc=5), "min_samples_split": ap[20, 30, ..., 100].l, "min_samples_leaf": ap[10, 20, ..., 50].l}
else:
    raise Exception(f"{alg=}")

with (sopen(local_cache_uri, ondup="skip") as local_storage, sopen(near_cache_uri, ondup="skip") as near_storage, sopen(remote_cache_uri, ondup="skip") as remote_storage, sopen(schedule_uri) as db):
    storages = {
        # "remote": remote_storage,
        "near": near_storage,
        "local": local_storage,
    }
    for sp in sps:
        sp = int(sp)
        print(f"{sp=} <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
        df = read_csv(f"{prefix}{sp}{suffix}", index_col="id_estudo")
        df.sort_values(targetvar, inplace=True, ascending=True, kind="stable")
        if demo:
            take = min(df.shape[0] // 2, 30)
            df = pd.concat([df.iloc[:take], df.iloc[-take:]], axis="rows")
        if targetvar.endswith("t2"):
            agevar = "idade_crianca_dias_t2"
        elif targetvar.endswith("t3"):
            agevar = "idade_crianca_dias_t3"
        elif targetvar.endswith("t4"):
            agevar = "idade_crianca_dias_t4"
        else:
            raise Exception(f"Unexpected timepoint suffix for target '{targetvar}'.")
        if noage:
            del df[agevar]
        print(df.shape)
        kfolds, kfolds_full = (df.shape[0] - 2, df.shape[0]) if kfolds0 == 0 else (kfolds0, kfolds0)

        if tree:  ###############################################################################################################
            hd = hdict(_verbose_=True)
            # noinspection PyTypeChecker
            hd.apply(tree_optimized_dv, df, search_space, trials, kfolds_full, alg, seed=0, out="best")
            hd = ch(hd, storages)
            best_params, best_score = hd.best
            # noinspection PyTypeChecker
            hd.apply(fit, alg, best_params, df, out="tree")
            hd = ch(hd, storages)
            best_estimator = hd.tree

            print("vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv")
            print(best_params)
            print("--------------------------------")
            print(best_score)
            print(" ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^")
            columns = df.columns.tolist()[:-1]
            plot_tree(best_estimator, filled=True, feature_names=columns, fontsize=font)
            plt.title(f"Decision tree for {targetvar}")
            plt.show()
            continue

        # L2O ##############################################################################################################
        d = hdict(df=df, alg_train=alg, columns=df.columns.tolist()[:-1], trials=trials, kfolds=kfolds, shap=shap, seed=seed, _jobs_=jobs)
        hits = {0: 0, 1: 0}
        tot, errors = {0: 0, 1: 0}, {0: [], 1: []}
        t, z = [], []
        bacc = 0
        shaps = SHAPs()
        ansi = d.hosh.ansi
        pairs = pairwise_sample(df.index, nsamp, seed)
        tasks = zip(repeat(targetvar), repeat(alg), repeat(d.id), pairs)
        for c, (targetvar0, alg0, did0, (idxa, idxb)) in enumerate((Scheduler(db, timeout=60) << tasks) if sched else tasks):
            if not sched:
                print(f"\r{ansi} {targetvar0, alg0, (idxa, idxb)}: {c:3} {100 * c / len(pairs):4.2f}% {bacc:5.3f}          ", end="", flush=True)

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

            # optimize
            Xw_tr = df.drop([idxa, idxb], axis="rows")
            Xw_ts = np.vstack([babya, babyb])
            # noinspection PyTypeChecker
            d.apply(tree_optimized_dv, Xw_tr, search_space, trials, kfolds, alg, seed=0, out="best")
            d = ch(d, storages)
            best_params, best_score = d.best
            if not sched:
                print(f"\r{ansi} {targetvar0, alg0, (idxa, idxb)}: {c:3} {100 * c / len(pairs):4.2f}% {bacc:5.3f}          ", end="", flush=True)

            # fit, predict
            d.apply(fitpredict, alg, best_params, Xw_tr, Xw_ts[:, :-1], out="zts")
            d = ch(d, storages)
            zts = d.zts
            if not sched:
                print(f"\r{ansi} {targetvar0, alg0, (idxa, idxb)}: {c:3} {100 * c / len(pairs):4.2f}% {bacc:5.3f}          ", end="", flush=True)

            if shap:
                raise Exception(f"")
                # noinspection PyTypeChecker
                # d.apply(shap_for_pair, d.result_train["best_params"], babya, babyb, Xw_tr, jobs=_._jobs_, out="result_shap")
                # d = ch(d, storages)
                # shp = d.result_shap
                # shaps.add(babya, babyb, shp)

            if sched:
                continue

            # accumulate
            expected = int(abs(baby_ya[0] - baby_yb[0]) >= delta)
            predicted = int(abs(zts[0] - zts[1]) >= delta)
            # expected = int(baby_ya[0] >= baby_yb[0])
            # predicted = int(zts[0] >= zts[1])
            t.append(expected)
            tot[expected] += 1
            z.append(predicted)
            hits[expected] += int(expected == predicted)

            # errors
            if expected != predicted:
                errors[expected].append((babydfa, babydfb))

            # temporary accuracy
            if tot[0] * tot[1] > 0:
                acc0 = hits[0] / tot[0]
                acc1 = hits[1] / tot[1]
                bacc = (acc0 + acc1) / 2

        if sched:
            continue

        # classification
        if tot[0] == 0 or tot[1] == 0:
            print(f"Resulted in class total with zero value: {tot=}")
            bacc = -1

        # precision_recall_curve
        aps = round(average_precision_score(t, z), 2) if bacc > 0 else None
        pr, rc = precision_recall_curve(t, z)[:2]
        auprc = round(auc(rc, pr), 2) if bacc > 0 else None
        print(f"\r{sp=} {delta=} {hits=} {tot=} \t{d.hosh.ansi} | {bacc=:4.3f} | {aps=} | {auprc=} ", flush=True)

    print("\n")
