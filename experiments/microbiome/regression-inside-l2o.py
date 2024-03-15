from itertools import repeat
from statistics import correlation
from sys import argv

import numpy as np
import pandas as pd
from argvsucks import handle_command_line
from hdict import hdict, _
from pandas import read_csv
from scipy.stats import kendalltau
from shelchemy import sopen
from shelchemy.scheduler import Scheduler
from sklearn.metrics import average_precision_score, r2_score
from sklearn.metrics import precision_recall_curve, auc
from sklearn.tree import plot_tree
from sympy.physics.control.control_plots import plt

from germina.aux import fit, fitpredict
from germina.config import local_cache_uri, remote_cache_uri, near_cache_uri, schedule_uri
from germina.runner import ch
from germina.sampling import pairwise_sample
from germina.shaps import SHAPs
from germina.trees import tree_optimized_dv_pair

center, max_trials = None, 10_000_000
dct = handle_command_line(argv, r2=False, batches=int, cache=False, font=12, alg=str, demo=False, delta=20, noage=False, sched=False, targetvar=str, jobs=int, seed=0, prefix=str, suffix=str, sps=list, nsamp=int, shap=False, tree=False)
print(dct)
use_r2, batches, use_cache, font, alg, demo, delta, noage, sched, targetvar, jobs, seed, prefix, suffix, sps, nsamp, shap, tree = dct["r2"], dct["batches"], dct["cache"], dct["font"], dct["alg"], dct["demo"], dct["delta"], dct["noage"], dct["sched"], dct["targetvar"], dct["jobs"], dct["seed"], dct["prefix"], dct["suffix"], dct["sps"], dct["nsamp"], dct["shap"], dct["tree"]
rnd = np.random.default_rng(0)
batch_size = int(alg.split("-")[1])
npairs = int(alg.split("-")[2])

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
        # kfolds, kfolds_full = (df.shape[0] - 2, df.shape[0]) if kfolds0 == 0 else (kfolds0, kfolds0)
        d = hdict(algname=alg, npairs=npairs, trials=max_trials, demo=demo, columns=df.columns.tolist()[:-1], shap=shap, seed=seed, _njobs_=jobs, _verbose_=True)

        if tree:  ###############################################################################################################
            best_r2 = best_bacc = -1000
            for batch in range(batches):
                start = batch_size * batch
                end = start + batch_size
                # noinspection PyTypeChecker
                d.apply(tree_optimized_dv_pair, df, start=start, end=end, njobs=_._njobs_, verbose=_._verbose_, out="best")
                if use_cache:
                    d = ch(d, storages)
                r2_params, bacc_params, r2, bacc = d.best
                if r2 > best_r2:
                    best_r2 = r2
                    best_r2_params = r2_params
                if bacc > best_bacc:
                    best_bacc = bacc
                    best_bacc_params = bacc_params
            # noinspection PyTypeChecker
            if r2:
                # noinspection PyTypeChecker
                d.apply(fit, alg, best_r2_params, df, out="tree")
            else:
                # noinspection PyTypeChecker
                d.apply(fit, alg, best_bacc_params, df, out="tree")
            if use_cache:
                d = ch(d, storages)
            best_estimator = d.tree

            print("vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv")
            print(best_r2_params)
            print(best_bacc_params)
            print("--------------------------------")
            print(best_r2, best_bacc)
            print(" ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^")
            columns = df.columns.tolist()[:-1]
            plot_tree(best_estimator, filled=True, feature_names=columns, fontsize=font)
            plt.title(f"Decision tree for {targetvar}")
            plt.show()
            continue

        # L2O ##############################################################################################################
        ansi = d.hosh.ansi
        pairs = pairwise_sample(df.index.tolist(), nsamp, seed)
        best_r2__dct, best_r2_params__dct = {}, {}
        best_bacc__dct, best_bacc_params__dct = {}, {}
        for until_batch in range(batches):
            start = batch_size * until_batch
            end = start + batch_size

            hits = {0: 0, 1: 0}
            tot, errors = {0: 0, 1: 0}, {0: [], 1: []}
            t, z = [], []
            t_diff, z_diff = [], []
            bacc = 0
            shaps = SHAPs()
            tasks = zip(repeat(f"{until_batch}/{batches}:{targetvar} {noage=} {seed=} {sp=}"), repeat(alg), repeat(d.id), pairs)
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

                # optimize  +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                Xw_tr = df.drop([idxa, idxb], axis="rows")
                Xw_ts = np.vstack([babya, babyb])
                # noinspection PyTypeChecker
                d.apply(tree_optimized_dv_pair, Xw_tr, start=start, end=end, njobs=_._njobs_, verbose=_._verbose_, out="best")
                if use_cache:
                    d = ch(d, storages)
                r2_params, bacc_params, r2, bacc = d.best
                if (idxa, idxb) not in best_r2__dct or r2 > best_r2__dct[(idxa, idxb)]:
                    best_r2__dct[(idxa, idxb)] = r2
                    best_r2_params__dct[(idxa, idxb)] = r2_params
                if (idxa, idxb) not in best_bacc__dct or bacc > best_bacc__dct[(idxa, idxb)]:
                    best_bacc__dct[(idxa, idxb)] = bacc
                    best_bacc_params__dct[(idxa, idxb)] = bacc_params
                if not sched:
                    print(f"\r{ansi} {targetvar0, alg0, (idxa, idxb)}: {c:3} {100 * c / len(pairs):4.2f}% {bacc:5.3f}          ", end="", flush=True)

                # fit, predict +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                if use_r2:
                    # noinspection PyTypeChecker
                    d.apply(fitpredict, params=best_r2_params__dct[(idxa, idxb)], Xwtr=Xw_tr, Xts=Xw_ts[:, :-1], out="zts")
                else:
                    # noinspection PyTypeChecker
                    d.apply(fitpredict, params=best_bacc_params__dct[(idxa, idxb)], Xwtr=Xw_tr, Xts=Xw_ts[:, :-1], out="zts")
                if use_cache:
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

                # accumulate +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                pass
                # expected = int(baby_ya[0] + delta >= baby_yb[0])
                # predicted = int(zts[0] + delta >= zts[1])
                # expected = int(abs(baby_ya[0] - baby_yb[0]) >= delta)
                # predicted = int(abs(zts[0] - zts[1]) >= delta)
                expected = int(baby_ya[0] >= baby_yb[0])
                predicted = int(zts[0] >= zts[1])
                # expected = int(baby_ya[0] / baby_yb[0] >= 1 + delta / 100)
                # predicted = int(zts[0] / zts[1] >= 1 + delta / 100)
                t.append(expected)
                t_diff.append(baby_ya[0] - baby_yb[0])
                tot[expected] += 1
                z.append(predicted)
                hits[expected] += int(expected == predicted)
                z_diff.append(zts[0] - zts[1])

                # errors +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                if expected != predicted:
                    errors[expected].append((babydfa, babydfb))

                # temporary accuracy +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                # print(baby_ya, baby_yb, expected, predicted)
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
            if bacc > 0:
                aps = average_precision_score(t, z)
                pr, rc = precision_recall_curve(t, z)[:2]
                auprc = auc(rc, pr)
                r2 = r2_score(t_diff, z_diff)
                tau = kendalltau(t_diff, z_diff)[0]
                pea = correlation(t_diff, z_diff)
                print(f"\r{sp=} {delta=} {hits=} {tot=} \t{d.hosh.ansi} | {bacc=:4.3f} | {aps=:4.3f} | {auprc=:4.3f} | "
                      f"{r2=:4.3f} | {tau=:4.3f} | {pea=:4.3f}", flush=True)

        print("\n")
