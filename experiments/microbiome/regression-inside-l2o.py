from itertools import repeat
from statistics import correlation
from sys import argv

import numpy as np
import pandas as pd
from argvsucks import handle_command_line
from hdict import hdict, _
from pairwiseprediction.combination import pairwise_diff
from pairwiseprediction.combination import pairwise_hstack
from pandas import read_csv, DataFrame
from scipy.stats import kendalltau
from shelchemy import sopen
from shelchemy.scheduler import Scheduler
from sklearn.metrics import average_precision_score, r2_score
from sklearn.metrics import precision_recall_curve, auc
from sklearn.tree import plot_tree
from sympy.physics.control.control_plots import plt

from germina.aux import fit, fitpredict
from germina.cols import pathway_lst, bacteria_lst, single_eeg_lst
from germina.config import local_cache_uri, remote_cache_uri, near_cache_uri, schedule_uri
from germina.runner import ch
from germina.sampling import pairwise_sample
from germina.shaps import SHAPs
from germina.trees import tree_optimized_dv_pair, tree_optimized_dv_pairdiff
from sklearn.tree import export_graphviz
import pydotplus

center, max_trials = None, 10_000_000
dct = handle_command_line(argv, source=str, r2=False, batches=int, cache=False, font=12, alg=str, demo=False, delta=float, noage=False, sched=False, targetvar=str, jobs=int, seed=0, prefix=str, suffix=str, sps=list, nsamp=int, shap=False, tree=False, diff=False)
print(dct)
source, use_r2, batches, use_cache, font, alg, demo, delta, noage, sched, targetvar, jobs, seed, prefix, suffix, sps, nsamp, shap, tree, diff = \
    dct["source"], dct["r2"], dct["batches"], dct["cache"], dct["font"], dct["alg"], dct["demo"], dct["delta"], dct["noage"], dct["sched"], dct["targetvar"], dct["jobs"], dct["seed"], dct["prefix"], dct["suffix"], dct["sps"], dct["nsamp"], dct["shap"], dct["tree"], dct["diff"]
if targetvar.endswith("t1"):
    agevar = "idade_crianca_dias_t1"
elif targetvar.endswith("t2"):
    agevar = "idade_crianca_dias_t2"
elif targetvar.endswith("t3"):
    agevar = "idade_crianca_dias_t3"
elif targetvar.endswith("t4"):
    agevar = "idade_crianca_dias_t4"
else:
    raise Exception(f"Unexpected timepoint suffix for target '{targetvar}'.")
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

        if source == "bact":
            predictors = bacteria_lst
        elif source == "path":
            predictors = pathway_lst
        elif source == "seeg":
            predictors = single_eeg_lst
        elif source == "deeg":
            predictors = dyadic_eeg_lst
        else:
            raise Exception(f"Unknown {source=}")
        df = df[predictors + [agevar, targetvar]]

        df.sort_values(targetvar, inplace=True, ascending=True, kind="stable")
        if demo:
            take = min(df.shape[0] // 2, 30)
            df = pd.concat([df.iloc[:take], df.iloc[-take:]], axis="rows")
        if noage:
            del df[agevar]
        print(df.shape)
        # kfolds, kfolds_full = (df.shape[0] - 2, df.shape[0]) if kfolds0 == 0 else (kfolds0, kfolds0)
        d = hdict(delta=delta, algname=alg, npairs=npairs, trials=max_trials, demo=demo, columns=df.columns.tolist()[:-1], shap=shap, seed=seed, _njobs_=jobs, _verbose_=True)

        if tree:  ###############################################################################################################
            best_r2 = best_bacc = -1000
            for batch in range(batches):
                start = batch_size * batch
                end = start + batch_size
                if diff:
                    Xv = df.to_numpy()
                    Xd = pairwise_diff(Xv, Xv)
                    idx = df.index.to_numpy().reshape(df.shape[0], -1)
                    Xd_idxs = pairwise_hstack(idx, idx)
                    indexed_Xd = np.hstack([Xd, Xd_idxs])
                    indexed_Xd = indexed_Xd[np.any(indexed_Xd[:, :-2] != 0, axis=1)]  # remove null rows
                    rnd = np.random.default_rng(seed)
                    rnd.shuffle(indexed_Xd)
                    indexed_Xd = indexed_Xd[:npairs]
                    # noinspection PyTypeChecker
                    d.apply(tree_optimized_dv_pairdiff, df, indexed_Xd, start=start, end=end, njobs=_._njobs_, verbose=_._verbose_, out="best")
                    Xtr = DataFrame(Xd[:npairs], columns=df.columns)
                else:
                    # noinspection PyTypeChecker
                    d.apply(tree_optimized_dv_pair, df, start=start, end=end, njobs=_._njobs_, verbose=_._verbose_, out="best")
                    Xtr = df
                if use_cache:
                    d = ch(d, storages)
                r2_params, bacc_params, r2, bacc_ = d.best
                if r2 > best_r2:
                    best_r2 = r2
                    best_r2_params = r2_params
                if bacc_ > best_bacc:
                    best_bacc = bacc_
                    best_bacc_params = bacc_params
            # noinspection PyTypeChecker
            if r2:
                # noinspection PyTypeChecker
                d.apply(fit, alg, best_r2_params, Xtr, out="tree")
            else:
                # noinspection PyTypeChecker
                d.apply(fit, alg, best_bacc_params, Xtr, out="tree")
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
            arq = f"{targetvar}_species{sp}_{alg}_{batches=}_{noage=:1}_{delta=}_{use_r2=:1}_{diff=:1}"

            # plot_tree(best_estimator, filled=True, feature_names=columns, fontsize=font)
            # plt.title(arq)
            # plt.show()

            dot_data = export_graphviz(best_estimator, feature_names=columns, out_file=None, filled=True, rounded=True)
            pydot_graph = pydotplus.graph_from_dot_data(dot_data)
            pydot_graph.write_pdf(f"tree/tree___{arq}.pdf")
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
            vts_lst, wts_lst = [], []  # continuous
            vts_diff, wts_diff = [], []
            yts_lst, zts_lst = [], []  # binary
            bacc = 0
            shaps = SHAPs()
            tasks = zip(repeat(f"{until_batch}/{batches}:{targetvar} {noage=:1} {delta=} {use_r2=:1} {seed=} {sp=} {diff=:1}"), repeat(alg), repeat(d.id), pairs)
            for c, (targetvar0, alg0, did0, (idxa, idxb)) in enumerate((Scheduler(db, timeout=60) << tasks) if sched else tasks):
                if not sched:
                    print(f"\r{ansi} {targetvar0, alg0, (idxa, idxb)}: {c:3} {100 * c / len(pairs):4.2f}% {bacc:5.3f}          ", end="", flush=True)

                # prepare current pair of babies and training set
                babydfa = df.loc[[idxa], :]
                babydfb = df.loc[[idxb], :]
                baby_va = babydfa.iloc[0, -1:]
                baby_vb = babydfb.iloc[0, -1:]
                if baby_va.isna().sum().sum() > 0 or baby_vb.isna().sum().sum() > 0:
                    continue  # skip NaN labels from testing set
                baby_va = baby_va.to_numpy()
                baby_vb = baby_vb.to_numpy()
                babya = babydfa.to_numpy()
                babyb = babydfb.to_numpy()

                # optimize  +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                Xv_tr = df.drop([idxa, idxb], axis="rows")
                Xv_ts = np.vstack([babya, babyb])
                if diff:
                    Xv = df.to_numpy()
                    Xd = pairwise_diff(Xv, Xv)
                    idx = df.index.to_numpy().reshape(df.shape[0], -1)
                    Xd_idxs = pairwise_hstack(idx, idx)
                    indexed_Xd = np.hstack([Xd, Xd_idxs])
                    Xd = Xd_idxs = None
                    indexed_Xd = indexed_Xd[np.any(indexed_Xd[:, :-2] != 0, axis=1)]  # remove null rows
                    rnd = np.random.default_rng(seed)
                    rnd.shuffle(indexed_Xd)
                    indexed_Xd = indexed_Xd[:npairs]
                    # noinspection PyTypeChecker
                    d.apply(tree_optimized_dv_pairdiff, Xv_tr, indexed_Xd, delta, start=start, end=end, njobs=_._njobs_, verbose=_._verbose_, out="best")
                else:
                    # noinspection PyTypeChecker
                    d.apply(tree_optimized_dv_pair, Xv_tr, start=start, end=end, njobs=_._njobs_, verbose=_._verbose_, out="best")
                if use_cache:
                    d = ch(d, storages)
                r2_params, bacc_params, r2, bacc_ = d.best
                if (idxa, idxb) not in best_r2__dct or r2 > best_r2__dct[(idxa, idxb)]:
                    best_r2__dct[(idxa, idxb)] = r2
                    best_r2_params__dct[(idxa, idxb)] = r2_params
                if (idxa, idxb) not in best_bacc__dct or bacc_ > best_bacc__dct[(idxa, idxb)]:
                    best_bacc__dct[(idxa, idxb)] = bacc_
                    best_bacc_params__dct[(idxa, idxb)] = bacc_params
                if not sched:
                    print(f"\r{ansi} {targetvar0, alg0, (idxa, idxb)}: {c:3} {100 * c / len(pairs):4.2f}% {bacc:5.3f}          ", end="", flush=True)

                # fit, predict +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                # remove rows containing forbidden indexes
                if diff:
                    allowed = (indexed_Xd[:, -2] != idxa) & (indexed_Xd[:, -1] != idxb) & (indexed_Xd[:, -2] != idxb) & (indexed_Xd[:, -1] != idxa)
                    Xv_tr = indexed_Xd[allowed, :-2]
                    Xv_tr = DataFrame(Xv_tr, columns=df.columns)
                    Xv_ts = babya - babyb
                if use_r2:
                    # noinspection PyTypeChecker
                    d.apply(fitpredict, params=best_r2_params__dct[(idxa, idxb)], Xwtr=Xv_tr, Xts=Xv_ts[:, :-1], out="wts")
                else:
                    # noinspection PyTypeChecker
                    d.apply(fitpredict, params=best_bacc_params__dct[(idxa, idxb)], Xwtr=Xv_tr, Xts=Xv_ts[:, :-1], out="wts")
                if use_cache:
                    d = ch(d, storages)
                wts = d.wts
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
                expected = int(baby_va[0] - baby_vb[0] >= delta)
                predicted = int(wts[0] >= delta) if diff else int(wts[0] - wts[1] >= delta)
                # expected = int(baby_ya[0] / baby_yb[0] >= 1 + delta / 100)
                # predicted = int(zts[0] / zts[1] >= 1 + delta / 100)
                vts_lst.extend([baby_va[0]] if diff else [baby_va[0], baby_vb[0]])
                wts_lst.extend([baby_vb[0] + wts[0]] if diff else [wts[0], wts[1]])
                yts_lst.append(expected)
                zts_lst.append(predicted)
                tot[expected] += 1
                hits[expected] += int(expected == predicted)
                vts_diff.append(baby_va[0] - baby_vb[0])
                wts_diff.append(wts[0] if diff else (wts[0] - wts[1]))

                # errors +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                if expected != predicted:
                    errors[expected].append((babydfa, babydfb))

                # temporary accuracy +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                # print(baby_ya, baby_yb, expected, predicted)
                if tot[0] * tot[1] > 0:
                    acc0 = hits[0] / tot[0]
                    acc1 = hits[1] / tot[1]
                    bacc = (acc0 + acc1) / 2
                    # print(f"{bacc:3.3f}, {tot}, {hits}")

            if sched:
                continue

            # classification
            if tot[0] == 0 or tot[1] == 0:
                print(f"Resulted in class total with zero value: {tot=}")
                bacc = -1

            # precision_recall_curve
            if bacc > 0:
                aps = average_precision_score(yts_lst, zts_lst)
                pr, rc = precision_recall_curve(yts_lst, zts_lst)[:2]
                auprc = auc(rc, pr)
                r2 = r2_score(vts_lst, wts_lst)
                tau = [round(a, 4) for a in kendalltau(vts_lst, wts_lst)]
                pea = correlation(vts_lst, wts_lst)

                r2_d = r2_score(vts_diff, wts_diff)
                tau_d = [round(a, 4) for a in kendalltau(vts_diff, wts_diff)]
                pea_d = correlation(vts_diff, wts_diff)
                print(f"\r{sp=} {delta=} {hits=} {tot=} \t{d.hosh.ansi} | "
                      f"{bacc=:4.3f} | {aps=:4.3f} | {auprc=:4.3f} | "
                      f"{r2=:4.3f} {tau=} {pea=:4.3f} | "
                      f"{r2_d=:4.3f} {tau_d=} {pea_d=:4.3f}", flush=True)

        print("\n")
