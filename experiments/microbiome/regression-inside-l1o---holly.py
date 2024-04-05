from itertools import repeat
from sys import argv

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydotplus
from argvsucks import handle_command_line
from hdict import hdict, _
from numpy import percentile
from pandas import read_csv, DataFrame
from shelchemy import sopen
from shelchemy.scheduler import Scheduler
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import export_graphviz

from germina.aux import fit, fitpredict
from germina.cols import pathway_lst, bacteria_lst, single_eeg_lst, dyadic_eeg_lst
from germina.config import local_cache_uri, remote_cache_uri, near_cache_uri, schedule_uri
from germina.runner import ch
from germina.stats import p_value
from germina.trees import tree_optimized_dv_pair

max_trials = 10_000_000
dct = handle_command_line(argv, z=False, roc=False, source=str, cache=False, font=12, alg=str, demo=False, noage=False, sched=False, targetvar=str, jobs=int, seed=0, prefix=str, suffix=str, sps=list, shap=False, tree=False)
print(dct)
z, roc, source, use_cache, font, alg, demo, noage, sched, targetvar, jobs, seed, prefix, suffix, sps, shap, tree = \
    dct["z"], dct["roc"], dct["source"], dct["cache"], dct["font"], dct["alg"], dct["demo"], dct["noage"], dct["sched"], dct["targetvar"], dct["jobs"], dct["seed"], dct["prefix"], dct["suffix"], dct["sps"], dct["shap"], dct["tree"]
rnd = np.random.default_rng(0)
runs = int(alg.split("-")[1])
npairs = int(alg.split("-")[2])
if source.endswith("eeg"):
    sps = [1]
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
        elif source == "bcpt":
            predictors = bacteria_lst + pathway_lst
        elif source == "seeg":
            predictors = single_eeg_lst
        elif source == "deeg":
            # tanto faz o  target no nome do arquivo para dyadic.
            df_deeg = read_csv(f"data/dyadic_bayley_8_t2.csv", index_col="id_estudo")
            selected0 = list(sorted(set(dyadic_eeg_lst).intersection(df_deeg.columns)))
            df_deeg = df_deeg[selected0]
            # df_deeg.drop([458, 501, 455, 427], axis="rows", inplace=True)
            idx = df_deeg.count(axis="rows").sort_values() > 52  # Accept 10% of babies with NaN for a single variable
            df_deeg = df_deeg.loc[:, idx]
            df = df.join(df_deeg, how="inner")
            predictors = dyadic_eeg_lst
        else:
            raise Exception(f"Unknown {source=}")

        if targetvar.endswith("t1"):
            agevar_lst = ["idade_crianca_dias_t1"]
        elif targetvar.endswith("t2"):
            agevar_lst = ["idade_crianca_dias_t2"]
        elif targetvar.endswith("t3"):
            agevar_lst = ["idade_crianca_dias_t3"]
        elif targetvar.endswith("t4"):
            agevar_lst = ["idade_crianca_dias_t4"]
        elif targetvar.endswith("t42"):
            agevar_lst = ["idade_crianca_dias_t2", "idade_crianca_dias_t4"]
            targetvar_start = targetvar[:-2] + targetvar[-1:]
            targetvar_end = targetvar[:-1]
            df[targetvar] = df[targetvar_end] - df[targetvar_start]
        elif targetvar.endswith("t43"):
            agevar_lst = ["idade_crianca_dias_t3", "idade_crianca_dias_t4"]
            targetvar_start = targetvar[:-2] + targetvar[-1:]
            targetvar_end = targetvar[:-1]
            df[targetvar] = df[targetvar_end] - df[targetvar_start]
        else:
            raise Exception(f"Unexpected timepoint suffix for target '{targetvar}'.")

        selected = list(sorted(set(predictors).intersection(df.columns))) + agevar_lst + [targetvar]
        df = df[selected]
        print("with NaNs", df.shape)
        df.dropna(axis="rows", inplace=True)
        print(df.shape)
        print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^")

        df.sort_values(targetvar, inplace=True, ascending=True, kind="stable")
        if demo:
            take = min(df.shape[0] // 2, 30)
            df = pd.concat([df.iloc[:take], df.iloc[-take:]], axis="rows")
        if noage:
            del df[agevar_lst]
        print(df.shape)
        # kfolds, kfolds_full = (df.shape[0] - 2, df.shape[0]) if kfolds0 == 0 else (kfolds0, kfolds0)
        d = hdict(algname=alg, npairs=npairs, trials=max_trials, demo=demo, columns=df.columns.tolist()[:-1], shap=shap, seed=seed, _njobs_=jobs, _verbose_=True)

        if tree:  ###############################################################################################################
            best_bacc = -1000
            if z:
                ss = StandardScaler()
                df = DataFrame(ss.fit_transform(df), index=df.index, columns=df.columns)
            # noinspection PyTypeChecker
            d.apply(tree_optimized_dv_pair, df, start=0, end=runs, njobs=_._njobs_, verbose=_._verbose_, out="best")
            Xtr = df
            if use_cache:
                d = ch(d, storages)
            r2_params, bacc_params, r2, bacc_ = d.best
            if bacc_ > best_bacc:
                best_bacc = bacc_
                best_bacc_params = bacc_params
            # noinspection PyTypeChecker
            d.apply(fit, alg, best_bacc_params, Xtr, out="tree")
            if use_cache:
                d = ch(d, storages)
            best_estimator = d.tree

            print("vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv")
            print(best_bacc_params)
            print("--------------------------------")
            print(best_bacc)
            print(" ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^")
            columns = df.columns.tolist()[:-1]
            filename = f"{targetvar}_{sp=}_{alg}_{noage=:1}__{source}"

            # plot_tree(best_estimator, filled=True, feature_names=columns, fontsize=font)
            # plt.title(arq)
            # plt.show()

            dot_data = export_graphviz(best_estimator, feature_names=columns, out_file=None, filled=True, rounded=True)
            pydot_graph = pydotplus.graph_from_dot_data(dot_data)
            pydot_graph.write_pdf(f"tree/tree___{filename}.pdf")
            continue

        # L2O ##############################################################################################################
        ansi = d.hosh.ansi
        best_bacc, best_bacc_params = -1, None

        hits_low_vs_nextnorm = {0: 0, 1: 0}
        hits_lownext_vs_norm = {0: 0, 1: 0}
        hits_low_vs_next_vs_norm = {0: 0, 1: 0, 2: 0}

        tot_low_vs_nextnorm = {0: 0, 1: 0}
        tot_lownext_vs_norm = {0: 0, 1: 0}
        tot_low_vs_next_vs_norm = {0: 0, 1: 0, 2: 0}

        bacc_low_vs_nextnorm = 0
        bacc_lownext_vs_norm = 0
        bacc_low_vs_next_vs_norm = 0
        p_low_vs_nextnorm = 0
        p_lownext_vs_norm = 0
        p_low_vs_next_vs_norm = 0

        vts_lst, wts_lst = [], []  # continuous (expected, predicted)
        yts_lst, zts_lst = [], []  # binary (expected, predicted)
        predicted_score_lst = []

        expected_label_lst__low_vs_nextnorm = []
        predicted_label_lst__low_vs_nextnorm = []

        expected_label_lst__lownext_vs_norm = []
        predicted_score_lst__lownext_vs_norm = []
        predicted_label_lst__lownext_vs_norm = []

        predicted_label_lst__low_vs_next_vs_norm = []
        expected_label_lst__low_vs_next_vs_norm = []
        predicted_score_lst__low_vs_next_vs_norm = []

        ys = []

        tasks = zip(repeat(f"{targetvar} {noage=:1} {seed=} {sp=} {source=}"), repeat(alg), repeat(d.id), df.index.tolist())
        for c, (targetvar0, alg0, did0, idx) in enumerate((Scheduler(db, timeout=60) << tasks) if sched else tasks):
            if not sched:
                print(f"\r{ansi} {targetvar0, alg0, idx}: {c:3} {100 * c / df.shape[0]:4.2f}% {bacc_lownext_vs_norm:5.3f} {p_lownext_vs_norm:.3f}\t", end="", flush=True)

            # prepare current pair of babies and training set
            baby_df = df.loc[[idx], :]
            baby_v_df = baby_df.iloc[0, -1:]
            if baby_v_df.isna().sum().sum() > 0:
                continue  # skip NaN labels from testing set

            # Build sets  +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            Xv_tr = df.drop([idx], axis="rows")
            Xv_ts = baby_df.to_numpy()
            if z:
                ss = StandardScaler()
                Xv_tr[targetvar] = ss.fit_transform(Xv_tr[targetvar].to_numpy().reshape(-1, 1))
                Xv_ts[:, -1:] = ss.transform(Xv_ts[:, -1:])
            ys.append(Xv_ts[0, -1])

            # optimize  +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            # noinspection PyTypeChecker
            d.apply(tree_optimized_dv_pair, Xv_tr, start=0, end=runs, njobs=_._njobs_, verbose=_._verbose_, out="best")
            if use_cache:
                d = ch(d, storages)
            r2_params, bacc_params, r2, bacc_ = d.best
            if bacc_ > best_bacc:
                best_bacc = bacc_
                best_bacc_params = bacc_params
            if not sched:
                print(f"\r{ansi} {targetvar0, alg0, idx}: {c:3} {100 * c / df.shape[0]:4.2f}% {bacc_lownext_vs_norm:5.3f} {p_lownext_vs_norm:.3f}\t", end="", flush=True)

            # fit, predict +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            best_params = best_bacc_params
            # noinspection PyTypeChecker
            d.apply(fitpredict, params=best_params, Xwtr=Xv_tr, Xts=Xv_ts[:, :-1], out="wts")
            if use_cache:
                d = ch(d, storages)
            wts = d.wts
            if not sched:
                print(f"\r{ansi} {targetvar0, alg0, idx}: {c:3} {100 * c / df.shape[0]:4.2f}% {bacc_lownext_vs_norm:5.3f} {p_lownext_vs_norm:.3f}\t", end="", flush=True)

            if sched:
                continue

            # accumulate +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            # me, mn, mx = np.mean(v), np.min(v), np.max(v)
            Xv_np = Xv_tr.to_numpy()
            v = Xv_np[:, -1]
            predicted_reg = wts[0]
            predicted_score_lst.append(-predicted_reg)

            for pct, tot, hits in [
                [15, tot_low_vs_nextnorm, hits_low_vs_nextnorm],
                [30, tot_lownext_vs_norm, hits_lownext_vs_norm],
            ]:
                cut = percentile(v, pct)
                expected_label = int(Xv_ts[0, -1] <= cut)
                predicted_label = int(predicted_reg <= cut)
                if pct == 15:
                    expected_label_lst__low_vs_nextnorm.append(expected_label)
                    predicted_label_lst__low_vs_nextnorm.append(predicted_label)
                else:
                    expected_label_lst__lownext_vs_norm.append(expected_label)
                    predicted_label_lst__lownext_vs_norm.append(predicted_label)
                tot[expected_label] += 1
                hits[predicted_label] += int(predicted_label == expected_label)
                if tot[0] * tot[1] > 0:
                    acc0 = hits[0] / tot[0]
                    acc1 = hits[1] / tot[1]
                    if pct == 15:
                        bacc_low_vs_nextnorm = (acc0 + acc1) / 2
                        p_low_vs_nextnorm = p_value(bacc_low_vs_nextnorm, c + 1)
                    else:
                        bacc_lownext_vs_norm = (acc0 + acc1) / 2
                        p_lownext_vs_norm = p_value(bacc_lownext_vs_norm, c + 1)

            # 3 classes
            tot = tot_low_vs_next_vs_norm
            hits = hits_low_vs_next_vs_norm
            cut1 = percentile(v, 15)
            cut2 = percentile(v, 30)
            expected_label = 0 if Xv_ts[0, -1] <= cut1 else (1 if Xv_ts[0, -1] <= cut2 else 2)
            predicted_label = 0 if predicted_reg <= cut1 else (1 if predicted_reg <= cut2 else 2)
            expected_label_lst__low_vs_next_vs_norm.append(expected_label)
            predicted_label_lst__low_vs_next_vs_norm.append(predicted_label)
            tot[expected_label] += 1
            hits[predicted_label] += int(predicted_label == expected_label)
            if tot[0] * tot[1] * tot[2] > 0:
                acc0 = hits[0] / tot[0]
                acc1 = hits[1] / tot[1]
                acc2 = hits[2] / tot[2]
                bacc_low_vs_next_vs_norm = (acc0 + acc1 + acc2) / 3
                p_low_vs_next_vs_norm = p_value(bacc_low_vs_next_vs_norm, c + 1)

        if sched:
            continue

        print()
        t = tot_low_vs_next_vs_norm
        h = hits_low_vs_next_vs_norm
        print(f"\r{targetvar}\t{source=} {sp=} {t=} {h=}"
              f"\n{bacc_low_vs_nextnorm=:.2f} {p_low_vs_nextnorm=:.3f}"
              f"\t{bacc_lownext_vs_norm=:.2f} {p_lownext_vs_norm=:.3f} "
              f"{r2=:.2f} ", flush=True)

        #     ROC
        if roc:
            title = f"ROC {source} at T{sp} → {targetvar}\nLo×NeNo:{bacc_low_vs_nextnorm:.2f}   LoNe×No{bacc_lownext_vs_norm:.2f}   Lo×Ne×No:{bacc_low_vs_next_vs_norm:.2f}"
            plt.title(title)
            plt.plot([0, 1], [0, 1])
            for text, bacc, expected_labels in [
                [f"LOW × NEXT+NORM ", bacc_low_vs_nextnorm, expected_label_lst__low_vs_nextnorm],
                [f"LOW+NEXT × NORM ", bacc_lownext_vs_norm, expected_label_lst__lownext_vs_norm],
                # [f"LOW × NEXT × NORM ({AUC=})", bacc_low_vs_next_vs_norm, expected_label_lst__low_vs_next_vs_norm, predicted_score_lst__low_vs_next_vs_norm],
            ]:
                # print(ys)
                # print(expected_labels)
                # print(predicted_score_lst)
                # print()
                fpr, tpr, thresh = roc_curve(expected_labels, predicted_score_lst)
                AUC = roc_auc_score(expected_labels, predicted_score_lst)
                plt.plot(fpr, tpr, label=text + f"({AUC=:.2f})")
            filename = title.split('\n')[0].replace(' ', '-')
            plt.legend(loc=0)
            plt.savefig(f"results/{filename}.pdf")
            plt.clf()

        print("\n")
