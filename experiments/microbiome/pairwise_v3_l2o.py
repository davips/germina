from sys import argv
from hdict import hdict, _
import numpy as np
import pandas as pd
from argvsucks import handle_command_line
from lange import ap
from pairwiseprediction.classifier import PairwiseClassifier
from pandas import read_csv
from scipy.stats import ttest_1samp
from shelchemy import sopen
from sklearn.model_selection import RandomizedSearchCV
from sklearn.tree import plot_tree
from sortedness.embedding import balanced
from sympy.physics.control.control_plots import plt
from torch import from_numpy

from germina.config import local_cache_uri, remote_cache_uri, near_cache_uri, schedule_uri
from germina.l2o import loo, predictors
from germina.runner import ch


def f(alg, seed, jobs, pairwise, delta, proportion=False, center=None, only_relevant_pairs_on_prediction=False, verbose=False):
    if verbose:
        print("Optmizing hyperparameters.")
    Opt: RandomizedSearchCV
    Opt, kwargs = predictors(alg, None, seed, jobs)
    alg_c = PairwiseClassifier(Opt, pairwise, delta, proportion=proportion, center=center, only_relevant_pairs_on_prediction=only_relevant_pairs_on_prediction, **kwargs)
    alg_c.fit(df.iloc[:, :-1])
    opt = alg_c._estimator
    return opt.best_estimator_, opt.best_params_, opt.best_score_, opt.cv_results_


def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results["rank_test_score"] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print(
                "Mean validation score: {0:.3f} (std: {1:.3f})".format(
                    results["mean_test_score"][candidate],
                    results["std_test_score"][candidate],
                )
            )
            print("Parameters: {0}".format(results["params"][candidate]))
            print("")


dct = handle_command_line(argv, noage=False, delta=float, trees=int, pct=False, demo=False, sched=False, perms=1, targetvar=str, jobs=int, alg=str, seed=0, prefix=str, sufix=str, trees_imp=int, feats=int, tfsel=int, forward=False, pairwise=str, sps=list, plot=False, nsamp=int, shap=False, tree=False)
print(dct)
noage, trees, delta, pct, demo, sched, perms, targetvar, jobs, alg, seed, prefix, sufix, trees_imp, feats, tfsel, forward, pairwise, sps, plot, nsamp, shap, tree = dct["noage"], dct["trees"], dct["delta"], dct["pct"], dct["demo"], dct["sched"], dct["perms"], dct["targetvar"], dct["jobs"], dct["alg"], dct["seed"], dct["prefix"], dct["sufix"], dct["trees_imp"], dct["feats"], dct["tfsel"], dct["forward"], dct["pairwise"], dct["sps"], dct["plot"], dct["nsamp"], dct["shap"], dct["tree"]
rnd = np.random.default_rng(0)
with (sopen(local_cache_uri, ondup="skip") as local_storage, sopen(near_cache_uri, ondup="skip") as near_storage, sopen(remote_cache_uri, ondup="skip") as remote_storage, sopen(schedule_uri) as db):
    storages = {
        # "remote": remote_storage,
        "near": near_storage,
        "local": local_storage,
    }
    for sp in sps:
        sp = int(sp)
        print(f"{sp=} <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
        df = read_csv(f"{prefix}{sp}{sufix}", index_col="id_estudo")
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
        # age = df[agevar]
        if noage:
            del df[agevar]  #####################################
            # df = df[["idade_crianca_dias_t2", "bayley_8_t2"]]
            # print(df.shape, "<<<<<<<<<<<<<<<<<")

        if tree:
            hd = hdict(_verbose_=True)
            # noinspection PyTypeChecker
            hd.apply(f, alg, seed, jobs, pairwise, delta, verbose=_._verbose_, out="tree")
            hd = ch(hd, storages)
            best_estimator, best_params, best_score, cv_results = hd.tree
            report(cv_results)
            print(best_params)
            print(best_score)
            plot_tree(best_estimator, filled=True)
            plt.title(f"Decision tree for {targetvar}")
            plt.show()
            # exit()
            continue

        ret = loo(df, permutation=0, pairwise=pairwise, threshold=delta,
                  alg=alg, n_estimators=trees,
                  n_estimators_imp=trees_imp,
                  n_estimators_fsel=tfsel, forward_fsel=forward, k_features_fsel=feats, k_folds_fsel=4,
                  db=db, storages=storages, sched=sched, shap=shap,
                  nsamp=nsamp, seed=seed, jobs=jobs)
        if ret:
            d, bacc, hits, tot, errors, aps, auprc, shaps = ret
            print(f"\r{sp=} {delta=} {trees=} {hits=} {tot=} \t{d.hosh.ansi} | {bacc=:4.3f} | {aps=} | {auprc=} ", flush=True)
            if shap:
                print(f"|||||||||||||||||||||||{sum(tot.values())} pairs\t|||||||||||||||||||||||")
                rel = shaps.relevance()
                rel = rel[(rel["toshap__p-value"] < 0.1) & (rel["toshap__mean"] > 0.0001)]
                rel.sort_values("toshap__mean", inplace=True, kind="stable", ascending=False)
                print(rel)
            if plot:
                missed = set()
                m0, model, epoch, quality_surrogate = balanced(df.to_numpy()[:, :-1], epochs=100, return_only_X_=False).values()
                for label, lst in errors.items():
                    for babydfa, babydfb in lst:
                        missed.add(babydfa.index.item())
                        missed.add(babydfb.index.item())
                        a, b = babydfa.to_numpy()[:, :-1], babydfb.to_numpy()[:, :-1]
                        m = model(from_numpy(np.vstack([a, b]).astype(np.float32))).detach().numpy()
                        plt.plot(m[:, 0], m[:, 1], 'o-', c="red" if label else "blue", alpha=0.05)
                # noinspection PyTypeChecker
                m1 = df.drop(missed, axis="rows").to_numpy()
                m = model(from_numpy(m1[:, :-1].astype(np.float32))).detach().numpy()
                mx, mn = m1[:, -1].max(), m1[:, -1].min()
                ampl = mx - mn
                ss = 500 * (0.01 + m1[:, -1] - mn) / ampl
                plt.scatter(m[:, 0], m[:, 1], c=((ss / 500) - 0.01) * ampl + mn, alpha=0.95, s=ss)
                plt.colorbar()
                plt.show()

        # permutation test
        scores_dct = dict(bacc_c=[], bacc_r=[], r2_c=[], r2_r=[])
        for permutation in ap[1, 2, ..., perms]:
            df_shuffled = df.copy()
            df_shuffled[targetvar] = rnd.permutation(df[targetvar].values)
            ret = loo(df_shuffled, permutation, pairwise=pairwise, threshold=delta,
                      alg=alg, n_estimators=trees,
                      n_estimators_imp=trees_imp,
                      n_estimators_fsel=tfsel, forward_fsel=forward, k_features_fsel=feats, k_folds_fsel=4,
                      db=db, storages=storages, sched=sched, shap=shap,
                      nsamp=nsamp, seed=seed, jobs=jobs)
            if ret:
                d, bacc_cp, hits_cp, totp, tot_cp, shp_c, errors, prc = ret
                # scores_dct["bacc_c"].append(bacc_cp - bacc_c)
                # scores_dct["bacc_r"].append(bacc_rp - bacc_r)
                # scores_dct["r2_c"].append(r2_cp - r2_c)
                # scores_dct["r2_r"].append(r2_rp - r2_r)

        if sched:
            print("Run again without providing flag `sched`.")
            continue

        if scores_dct["bacc_c"]:
            print(f"\n{sp=} p-values: ", end="")
            for measure, scores in scores_dct.items():
                p = ttest_1samp(scores, popmean=0, alternative="greater")[1]
                print(f"  {measure}={p:4.3f}", end="")
        print("\n")
