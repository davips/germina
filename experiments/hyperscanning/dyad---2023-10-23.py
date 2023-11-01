import warnings
from datetime import datetime
from itertools import repeat

from germina.nan import remove_cols
from pandas import DataFrame
from shelchemy.scheduler import Scheduler
from sklearn import clone
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.inspection import permutation_importance
from xgboost import XGBClassifier

from pprint import pprint
from sys import argv

from argvsucks import handle_command_line
from germina.runner import ch
from hdict import hdict, apply, _, field
from shelchemy import sopen

from germina.config import local_cache_uri, remote_cache_uri, near_cache_uri, schedule_uri
from germina.dataset import join, osf_except_target_vars__no_t, eeg_t2_vars
from germina.loader import load_from_csv, clean_for_dalex, get_balance, train_xgb, build_explainer, explain_modelparts, explain_predictparts, importances, load_from_osf, load_from_synapse

from sklearn.model_selection import LeaveOneOut, permutation_test_score, StratifiedKFold, KFold
import warnings
from sklearn.ensemble import RandomForestRegressor as RFr

warnings.filterwarnings('ignore')
__ = enable_iterative_imputer
dct = handle_command_line(argv, pvalruns=int, importanceruns=int, imputertrees=int, seed=int, target=str, trees=int, vif=False, nans=False, sched=False, up="", measures=list, targets=list)
print(datetime.now())
pprint(dct, sort_dicts=False)
print()
path = "data/"
d = hdict(
    n_permutations=dct["pvalruns"],
    n_repeats=dct["importanceruns"],
    imputation_trees=dct["imputertrees"],
    random_state=dct["seed"],
    target=dct["target"],
    measures=dct["measures"], targets=dct["targets"],
    max_iter=dct["trees"], n_estimators=dct["trees"],
    n_splits=5,
    shuffle=True,
    index="id_estudo", join="inner", n_jobs=-1, return_name=False,
    osf_filename="germina-osf-request---davi121023"
)
cfg = hdict(d)
for noncfg in ["index", "join", "n_jobs", "return_name", "osf_filename"]:
    del cfg[noncfg]
vif, nans, sched, storage_to_be_updated = dct["vif"], dct["nans"], dct["sched"], dct["up"]
with (sopen(local_cache_uri) as local_storage, sopen(near_cache_uri) as near_storage, sopen(remote_cache_uri) as remote_storage, sopen(schedule_uri) as db):
    storages = {
        "remote": remote_storage,
        "near": near_storage,
        "local": local_storage,
    }

    d = d >> apply(StratifiedKFold).cv
    res = {}
    res_importances = {}
    for measure in d.measures:
        res[measure] = {"description": [], "score": [], "p-value": []}
        res_importances[measure] = {"description": [], "variable": [], "importance-mean": [], "importance-stdev": []}
    d["res_importances"] = res_importances

    d = load_from_synapse(d, storages, storage_to_be_updated, path, vif, "synapse/EEG-september-nosensorvars", "Xdyadic")
    d = d >> apply(lambda Xdyadic: Xdyadic.dropna()).Xdyadic
    print(f"Removed NaNs  ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓ {d.Xdyadic.shape} ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑")
    d = load_from_csv(d, storages, storage_to_be_updated, path, vif, d.osf_filename, "single", transpose=False, vars=eeg_t2_vars)
    d = ch(d, storages, storage_to_be_updated)

    print("Separate subset from dataset 'EEG single'  --------------------------------------------------------------------------------------------------------------------------------------------------------")
    d = d >> apply(lambda single, Xdyadic: single.loc[Xdyadic.index]).single_small
    d = d >> apply(lambda single, Xdyadic: single.loc[~single.index.isin(Xdyadic.index)]).single_large
    d = ch(d, storages, storage_to_be_updated)

    print(datetime.now(), f"Model imputation {d.n_estimators=} {d.imputation_trees=}--------------------------------------------------------------------------------------------------------------------------------------------------------")
    d = d >> apply(RFr, n_estimators=_.imputation_trees).imputation_alg
    d = d >> apply(lambda imputation_alg, single_large: IterativeImputer(estimator=clone(imputation_alg)).fit(X=single_large)).imputer
    d = ch(d, storages, storage_to_be_updated)

    print(datetime.now(), f"Impute missing values for single EEG small -----------------------------------------------------------------------------------------------------------")
    d = d >> apply(lambda imputer, single_small: DataFrame(imputer.transform(X=single_small), index=single_small.index, columns=single_small.columns)).Xsingle
    d = ch(d, storages, storage_to_be_updated)
    print(f"X {d.Xsingle.shape} ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓\n", d.Xsingle, "↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑\n")
    print(d.Xsingle)

    tasks = zip(repeat((cfg.hosh, f"{vif=}")), d.targets)
    for (h, vi), target in (Scheduler(db, timeout=50) << tasks) if sched else tasks:
        if not sched:
            print(f"\t{h.ansi}\t{vi}\t{target}\t", datetime.now(), f"\t-----------------------------------")

        d = load_from_csv(d, storages, storage_to_be_updated, path, vif, d.osf_filename, "y", transpose=False, vars=[target], verbose=False)
        d = d >> apply(lambda y, Xdyadic: y.loc[Xdyadic.index].dropna()).y
        d = d >> apply(lambda y: (y > y.median()).astype(int)).y
        d = ch(d, storages, storage_to_be_updated)

        for Xvar in ["Xsingle", "Xdyadic"]:
            d = d >> apply(lambda X, y: X.loc[X.index.isin(y.index)], _[Xvar]).X
            d = get_balance(d, storages, storage_to_be_updated)
            d = d >> apply(KFold).cv
            constructors = {"RFr": RFr}
            for k, constructor in constructors.items():
                d = d >> apply(constructor).alg
                d = ch(d, storages, storage_to_be_updated)

                for m in ["r2"]:
                    d["scoring"] = m
                    rets = [f"{m}_scores", f"{m}_permscores", f"{m}_pval"]
                    d = d >> apply(permutation_test_score, _.alg)(*rets)
                    d = ch(d, storages, storage_to_be_updated)
                    print(f"{m:20} (p-value):\t{d[rets[0]]:.4f} ({d[rets[2]]:.4f})")

                    # # Importances
                    # d = d >> apply(lambda alg, X, y: clone(alg).fit(X, y)).estimator
                    # d = d >> apply(permutation_importance).importances
                    # d = ch(d, storages, storage_to_be_updated)
                    # r = d.importances
                    # for i in r.importances_mean.argsort()[::-1]:
                    #     if r.importances_mean[i] - r.importances_std[i] > 0:
                    #         print(f"importance   \t                 \t{r.importances_mean[i]:.6f}\t{r.importances_std[i]:.6f}\t{m:22}\t{d.target_var:20}\t{d.X.columns[i]}")
                    # print()

    print("All finished")

    #
    #     print("Induce classifier -------------------------------------------------------------------------------------------------------------------------------------------------------")
    #     print(datetime.now())
    #     d = d >> apply(StratifiedKFold).cv
    #     taskmark = d.hosh - (d.hoshes["n_jobs"] * b"n_jobs")
    #     constructors = {"HGBc": HGBc, "RFc": RFc, "XGBc": XGBc, "LGBMc": LGBMc, "ETc": ETc}
    #     tasks = zip(repeat(taskmark), constructors.keys())
    #     with sopen(schedule_uri) as db:
    #         for h, k in (Scheduler(db, timeout=60) << tasks) if sched else tasks:
    #             print(datetime.now())
    #             if not sched:
    #                 print(f"{h.ansi} {k} ################################################################################################################################################################")
    #             constructor = constructors[k]
    #             d = d >> apply(constructor).alg
    #             d = ch(d, storages, storage_to_be_updated)
    #
    #             for m in ["balanced_accuracy", "precision", "recall"][:1]:
    #                 d["scoring"] = m
    #                 rets = [f"{m}_scores", f"{m}_permscores", f"{m}_pval"]
    #                 d = d >> apply(permutation_test_score, _.alg)(*rets)
    #                 d = ch(d, storages, storage_to_be_updated)
    #                 print(f"{m:20} (p-value):\t{d[rets[0]]:.4f} ({d[rets[2]]:.4f})")
    #
    #                 # # Importances
    #                 # d = d >> apply(lambda alg, X, y: clone(alg).fit(X, y)).estimator
    #                 # d = d >> apply(permutation_importance).importances
    #                 # d = ch(d, storages, storage_to_be_updated)
    #                 # r = d.importances
    #                 # for i in r.importances_mean.argsort()[::-1]:
    #                 #     if r.importances_mean[i] - r.importances_std[i] > 0:
    #                 #         print(f"importance   \t                 \t{r.importances_mean[i]:.6f}\t{r.importances_std[i]:.6f}\t{m:22}\t{d.target_var:20}\t{d.X.columns[i]}")
    #                 # print()
    #
    # print("All finished")
