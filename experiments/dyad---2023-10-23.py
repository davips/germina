from datetime import datetime
from functools import reduce
from itertools import repeat
from operator import mul, add
from pprint import pprint
from sys import argv

import numpy as np
from argvsucks import handle_command_line
from hdict import apply, hdict, _
from hdict.dataset.pandas_handling import file2df
from lightgbm import LGBMClassifier as LGBMc
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
from numpy import array
from pandas import DataFrame
from shelchemy import sopen
from shelchemy.scheduler import Scheduler
from sklearn import clone
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier as ETc
from sklearn.ensemble import HistGradientBoostingClassifier as HGBc, RandomForestRegressor as RFr
from sklearn.ensemble import RandomForestClassifier as RFc
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.inspection import permutation_importance
from sklearn.manifold import MDS
from sklearn.model_selection import StratifiedKFold, permutation_test_score, KFold
from sklearn.preprocessing import StandardScaler
from sortedness.global_ import cov2dissimilarity
from xgboost import XGBClassifier as XGBc

from germina.config import local_cache_uri, remote_cache_uri, near_cache_uri, schedule_uri
from germina.dataset import join, osf_except_target_vars__no_t, eeg_vars__no_t
from germina.nan import only_abundant, remove_cols, remove_nan_rows_cols
from germina.runner import drop_many_by_vif, ch, sgid2estudoid

__ = enable_iterative_imputer
dct = handle_command_line(argv, pvalruns=int, importanceruns=int, imputertrees=int, seed=int, target=str, trees=int, vifdomain=False, vifall=False, nans=False, sched=False, up="")
print(datetime.now())
pprint(dct, sort_dicts=False)
print()
path = "data/"
d = hdict(
    n_permutations=dct["pvalruns"],
    n_repeats=dct["importanceruns"],
    imputrees=dct["imputertrees"],
    random_state=dct["seed"],
    target_var=dct["target"],  # "ibq_reg_cat_t3", bayley_average_t4
    max_iter=dct["trees"], n_estimators=dct["trees"],
    n_splits=5,
    index="id_estudo", join="inner", shuffle=True, n_jobs=-1, return_name=False
)
vifdomain, vifall, nans, sched, storage_to_be_updated = dct["vifdomain"], dct["vifall"], dct["nans"], dct["sched"], dct["up"]

with (sopen(local_cache_uri) as local_storage, sopen(near_cache_uri) as near_storage, sopen(remote_cache_uri) as remote_storage):
    storages = {
        "remote": remote_storage,
        "near": near_storage,
        "local": local_storage,
    }

    print("Load synapse dyadic data ------------------------------------------------------------------------------------------------------------------------------------------------")
    print(datetime.now())
    d = d >> apply(file2df, path + "synapse---LP_LDESS_dataset_Aug_2023---dyadic.csv").df_dyadic
    d = ch(d, storages, storage_to_be_updated)
    print("Loaded ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓\n", d.df_dyadic, "↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑\n")
    d = d >> apply(sgid2estudoid, _.df_dyadic, "ID:").df_dyadic
    d = ch(d, storages, storage_to_be_updated)
    print("Fixed id ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓\n", d.df_dyadic, "↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑\n")
    if vifdomain:
        print("Apply by-domain VIF ----------------------------------------------------------------------------------------------------------------------------")
        print(datetime.now())
        d = drop_many_by_vif(d, "df_dyadic", storages, storage_to_be_updated)
        d = ch(d, storages, storage_to_be_updated)
        print("after VIF ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓\n", d.df_dyadic, "↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑\n")

    print("Std ------------------------------------------------------------------------------------------------------------------------------------------------")
    print(datetime.now())
    d = d >> apply(lambda x: DataFrame(StandardScaler().fit_transform(x)), _.df_dyadic).df_dyadic_std
    print("Scaled ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓\n", d.df_dyadic_std, "↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑\n")

    seed = 3
    fig, ax = plt.subplots()
    pca = PCA(random_state=seed)
    pca.fit(d.df_dyadic_std)
    print(pca.explained_variance_)
    a = StandardScaler().fit_transform(pca.transform(d.df_dyadic_std))
    mds = MDS(n_jobs=-1, random_state=seed)
    m = cov2dissimilarity(d.df_dyadic_std.transpose().cov().to_numpy())
    b = StandardScaler().fit_transform(mds.fit_transform(m))
    x, y = a[:, :2], b[:, :2]
    lc = LineCollection(list(zip(x, y)), colors="gray")
    ax.add_collection(lc)
    plt.scatter(x[:, 0], x[:, 1])
    plt.scatter(y[:, 0], y[:, 1])
    plt.show()

    print("Load synapse PCI data ------------------------------------------------------------------------------------------------------------------------------------------------")
    print(datetime.now())
    d = d >> apply(file2df, path + "synapse---LP_LDESS_dataset_Aug_2023---pci.csv").df_pci
    d = d >> apply(sgid2estudoid, _.df_pci, "ID:").df_pci
    if vifdomain:
        print("Apply by-domain VIF ----------------------------------------------------------------------------------------------------------------------------")
        print(datetime.now())
        d = drop_many_by_vif(d, "df_pci", storages, storage_to_be_updated)
    d = ch(d, storages, storage_to_be_updated)
    print("Loaded ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓\n", d.df_pci, "↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑\n")

    print("Load OSF non-target data -----------------------------------------------------------------------------------------------------------------------------------------------------------")
    print(datetime.now())
    d = d >> apply(file2df, path + "germina-osf-request---davi121023.csv").df_osf_full
    osf_except_target_vars = ["id_estudo"]
    for v in sorted(osf_except_target_vars__no_t):
        for i in range(7):
            sub = f"{v}_t{i}"
            if sub in d.df_osf_full:
                osf_except_target_vars.append(sub)
    osf_except_target_vars.sort()
    d["osf_except_target_vars"] = osf_except_target_vars
    d = d >> apply(lambda df_osf_full, osf_except_target_vars: df_osf_full[osf_except_target_vars]).df_osf_except_targets
    d = ch(d, storages, storage_to_be_updated)
    print("Loaded ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓\n", d.df_osf_except_targets, "↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑\n")

    print("Load OSF EEG data -----------------------------------------------------------------------------------------------------------------------------------------------------------")
    print(datetime.now())
    eeg_vars = ["id_estudo"]
    for v in sorted(eeg_vars__no_t):
        for i in range(7):
            sub = f"{v}_t{i}"
            if sub in d.df_osf_full:
                eeg_vars.append(sub)
    eeg_vars.sort()
    d["eeg_vars"] = eeg_vars
    d = d >> apply(lambda df_osf_full, eeg_vars: df_osf_full[eeg_vars]).df_eeg
    d = ch(d, storages, storage_to_be_updated)
    print("Loaded ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓\n", d.df_eeg, "↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑\n")
    exit()

    # print("Left join dyadic with EEG -------------------------------------------------------------------------------------------------------------------------------------------------------------")
    # print(datetime.now())
    # d = d >> apply(join, df=_.df_dyadic, other=_.df_eeg, join="left").df
    # d = ch(d, storages, storage_to_be_updated)
    # print(f"Joined  ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓\n", d.df, "↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑\n")

    # Get right T for each baby bestid()

    # print("Left join dyadic with target -------------------------------------------------------------------------------------------------------------------------------------------------------------")
    # print(datetime.now())
    # d = d >> apply(lambda df_osf_full, target_var: df_osf_full[[target_var, "id_estudo"]].reindex(sorted([target_var, "id_estudo"]), axis=1)).df_target
    # d = d >> apply(join, df=_.df_dyadic, other=_.df_target, join="left").df
    # d = ch(d, storages, storage_to_be_updated)
    # print(f"Joined target {d.target_var} ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓\n", d.df, "↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑\n")

    d["df"] = _.df_eeg

    print("Remove NaNs -------------------------------------------------------------------------------------------------------------------------------------------------")
    print(datetime.now())
    d = d >> apply(remove_nan_rows_cols, keep=[]).df
    d = ch(d, storages, storage_to_be_updated)
    print(f"↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓\n", d.df, "↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑\n")

    print("Left join with target -------------------------------------------------------------------------------------------------------------------------------------------------------------")
    print(datetime.now())
    d = d >> apply(lambda df_osf_full, target_var: df_osf_full[[target_var, "id_estudo"]].reindex(sorted([target_var, "id_estudo"]), axis=1)).df_target
    d = d >> apply(join, df=_.df, other=_.df_target, join="left").df
    d = ch(d, storages, storage_to_be_updated)
    print(f"Joined target {d.target_var} ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓\n", d.df, "↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑\n")

    print("Remove NaNs -------------------------------------------------------------------------------------------------------------------------------------------------")
    print(datetime.now())
    d = d >> apply(remove_nan_rows_cols, keep=[]).df
    d = ch(d, storages, storage_to_be_updated)
    print(f"↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓\n", d.df, "↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑\n")

    print("Separate X from dataset -----------------------------------------------------------------------------------------------------------")
    print(datetime.now())
    d = d >> apply(remove_cols, df=_.df, cols=[d.target_var], keep=[]).X
    d = ch(d, storages, storage_to_be_updated)
    print(f"X {d.X.shape} ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓\n", d.X, "↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑\n")

    print("Separate y from dataset -------------------------------------------------------------------------------------------------------------------------------------------------")
    print(datetime.now())
    d = d >> apply(lambda df, target_var: df[target_var]).y
    d = ch(d, storages, storage_to_be_updated)
    print(f"y {d.y.shape} ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓\n", d.y, "↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑\n")

    print("Calculate class balance -------------------------------------------------------------------------------------------------------------------------------------------------")
    print(datetime.now())
    d = d >> apply(lambda X: X.shape).Xshape
    d = d >> apply(lambda y: y.shape).yshape
    d = d >> apply(lambda y: np.unique(y, return_counts=True))("unique_labels", "counts")
    d = d >> apply(lambda y, counts: counts / len(y)).proportions
    d = ch(d, storages, storage_to_be_updated)
    print("X, y:", d.Xshape, d.yshape)
    print(f"{d.counts=}\t{d.proportions=}")

    print("Induce regressor -------------------------------------------------------------------------------------------------------------------------------------------------------")
    print(datetime.now())
    d = d >> apply(KFold).cv
    taskmark = d.hosh - (d.hoshes["n_jobs"] * b"n_jobs")
    constructors = {"RFr": RFr}
    tasks = zip(repeat(taskmark), constructors.keys())
    with sopen(schedule_uri) as db:
        for h, k in (Scheduler(db, timeout=60) << tasks) if sched else tasks:
            print(datetime.now())
            if not sched:
                print(f"{h.ansi} {k} ################################################################################################################################################################")
            constructor = constructors[k]
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
