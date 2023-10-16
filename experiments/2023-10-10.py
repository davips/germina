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
from pandas import DataFrame
from shelchemy import sopen
from shelchemy.scheduler import Scheduler
from sklearn import clone
from sklearn.ensemble import ExtraTreesClassifier as ETc
from sklearn.ensemble import HistGradientBoostingClassifier as HGBc, RandomForestRegressor as RFr
from sklearn.ensemble import RandomForestClassifier as RFc
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.inspection import permutation_importance
from sklearn.model_selection import StratifiedKFold, permutation_test_score
from xgboost import XGBClassifier as XGBc

from germina.config import local_cache_uri, remote_cache_uri, near_cache_uri, schedule_uri
from germina.dataset import join, osf_except_target_vars, vif_dropped_vars
from germina.nan import only_abundant, remove_cols
from germina.runner import drop_many_by_vif, ch

__ = enable_iterative_imputer
dct = handle_command_line(argv, pvalruns=int, importanceruns=int, imputertrees=int, seed=int, target=str, trees=int, vifdomain=False, vifall=False, nans=False, sched=False)
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
    index="id_estudo", join="inner", shuffle=True, n_jobs=-1, return_name=False
)
vifdomain, vifall, nans, sched = dct["vifdomain"], dct["vifall"], dct["nans"], dct["sched"]

with sopen(local_cache_uri) as local_storage, sopen(near_cache_uri) as near_storage, sopen(remote_cache_uri) as remote_storage:
    storages = {
        "remote": remote_storage,
        "near": near_storage,
        "local": local_storage,
    }
    to_be_updated = None  # "near"

    print("Load microbiome CSV data ------------------------------------------------------------------------------------------------------------------------------------------------")
    d = d >> apply(file2df, path + "data_microbiome___2023-07-04___vias_metabolicas_valor_absoluto_T1_n525.csv").df_microbiome_pathways1
    d = d >> apply(only_abundant, _.df_microbiome_pathways1).df_microbiome_pathways1

    d = d >> apply(file2df, path + "data_microbiome___2023-07-04___T1_vias_relab_superpathways.csv").df_microbiome_super1

    d = d >> apply(file2df, path + "data_microbiome___2023-07-04___vias_metabolicas_valor_absoluto_T2_n441.csv").df_microbiome_pathways2
    d = d >> apply(only_abundant, _.df_microbiome_pathways2).df_microbiome_pathways2

    d = d >> apply(file2df, path + "data_microbiome___2023-07-04___T2_vias_relab_superpathways.csv").df_microbiome_super2

    if vifdomain:
        print("Remove previously known vars selected through by-domain VIF -------------------------------------------------------------------------------------------------------------")
        d = d >> apply(remove_cols, _.df_microbiome_pathways1, cols=vif_dropped_vars, keep=[], debug=False).df_microbiome_pathways1
        d = d >> apply(remove_cols, _.df_microbiome_super1, cols=vif_dropped_vars, keep=[], debug=False).df_microbiome_super1
        d = d >> apply(remove_cols, _.df_microbiome_pathways2, cols=vif_dropped_vars, keep=[], debug=False).df_microbiome_pathways2
        d = d >> apply(remove_cols, _.df_microbiome_super2, cols=vif_dropped_vars, keep=[], debug=False).df_microbiome_super2

        print("Apply by-domain VIF again, it doest not hurt ----------------------------------------------------------------------------------------------------------------------------")
        d = drop_many_by_vif(d, "df_microbiome_pathways1", storages, to_be_updated)
        d = drop_many_by_vif(d, "df_microbiome_super1", storages, to_be_updated)
        d = drop_many_by_vif(d, "df_microbiome_pathways2", storages, to_be_updated)
        d = drop_many_by_vif(d, "df_microbiome_super2", storages, to_be_updated)

    d = ch(d, storages, to_be_updated)
    # d.show()

    print("Join microbiome CSV data ------------------------------------------------------------------------------------------------------------------------------------------------")
    d["df"] = _.df_microbiome_pathways1
    d = d >> apply(join, other=_.df_microbiome_super1).df
    d = d >> apply(join, other=_.df_microbiome_pathways2).df
    d = d >> apply(join, other=_.df_microbiome_super2).df
    d = ch(d, storages, to_be_updated)
    print("Joined ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓\n", d.df, "↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑\n")

    print("Load OSF data -----------------------------------------------------------------------------------------------------------------------------------------------------------")
    d = d >> apply(file2df, path + "germina-osf-request---davi121023.csv").df_osf_full
    osf_except_dropped_vars = ["id_estudo"]
    seq = set(osf_except_target_vars).difference(vif_dropped_vars) if vifdomain else osf_except_target_vars
    for v in sorted(seq):
        for i in range(7):
            sub = f"{v}_t{i}"
            if sub in d.df_osf_full:
                osf_except_dropped_vars.append(sub)
    osf_except_dropped_vars.sort()
    d["undropped_osf_vars"] = osf_except_dropped_vars
    d = d >> apply(lambda df_osf_full, undropped_osf_vars: df_osf_full[undropped_osf_vars]).df_undropped_osf
    d = ch(d, storages, to_be_updated)

    print("Join OSF data -----------------------------------------------------------------------------------------------------------------------------------------------------------")
    csv_dup_vars = []
    for v in sorted(d.df.columns):
        if v in d.df_undropped_osf:
            csv_dup_vars.append(v)
    d = d >> apply(remove_cols, cols=csv_dup_vars, keep=[]).df
    d = d >> apply(join, other=_.df_undropped_osf).df_before_vif
    d = ch(d, storages, to_be_updated)
    print("Joined OSF ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓\n", d.df_before_vif, "↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑\n")

    # print("Format problematic attributes.")   todo ?

    d["df_after_vif"] = d.df_before_vif
    d = ch(d, storages, to_be_updated)
    if vifall:
        print("Overall removal of NaNs and VIF application -----------------------------------------------------------------------------------------------------------------------------")
        d = drop_many_by_vif(d, "df_after_vif", storages, to_be_updated)  # está removendo rows e cols
        d = ch(d, storages, to_be_updated)

    print("Join target -------------------------------------------------------------------------------------------------------------------------------------------------------------")
    d = d >> apply(lambda df_osf_full, target_var: df_osf_full[[target_var, "id_estudo"]].reindex(sorted([target_var, "id_estudo"]), axis=1)).df_target
    d = d >> apply(join, df=_.df_after_vif, other=_.df_target).df_after_vif
    d = d >> apply(join, df=_.df_before_vif, other=_.df_target).df_before_vif
    d = ch(d, storages, to_be_updated)
    print(f"Joined target {d.target_var} with df_before_vif ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓\n", d.df_before_vif, "↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑\n")
    print(f"Joined target {d.target_var} with df_after_vif ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓\n", d.df_after_vif, "↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑\n")

    if nans and (vifall or vifdomain):
        print("Restart now to recover NaNs removed by VIF ---------------------------------------------------------------------------------------------------------------------------------------------")
        d = d >> apply(lambda df_after_vif: df_after_vif.columns.to_list()).columns
        d = ch(d, storages, to_be_updated)
        d = d >> apply(lambda df_before_vif, columns: df_before_vif[columns]).df
        d = ch(d, storages, to_be_updated)
        print(f"Noncolinear dataset with NaNs again ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓\n", d.df, "↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑\n")
    else:
        d["df"] = _.df_after_vif
        d = ch(d, storages, to_be_updated)

    print("Separate quintiles 2,3,4 and NaN-labeled rows for IterativeImputer ------------------------------------------------------------------------------------------------------")
    d = d >> apply(lambda df, target_var: df[df[target_var].isna() | (df[target_var] > 1) & (df[target_var] < 5)]).df_for_imputer
    d = d >> apply(remove_cols, df=_.df_for_imputer, cols=[d.target_var], keep=[]).df_for_imputer
    d = ch(d, storages, to_be_updated)
    print(f"df_for_imputer ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓\n", d.df_for_imputer, "↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑\n")

    print("Model imputation --------------------------------------------------------------------------------------------------------------------------------------------------------")
    d = d >> apply(RFr, n_estimators=_.imputrees).imputalg
    d = d >> apply(lambda imputalg, df_for_imputer: IterativeImputer(estimator=clone(imputalg)).fit(X=df_for_imputer)).imputer
    d = ch(d, storages, to_be_updated)
    # d.show()

    print("Build dataset with quintiles 1,5 and exclude NaN-labeled rows -----------------------------------------------------------------------------------------------------------")
    d = d >> apply(lambda df, target_var: df[df[target_var].notna() & ((df[target_var] == 1) | (df[target_var] == 5))]).df_dataset
    d = ch(d, storages, to_be_updated)
    print(f"df_dataset ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓\n", d.df_dataset, "↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑\n")

    print("Separate X from dataset and fill missing values using imputer -----------------------------------------------------------------------------------------------------------")
    d = d >> apply(remove_cols, df=_.df_dataset, cols=[d.target_var], keep=[]).df_dataset_except_target
    d = d >> apply(lambda imputer, df_dataset_except_target: DataFrame(imputer.transform(X=df_dataset_except_target), columns=df_dataset_except_target.columns)).X
    d = ch(d, storages, to_be_updated)
    print(f"X {d.X.shape} ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓\n", d.X, "↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑\n")

    print("Separate y from dataset -------------------------------------------------------------------------------------------------------------------------------------------------")
    d = d >> apply(lambda df_dataset, target_var: df_dataset[target_var] // 5).y
    d = ch(d, storages, to_be_updated)
    print(f"y {d.y.shape} ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓\n", d.y, "↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑\n")

    print("Calculate class balance -------------------------------------------------------------------------------------------------------------------------------------------------")
    d = d >> apply(lambda X: X.shape).Xshape
    d = d >> apply(lambda y: y.shape).yshape
    d = d >> apply(lambda y: np.unique(y, return_counts=True))("unique_labels", "counts")
    d = d >> apply(lambda y, counts: counts / len(y)).proportions
    d = ch(d, storages, to_be_updated)
    print("X, y:", d.Xshape, d.yshape)
    print(f"{d.counts=}\t{d.proportions=}")

    print("Induce classifier -------------------------------------------------------------------------------------------------------------------------------------------------------")
    d = d >> apply(StratifiedKFold).cv
    taskmark = d.hosh - (d.hoshes["n_jobs"] * b"n_jobs")
    constructors = {"HGBc": HGBc, "RFc": RFc, "XGBc": XGBc, "LGBMc": LGBMc, "ETc": ETc}
    tasks = zip(repeat(taskmark), constructors.keys())
    with sopen(schedule_uri) as db:
        for h, k in (Scheduler(db, timeout=20) << tasks) if sched else tasks:
            if not sched:
                print(f"{h.ansi} {k} ################################################################################################################################################################")
            constructor = constructors[k]
            d = d >> apply(constructor).alg
            d = ch(d, storages, to_be_updated)

            for m in ["balanced_accuracy", "precision", "recall"]:
                d["scoring"] = m
                rets = [f"{m}_scores", f"{m}_permscores", f"{m}_pval"]
                d = d >> apply(permutation_test_score, _.alg)(*rets)
                d = ch(d, storages, to_be_updated)
                print(f"{m:20} (p-value):\t{d[rets[0]]:.4f} ({d[rets[2]]:.4f})")

                # Importances
                d = d >> apply(lambda alg, X, y: clone(alg).fit(X, y)).estimator
                d = d >> apply(permutation_importance).importances
                d = ch(d, storages, to_be_updated)
                r = d.importances
                for i in r.importances_mean.argsort()[::-1]:
                    if r.importances_mean[i] - r.importances_std[i] > 0:
                        print(f"importance   \t                 \t{r.importances_mean[i]:.6f}\t{r.importances_std[i]:.6f}\t{m:22}\t{d.target_var:20}\t{d.X.columns[i]}")
                print()

print("All finished")
