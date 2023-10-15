from pprint import pprint

import numpy as np
from hdict import apply, hdict, _
from hdict.dataset.pandas_handling import file2df
from shelchemy import sopen
from sklearn import clone
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import StratifiedKFold, permutation_test_score

from germina.config import local_cache_uri, remote_cache_uri
from germina.dataset import join, osf_except_target_vars, vif_dropped_vars
from germina.nan import only_abundant, remove_cols
from germina.runner import drop_many_by_vif, ch
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

path = "data/"
loc, rem = True, True
sync = False
# bayley_average_t4
trees = 1000
d = hdict(target_var="ibq_reg_cat_t3", index="id_estudo", join="inner", shuffle=True, n_jobs=-1, return_name=False, random_state=0, max_iter=trees, n_estimators=trees,
          n_permutations=1000,
          imputrees=20)

with sopen(local_cache_uri) as local, sopen(remote_cache_uri) as remote:
    print("Load microbiome CSV data ----------------------------------")
    d = d >> apply(file2df, path + "data_microbiome___2023-07-04___vias_metabolicas_valor_absoluto_T1_n525.csv").df_microbiome_pathways1
    d = d >> apply(only_abundant, _.df_microbiome_pathways1).df_microbiome_pathways1

    d = d >> apply(file2df, path + "data_microbiome___2023-07-04___T1_vias_relab_superpathways.csv").df_microbiome_super1

    d = d >> apply(file2df, path + "data_microbiome___2023-07-04___vias_metabolicas_valor_absoluto_T2_n441.csv").df_microbiome_pathways2
    d = d >> apply(only_abundant, _.df_microbiome_pathways2).df_microbiome_pathways2

    d = d >> apply(file2df, path + "data_microbiome___2023-07-04___T2_vias_relab_superpathways.csv").df_microbiome_super2

    print("Remove previously known vars selected through by-domain VIF ----------------------------------")
    d = d >> apply(remove_cols, _.df_microbiome_pathways1, cols=vif_dropped_vars, keep=[], debug=False).df_microbiome_pathways1
    d = d >> apply(remove_cols, _.df_microbiome_super1, cols=vif_dropped_vars, keep=[], debug=False).df_microbiome_super1
    d = d >> apply(remove_cols, _.df_microbiome_pathways2, cols=vif_dropped_vars, keep=[], debug=False).df_microbiome_pathways2
    d = d >> apply(remove_cols, _.df_microbiome_super2, cols=vif_dropped_vars, keep=[], debug=False).df_microbiome_super2

    print("Apply by-domain VIF again, it doest not hurt ----------------------------------")
    d = drop_many_by_vif(d, "df_microbiome_pathways1", loc, rem, local, remote, sync)
    d = drop_many_by_vif(d, "df_microbiome_super1", loc, rem, local, remote, sync)
    d = drop_many_by_vif(d, "df_microbiome_pathways2", loc, rem, local, remote, sync)
    d = drop_many_by_vif(d, "df_microbiome_super2", loc, rem, local, remote, sync)

    d = ch(d, loc, rem, local, remote, sync)
    # d.show()

    print("Join microbiome CSV data ----------------------------------")
    d["df"] = _.df_microbiome_pathways1
    d = d >> apply(join, other=_.df_microbiome_super1).df
    d = d >> apply(join, other=_.df_microbiome_pathways2).df
    d = d >> apply(join, other=_.df_microbiome_super2).df
    d = ch(d, loc, rem, local, remote, sync)
    print("Joined ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓\n", d.df, "↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑\n")

    print("Load OSF data ----------------------------------")
    d = d >> apply(file2df, path + "germina-osf-request---davi121023.csv").df_osf_full
    osf_except_dropped_vars = ["id_estudo"]
    for v in sorted(set(osf_except_target_vars).difference(vif_dropped_vars)):
        for i in range(7):
            sub = f"{v}_t{i}"
            if sub in d.df_osf_full:
                osf_except_dropped_vars.append(sub)
    osf_except_dropped_vars.sort()
    d["undropped_osf_vars"] = osf_except_dropped_vars
    d = d >> apply(lambda df_osf_full, undropped_osf_vars: df_osf_full[undropped_osf_vars]).df_undropped_osf
    d = ch(d, loc, rem, local, remote, sync)

    print("Join OSF data ----------------------------------")
    d = d >> apply(join, other=_.df_undropped_osf).df_before_vif
    d = ch(d, loc, rem, local, remote, sync)
    print("Joined OSF ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓\n", d.df, "↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑\n")

    # print("Format problematic attributes.")   todo ?

    print("Overall removal of NaNs and VIF application ----------------------------------")
    # pprint([d.hosh, d.hoshes])
    d = drop_many_by_vif(d, "df_before_vif", loc, rem, local, remote, sync)  # está removendo rows e cols
    d = ch(d, loc, rem, local, remote, sync)
    # d.show()
    # d["df"] = d.df_before_vif

    print("Join target ----------------------------------")
    d = d >> apply(lambda df_osf_full, target_var: df_osf_full[[target_var, "id_estudo"]].reindex(sorted([target_var, "id_estudo"]), axis=1)).df_target
    d = d >> apply(join, other=_.df_target).df
    d = d >> apply(join, df=_.df_before_vif, other=_.df_target).df_before_vif
    d = ch(d, loc, rem, local, remote, sync)
    print(f"Joined target {d.target_var} ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓\n", d.df, "↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑\n")

    print("Restart now by using only noncolinear columns to recover NaNs ----------------------------------------------------")
    d = d >> apply(lambda df: df.columns.to_list()).columns
    d = ch(d, loc, rem, local, remote, sync)
    print(d.df.columns)
    d = d >> apply(lambda df_before_vif, columns: df_before_vif[columns]).df
    d = ch(d, loc, rem, local, remote, sync)
    print(f"Noncolinear dataset with NaNs again ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓\n", d.df, "↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑\n")

    print(d.df.index, "falta ainda checar se ainda aparece id_estudo em X (ou ao menos como ultima coluna de df)")
    if d:
        raise Exception(f"")

    print("Separate quintiles 2,3,4 and NaN-labeled rows for IterativeImputer ----------------------------------")
    d = d >> apply(lambda df, target_var: df[df[target_var].isna() | (df[target_var] > 1) & (df[target_var] < 5)]).df_for_imputer
    d = d >> apply(remove_cols, df=_.df_for_imputer, cols=[d.target_var], keep=[]).df_for_imputer
    d = ch(d, loc, rem, local, remote, sync)
    print(f"df_for_imputer ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓\n", d.df_for_imputer, "↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑\n")

    print("Model imputation ----------------------------------")
    d = d >> apply(RandomForestRegressor, n_estimators=_.imputrees).imputalg
    d = d >> apply(lambda imputalg, df_for_imputer: IterativeImputer(estimator=clone(imputalg)).fit(X=df_for_imputer)).imputer
    d = ch(d, loc, rem, local, remote, sync)
    # d.show()

    print("Build dataset with quintiles 1,5 and exclude NaN-labeled rows ----------------------------------")
    d = d >> apply(lambda df, target_var: df[df[target_var].notna() & ((df[target_var] == 1) | (df[target_var] == 5))]).df_dataset
    d = ch(d, loc, rem, local, remote, sync)
    print(f"df_dataset ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓\n", d.df_dataset, "↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑\n")

    print("Separate X from dataset and fill missing values using imputer ----------------------------------")
    d = d >> apply(remove_cols, df=_.df_dataset, cols=[d.target_var], keep=[]).df_dataset_except_target
    d = d >> apply(lambda imputer, df_dataset_except_target: imputer.transform(X=df_dataset_except_target)).X
    d = ch(d, loc, rem, local, remote, sync)
    print(f"X ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓\n", d.X, "↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑\n")

    print("Separate y from dataset ----------------------------------")
    d = d >> apply(lambda df_dataset, target_var: df_dataset[target_var]).y
    d = ch(d, loc, rem, local, remote, sync)
    print(f"X ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓\n", d.y, "↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑\n")

    print("Calculate class balance ----------------------------------")
    d = d >> apply(lambda X: X.shape).Xshape
    d = d >> apply(lambda y: y.shape).yshape
    d = d >> apply(lambda y: np.unique(y, return_counts=True))("unique_labels", "counts")
    d = d >> apply(lambda y, counts: counts / len(y)).proportions
    d = ch(d, loc, rem, local, remote, sync)
    print("X, y:", d.Xshape, d.yshape)
    print(f"{d.counts=}\t{d.proportions=}")

    print("Model ----------------------------------")
    d = d >> apply(StratifiedKFold).cv
    d = d >> apply(HistGradientBoostingClassifier).alg
    d = ch(d, loc, rem, local, remote, sync)

    d = d >> apply(permutation_test_score, _.alg, _.X, _.y, cv=_.cv, scoring="balanced_accuracy")("bacc_scores", "bacc_permscores", "bacc_pval")
    d = ch(d, loc, rem, local, remote, sync)

    d = d >> apply(permutation_test_score, _.alg, _.X, _.y, cv=_.cv, scoring="precision")("pr_scores", "pr_permscores", "pr_pval")
    d = ch(d, loc, rem, local, remote, sync)

    d = d >> apply(permutation_test_score, _.alg, _.X, _.y, cv=_.cv, scoring="recall")("rc_scores", "rc_permscores", "rc_pval")
    d = ch(d, loc, rem, local, remote, sync)

    print()
    print()
    print()
    print("balanced_accuracy", d.bacc_scores, d.bacc_pval)
    print("precision", d.pr_scores, d.pr_pval)
    print("recall", d.rc_scores, d.rc_pval)

    # # Importances
    # model = f"{target}_{classifier_field}_model"
    # d = d >> apply(lambda c, *args, **kwargs: clone(c).fit(*args, **kwargs), field(classifier_field), _.X, _.y)(model)
    # importances_field_name = f"{target}_{classifier_field}_importances"
    # d = d >> apply(permutation_importance, field(model), _.X, _.y, n_repeats=100, scoring=m, n_jobs=-1)(importances_field_name)
    # d = ch(d, loc, rem, local, remote, sync)
    # r = d[importances_field_name]
    # for i in r.importances_mean.argsort()[::-1]:
    #     if r.importances_mean[i] - r.importances_std[i] > 0:
    #         print(f"importance   \t                 \t{r.importances_mean[i]:.6f}\t{r.importances_std[i]:.6f}\t{m:22}\t{classifier_field:24}\t{target:20}\t{d.columns[i]}")

    # d = d >> apply(lambda alg, *args, **kwargs: clone(alg).fit(*args, **kwargs), X=_.X, y=_.y).model
    # d.model.predict(d.X)
