import os
from pprint import pprint

import numpy as np
from hdict import apply, hdict, _
from hdict.dataset.pandas_handling import file2df
from shelchemy import sopen
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, permutation_test_score

from germina.config import local_cache_uri
from germina.dataset import join, metavars_no_target, vif_dropped
from germina.nan import only_abundant, remove_cols
from germina.runner import drop_many_by_vif, ch

path = "data/"
loc, rem, remote = True, False, None
sync = False
vif = True
# bayley_average_t4
d = hdict(target="ibq_reg_cat_t3", index="id_estudo", join="inner", shuffle=True, n_jobs=-1, return_name=False, random_state=0, n_permutations=10000, max_iter=3000)

with sopen(local_cache_uri) as local:
    d = d >> apply(file2df, path + "data_microbiome___2023-07-04___vias_metabolicas_valor_absoluto_T1_n525.csv").microbiome_pathways1
    d = d >> apply(only_abundant, _.microbiome_pathways1).microbiome_pathways1
    if vif:
        d = d >> apply(remove_cols, _.microbiome_pathways1, cols=vif_dropped, keep=[], debug=False).microbiome_pathways1
        d = drop_many_by_vif(d, "microbiome_pathways1", loc, rem, local, remote, sync)

    d = d >> apply(file2df, path + "data_microbiome___2023-07-04___T1_vias_relab_superpathways.csv").microbiome_super1
    if vif:
        d = d >> apply(remove_cols, _.microbiome_super1, cols=vif_dropped, keep=[], debug=False).microbiome_super1
        d = drop_many_by_vif(d, "microbiome_super1", loc, rem, local, remote, sync)

    d = d >> apply(file2df, path + "data_microbiome___2023-07-04___vias_metabolicas_valor_absoluto_T2_n441.csv").microbiome_pathways2
    d = d >> apply(only_abundant, _.microbiome_pathways2).microbiome_pathways2
    if vif:
        d = d >> apply(remove_cols, _.microbiome_pathways2, cols=vif_dropped, keep=[], debug=False).microbiome_pathways2
        d = drop_many_by_vif(d, "microbiome_pathways2", loc, rem, local, remote, sync)

    d = d >> apply(file2df, path + "data_microbiome___2023-07-04___T2_vias_relab_superpathways.csv").microbiome_super2
    if vif:
        d = d >> apply(remove_cols, _.microbiome_super2, cols=vif_dropped, keep=[], debug=False).microbiome_super2
        d = drop_many_by_vif(d, "microbiome_super2", loc, rem, local, remote, sync)
    d = ch(d, loc, rem, local, remote, sync)
    d.show()

    # Join all non OSF data #####################################################################################################
    d["df"] = _.microbiome_pathways1
    d = d >> apply(join, other=_.microbiome_super1).df
    d = d >> apply(join, other=_.microbiome_pathways2).df
    d = d >> apply(join, other=_.microbiome_super2).df
    d = ch(d, loc, rem, local, remote, sync)
    print("Joined------------------------------------------------------------------------\n", d.df, "______________________________________________________\n")

    # Join metadata #############################################################################################################
    d = d >> apply(file2df, path + "germina-osf-request---davi121023.csv").metadata_full
    metavars = ["id_estudo"]
    for v in sorted(set(metavars_no_target).difference(vif_dropped)):
        for i in range(7):
            sub = f"{v}_t{i}"
            if sub in d.metadata_full:
                metavars.append(sub)
    metavars.sort()
    d = d >> apply(lambda metadata_full, mtvs: metadata_full[mtvs], mtvs=metavars).metadata

    print("Format problematic attributes.")
    # d = d >> apply(bina, _.metadata, attribute="antibiotic", positive_category="yes").metadata
    # d = d >> apply(bina, _.metadata, attribute="EBF_3m", positive_category="EBF").metadata
    # for i in range(7):
    #     d = d >> apply(loga, _.metadata, attribute=f"renda_familiar_total_t{i}").metadata
    d = d >> apply(join, other=_.metadata).df
    # d = d >> apply(remove_nan_rows_cols, keep=["id_estudo", target]).df
    d = ch(d, loc, rem, local, remote, sync)
    print("With metadata----------------------------------------------------------\n", d.df, "______________________________________________________\n")

    ##############################   VIF    #####################################################################################
    if vif:
        pprint([d.hosh, d.hoshes])
        d = drop_many_by_vif(d, "df", loc, rem, local, remote, sync)
        d = ch(d, loc, rem, local, remote, sync)

    # Join targets ##############################################################################################################
    d = d >> apply(lambda metadata_full, target: metadata_full[[target, "id_estudo"]].reindex(sorted([target, "id_estudo"]), axis=1)).t

    d = d >> apply(join, other=_.t).df
    d = ch(d, loc, rem, local, remote, sync)
    print(d.metadata_full.columns.tolist())
    print("Dataset-----------------------------------------------------------------------\n", d.df, "______________________________________________________\n")

    print("Restart now by using only noncolinear columns ----------------------------------------------------")
    d = d >> apply(lambda df: df.columns.to_list()).columns
    d = ch(d, loc, rem, local, remote, sync)
    print(d.df.columns)
    d = d >> apply(lambda metadata_full, columns: metadata_full[columns + ["id_estudo"]]).df
    d = ch(d, loc, rem, local, remote, sync)
    print("Noncolinear dataset with NaNs again ------------------------------------------------------------\n", d.df, "______________________________________________________\n")

    print("############################# remove NaNs from y ######################")
    d = d >> apply(lambda df, target: df[df[target].notna()]).df
    d = ch(d, loc, rem, local, remote, sync)

    print(f"########################## X, y #######################################")
    d = d >> apply(remove_cols, cols=[d.target], keep=[]).X
    d = d >> apply(lambda df, target: df[target]).y
    d = ch(d, loc, rem, local, remote, sync)
    d = d >> apply(lambda X: X.shape).Xshape
    d = d >> apply(lambda y: y.shape).yshape
    d = d >> apply(lambda y: np.unique(y, return_counts=True))("unique_labels", "counts")
    d = d >> apply(lambda y, counts: counts / len(y)).proportions
    d = ch(d, loc, rem, local, remote, sync)
    print("X, y:", d.Xshape, d.yshape)
    print(f"{d.counts=}\t{d.proportions=}")

    print("############################# exclui quintis ######################")
    d = d >> apply(lambda df, target: df[(df[target] == 1) | (df[target] == 5)]).df
    d = d >> apply(remove_cols, cols=[d.target], keep=[]).X
    d = d >> apply(lambda df, target: df[target]).y
    d = ch(d, loc, rem, local, remote, sync)
    d = d >> apply(lambda X: X.shape).Xshape
    d = d >> apply(lambda y: y.shape).yshape
    d = d >> apply(lambda y: np.unique(y, return_counts=True))("unique_labels", "counts")
    d = d >> apply(lambda y, counts: counts / len(y)).proportions
    d = ch(d, loc, rem, local, remote, sync)
    print("X, y:", d.Xshape, d.yshape)
    print(f"{d.counts=}\t{d.proportions=}")

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
