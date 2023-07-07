import sys
from pprint import pprint

import numpy as np
from lightgbm import LGBMClassifier as LGBMc
from numpy import array
from numpy import mean, std
from pandas import DataFrame
from sklearn import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier as ETc
from sklearn.ensemble import RandomForestClassifier as RFc
from sklearn.inspection import permutation_importance
from sklearn.linear_model import SGDClassifier as SGDc
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold, cross_val_score, cross_val_predict, StratifiedKFold, permutation_test_score
from xgboost import XGBClassifier as XGBc

from germina.config import local_cache_uri, remote_cache_uri
from germina.dataset import join
from germina.nan import remove_cols, bina, loga, remove_nan_rows_cols
from hdict import _, apply, cache, hdict
from hdict import field
from hdict.dataset.pandas_handling import file2df
from shelchemy import sopen


def run(d: hdict, t1=False, t2=False, microbiome=False, microbiome_extra=False, eeg=False, metavars=None, targets_meta=None, targets_eeg1=None, targets_eeg2=None, stratifiedcv=True, path="data/"):
    lst = []
    if t1:
        lst.append("t1")
    if t2:
        lst.append("t2")
    if microbiome:
        lst.append("microbiome")
    if microbiome_extra:
        lst.append("microbiome_extra")
    if eeg:
        lst.append("eeg")
    if metavars:
        lst.append(f"{metavars=}")
    if targets_meta:
        lst.append(f"{targets_meta=}")
    if targets_eeg1:
        lst.append(f"{targets_eeg1=}")
    if targets_eeg2:
        lst.append(f"{targets_eeg2=}")
    if stratifiedcv:
        lst.append("stratifiedcv")
    name = "out/" + "§".join(lst).replace("_", "_").replace("§", "---") + ".txt"
    name = name.replace("=", "").replace("[", "«").replace("]", "»").replace(", ", ",").replace("'", "")
    print(name)
    oldout = sys.stdout
    with open(name, 'w') as sys.stdout:
        newout = sys.stdout
        # sys.stdout = oldout

        print(f"Scenario: {t1=}, {t2=}, {microbiome=}, {microbiome_extra=}, {eeg=},\n"
              f"{metavars=},\n"
              f"{targets_meta=},\n"
              f"{targets_eeg1=},\n"
              f"{targets_eeg2=}")
        pprint(dict(d))
        print()
        d = d >> dict(join="inner", shuffle=True, n_jobs=-1, return_name=False)

        if metavars is None:
            metavars = []
        if targets_meta is None:
            targets_meta = []
        if targets_eeg2 is None:
            targets_eeg2 = []
        if targets_eeg1 is None:
            targets_eeg1 = []
        targets = targets_meta + targets_eeg1 + targets_eeg2
        with sopen(local_cache_uri) as local, sopen(remote_cache_uri) as remote:
            if microbiome:  #################################################################################################################
                if t1:
                    d = d >> apply(file2df, path + "data_microbiome___2023-06-18___alpha_diversity_n525.csv").microbiome_alpha1
                    if microbiome_extra:
                        d = d >> apply(file2df, path + "data_microbiome___2023-07-04___vias_metabolicas_valor_absoluto_T1_n525.csv").microbiome_pathways1
                        d = d >> apply(file2df, path + "data_microbiome___2023-06-18___especies_3_meses_n525.csv").microbiome_species1
                if t2:
                    d = d >> apply(file2df, path + "data_microbiome___2023-06-18___alpha_diversity_n525.csv").microbiome_alpha2
                    if microbiome_extra:
                        d = d >> apply(file2df, path + "data_microbiome___2023-07-04___vias_metabolicas_valor_absoluto_T2_n441.csv").microbiome_species1
                        d = d >> apply(file2df, path + "data_microbiome___2023-06-18___especies_6_meses_n525.csv").microbiome_species2

            if eeg:  ########################################################################################################################
                if t1 or targets_eeg1:
                    d = d >> apply(file2df, path + "data_eeg___2023-06-20___T1_RS_average_dwPLI_withEEGCovariates.csv").eeg1
                if t2 or targets_eeg2:
                    d = d >> apply(file2df, path + "data_eeg___2023-06-20___T2_RS_average_dwPLI_withEEGCovariates.csv").eeg2
                if targets_eeg1:
                    d = d >> apply(DataFrame.__getitem__, _.eeg1, ["id_estudo"] + targets_eeg1).eeg1
                if targets_eeg2:
                    d = d >> apply(DataFrame.__getitem__, _.eeg2, ["id_estudo"] + targets_eeg2).eeg2

            # join #######################################################################################################################
            if microbiome:
                if t1:
                    d["df"] = _.microbiome_alpha1
                    if microbiome_extra:
                        d = d >> apply(join, other=_.microbiome_pathways1).df
                        d = d >> apply(join, other=_.microbiome_species1).df
                if t2:
                    if "df" not in d:
                        d["df"] = _.microbiome_alpha2
                    else:
                        d = d >> apply(join, other=_.microbiome_alpha2).df
                    if microbiome_extra:
                        d = d >> apply(join, other=_.microbiome_pathways2).df
                        d = d >> apply(join, other=_.microbiome_species2).df
            if eeg or targets_eeg1 or targets_eeg2:
                if t1 or targets_eeg1:
                    if "df" not in d:
                        d["df"] = _.eeg1
                    else:
                        d = d >> apply(join, other=_.eeg1).df
                if t2 or targets_eeg2:
                    if "df" not in d:
                        d["df"] = _.eeg2
                    else:
                        d = d >> apply(join, other=_.eeg2).df
            # d = d >> apply(remove_nan_rows_cols, cols_at_a_time=0, keep=["id_estudo"] + targets).df
            d = d >> cache(remote) >> cache(local)
            print("Joined------------------------------------------------------------------------\n", d.df, "______________________________________________________\n")

            # Join metadata #############################################################################################################
            if metavars:
                d = d >> apply(file2df, path + "metadata___2023-06-18.csv").metadata
                d = d >> apply(DataFrame.__getitem__, _.metadata, metavars + ["id_estudo"]).metadata
                print("Format problematic attributes.")
                d = d >> apply(bina, _.metadata, attribute="antibiotic", positive_category="yes").metadata
                d = d >> apply(bina, _.metadata, attribute="EBF_3m", positive_category="EBF").metadata
                d = d >> apply(loga, _.metadata, attribute="renda_familiar_total_t0").metadata
                d = d >> apply(join, other=_.metadata).df
                d = d >> cache(remote) >> cache(local)
                print("Metadata----------------------------------------------------------------------\n", d.df, "______________________________________________________\n")

            # Join targets ##############################################################################################################
            if targets_meta:
                d = d >> apply(file2df, path + "metadata___2023-06-18.csv").targets
                d = d >> apply(DataFrame.__getitem__, _.targets, targets + ["id_estudo"]).targets
                d = d >> apply(join, other=_.targets).df
                d = d >> cache(remote) >> cache(local)
            print("Dataset-----------------------------------------------------------------------\n", d.df, "______________________________________________________\n")

            # Remove NaNs ##################################################################################################################
            d = d >> apply(remove_nan_rows_cols, keep=["id_estudo"] + targets).df
            print("Dataset without NaNs ------------------------------------------------------------\n", d.df, "______________________________________________________\n")

            # Visualize ####################################################################################################################
            print("Vars:", d.df.columns)
            # d.df.to_csv(f"/tmp/{'-'.join(areas + ['metadata'])}.csv")
            # d.df: DataFrame
            # for target in targets:
            #     d.df[target].hist(bins=3)
            # plt.show()
            #

            # Train #######################################################################################################################
            if stratifiedcv:
                d = d >> apply(StratifiedKFold).cv
            else:
                d = d >> apply(KFold).cv
            for target in targets:
                print("=======================================================")
                print(target)
                print("=======================================================")

                # Prepare dataset.
                d = d >> apply(getattr, _.df, target).t
                if target.startswith("r_20hz_post_pre_waveleting_t"):
                    d["cuts"] = [0.4, 0.7]
                elif target.startswith("Beta_t"):
                    d["cuts"] = [0.005, 0.010]
                elif target.startswith("Number_Segs_Post_Seg_Rej_t"):
                    d["cuts"] = [40, 45]
                elif target.startswith("ibq_reg"):
                    d["cuts"] = [5, 6]

                d = d >> apply(getattr, _.df, target).t
                # print("t: ", list(d.df[target]))
                d = d >> apply(lambda x, cuts: np.digitize(x, cuts), _.t).t
                d = d >> apply(lambda df, t: df[t != 1], _.df, _.t).df
                d = d >> apply(remove_cols, _.df, targets, keep=[]).X
                d = d >> apply(getattr, _.df, target).y
                d = d >> apply(lambda x, cuts: np.digitize(x, cuts), _.y).y
                d = d >> apply(lambda y: y // 2, _.y).y

                d = d >> cache(remote) >> cache(local)
                print("X:", d.X.shape)
                print("y:", d.y.shape)

                clas_names = []
                clas = {
                    DummyClassifier: {},
                    RFc: {},
                    XGBc: {},
                    # CATc: {"subsample": 0.1},
                    LGBMc: {},
                    ETc: {},
                    SGDc: {},
                }
                for cla, kwargs in clas.items():
                    clas_names.append(cla.__name__)
                    # print(clas_names[-1])
                    d = d >> apply(cla, **kwargs)(clas_names[-1])

                # Prediction power.
                ['accuracy', 'adjusted_mutual_info_score', 'adjusted_rand_score', 'average_precision',
                 'balanced_accuracy', 'completeness_score', 'explained_variance',
                 'f1', 'f1_macro', 'f1_micro', 'f1_samples', 'f1_weighted', 'fowlkes_mallows_score',
                 'homogeneity_score', 'jaccard', 'jaccard_macro', 'jaccard_micro', 'jaccard_samples', 'jaccard_weighted',
                 'matthews_corrcoef', 'max_error', 'mutual_info_score',
                 'neg_brier_score', 'neg_log_loss', 'neg_mean_absolute_error', 'neg_mean_absolute_percentage_error', 'neg_mean_gamma_deviance', 'neg_mean_poisson_deviance', 'neg_mean_squared_error', 'neg_mean_squared_log_error', 'neg_median_absolute_error', 'neg_negative_likelihood_ratio', 'neg_root_mean_squared_error',
                 'normalized_mutual_info_score', 'positive_likelihood_ratio',
                 'precision', 'precision_macro', 'precision_micro', 'precision_samples', 'precision_weighted',
                 'r2', 'rand_score',
                 'recall', 'recall_macro', 'recall_micro', 'recall_samples', 'recall_weighted',
                 'roc_auc', 'roc_auc_ovo', 'roc_auc_ovo_weighted', 'roc_auc_ovr', 'roc_auc_ovr_weighted', 'top_k_accuracy', 'v_measure_score']
                scos = ["precision", "recall", "balanced_accuracy", "roc_auc"]
                for m in scos:
                    print(m)
                    print("vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv")
                    for classifier_field in clas_names:
                        scores_fi = f"{m}_{classifier_field}"
                        permscores_fi = f"perm_{scores_fi}"
                        pval_fi = f"pval_{scores_fi}"
                        # d = d >> apply(cross_val_score, field(classifier_field), _.X, _.y, cv=_.cv, scoring=m)(scores_fi)
                        d = d >> apply(permutation_test_score, field(classifier_field), _.X, _.y, cv=_.cv, scoring=m)(scores_fi, permscores_fi, pval_fi)
                        d = d >> cache(remote) >> cache(local)
                        me = mean(d[scores_fi])
                        if classifier_field == "DummyClassifier":
                            ref = me
                        print(f"{classifier_field:24} {me:.6f} {std(d[scores_fi]):.6f}{'    <----' if me > ref else ''} p-value={d[pval_fi]}")
                    print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")

                # ConfusionMatrix; prediction and hit agreement.
                zs, hs = {}, {}
                for classifier_field in clas_names:
                    print(classifier_field)
                    field_name_z = f"{classifier_field}_z"
                    d = d >> apply(cross_val_predict, field(classifier_field), _.X, _.y, cv=_.cv)(field_name_z)
                    d = d >> cache(remote) >> cache(local)
                    z = d[field_name_z]
                    zs[classifier_field[:10]] = z
                    hs[classifier_field[:10]] = (z == d.y).astype(int)
                    print(f"{confusion_matrix(d.y, z)}")
                print("Prediction:")
                Z = array(list(zs.values()))
                zs["   AND    "] = np.logical_and.reduce(Z, axis=0).astype(int)
                zs["   OR     "] = np.logical_or.reduce(Z, axis=0).astype(int)
                zs["   SUM    "] = np.sum(Z, axis=0).astype(int)
                zs["   NOR    "] = np.logical_not(np.logical_or.reduce(Z, axis=0)).astype(int)
                zs["   ==     "] = (np.logical_and.reduce(Z, axis=0) | np.logical_not(np.logical_or.reduce(Z, axis=0))).astype(int)
                for k, z in zs.items():
                    if "AND" in k:
                        print()
                    print(k, sum(z), ",".join(map(str, z)))
                print()
                print("Hit:")
                H = array(list(hs.values()))
                hs["   AND    "] = np.logical_and.reduce(H, axis=0).astype(int)
                hs["   OR     "] = np.logical_or.reduce(H, axis=0).astype(int)
                hs["   SUM    "] = np.sum(H, axis=0).astype(int)
                hs["   NOR    "] = np.logical_not(np.logical_or.reduce(H, axis=0)).astype(int)
                hs["   ==     "] = (np.logical_and.reduce(H, axis=0) | np.logical_not(np.logical_or.reduce(H, axis=0))).astype(int)
                for k, h in hs.items():
                    if "AND" in k:
                        print()
                    print(k, sum(h), "\t", ",".join(map(str, h)))
                print()

                # Importances
                for classifier_field in clas_names:
                    model = f"{target}_{classifier_field}_model"
                    d = d >> apply(lambda c, *args, **kwargs: clone(c).fit(*args, **kwargs), field(classifier_field), _.X, _.y)(model)
                    importances_field_name = f"{target}_{classifier_field}_importances"
                    d = d >> apply(permutation_importance, field(model), _.X, _.y, n_repeats=20, scoring=scos, n_jobs=-1)(importances_field_name)
                    d = d >> cache(remote) >> cache(local)
                    fst = True
                    for metric in d[importances_field_name]:
                        r = d[importances_field_name][metric]
                        for i in r.importances_mean.argsort()[::-1]:
                            if r.importances_mean[i] - r.importances_std[i] > 0:
                                if fst:
                                    print(f"Importances {classifier_field:<20} ----------------------------")
                                    fst = False
                                print(f"  {metric:<17} {d.X.columns[i][-25:]:<17} {r.importances_mean[i]:.6f} +/- {r.importances_std[i]:.6f}")
                        # if not fst:
                        #     print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
                        print()
                print()
        # sys.stdout = oldout
        # d.show()
        # sys.stdout = newout

    sys.stdout = oldout
