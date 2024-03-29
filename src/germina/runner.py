import sys
from contextlib import nullcontext
from pprint import pprint

import numpy as np
import pandas as pd
from hdict import _, apply, cache
from hdict import field
from hdict import hdict
from hdict.dataset.pandas_handling import file2df
from hosh import Hosh
from imodels.tree.figs import FIGS, FIGSClassifier
from lightgbm import LGBMClassifier as LGBMc
from numpy import array, quantile, where, extract, argsort
from numpy import mean, std
from pandas import DataFrame
from shelchemy import sopen
from shelchemy.scheduler import Scheduler
from sklearn import clone
from sklearn.ensemble import ExtraTreesClassifier as ETc, StackingClassifier
from sklearn.ensemble import RandomForestClassifier as RFc
from sklearn.inspection import permutation_importance
from sklearn.linear_model import SGDClassifier as SGDc
from sklearn.metrics import confusion_matrix, balanced_accuracy_score
from sklearn.model_selection import KFold, cross_val_predict, StratifiedKFold, permutation_test_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
from xgboost import XGBClassifier as XGBc

from germina.config import remote_cache_uri, local_cache_uri, schedule_uri
from germina.dataset import join, ensemble_predict, concat
from germina.nan import remove_cols, bina, loga, remove_nan_rows_cols, only_abundant, hasNaN, remove_nan_cols_rows


def setindex(df, old_indexname="id_estudo"):
    if df.index is not None and df.index.name != "id_estudo":
        df = df.reset_index()
        df.rename(columns={old_indexname: "id_estudo"}, inplace=True)
        df.set_index("id_estudo", inplace=True)
    return df


def sgid2estudoid(df: DataFrame, path="data", filename="anthonieta---ChecklistHyperscanning-Idade.csv"):
    translator_df = file2df(f"{path}/{filename}", return_name=False)
    if df.index.name in ["IDs", "ID:"]:
        df = df.reset_index()
    if "IDs" in df.columns:
        df = df.assign(ID=df["IDs"].to_list())
        df.drop("IDs", axis=1, inplace=True)
    if "ID:" in df.columns:
        df = df.assign(ID=df["ID:"].to_list())
        df.drop("ID:", axis=1, inplace=True)
    df = pd.merge(df, translator_df[["ID", "id_estudo"]], on="ID", how="inner")
    df.set_index("id_estudo", inplace=True)
    df.drop("ID", axis=1, inplace=True)
    df = setindex(df)
    return df


def bestid(df, path="data", filename="anthonieta---ChecklistHyperscanning-Idade.csv"):
    translator_df = file2df(f"{path}/{filename}", return_name=False)
    df = pd.merge(df, translator_df[["MelhorTparaEEGtradicional", "id_estudo"]], on="id_estudo", how="left")
    df.set_index("id_estudo", inplace=True)
    df.apply(lambda x: print(x), axis=1)
    df.drop("MelhorTparaEEGtradicional", axis=1, inplace=True)
    exit()
    # todo
    return df


def ch(d, storages, to_be_updated=""):
    for storage in storages.values():
        d = d >> cache(storage)
    if to_be_updated != "":
        d = d >> cache(storages[to_be_updated])
    d.evaluate()
    return d


def drop_many_by_vif(d, dffield, storages, to_be_updated, keepcols, keeprows):
    d = d >> apply(lambda df: df.loc[keeprows], d[dffield]).dfkeep
    if hasNaN(d[dffield], debug=False) > 1:
        d = d >> apply(remove_nan_rows_cols, field(dffield), keep=[], cols_at_a_time=2)(dffield)
    d = d >> apply(lambda df: df.astype(float), field(dffield))(dffield)
    lstfield = f"{dffield}_dropped"
    d[lstfield] = old = []
    while True:
        d = d >> apply(drop_by_vif, df=field(dffield), dropped=_[lstfield])(lstfield)
        d = ch(d, storages, to_be_updated)
        if d[lstfield] == old:
            break
        old = d[lstfield]
    d = d >> apply(lambda df, dfkeep: dfkeep.loc[dfkeep.index.difference(df.index)], field(dffield)).dfkeep
    d = d >> apply(concat, field(dffield), other=_.dfkeep)(dffield)
    d = d >> apply(remove_cols, field(dffield), field(lstfield), keep=keepcols, debug=False)(dffield)
    d = ch(d, storages, to_be_updated)
    return d


def drop_by_vif(df: DataFrame, dropped=None, thresh=5.0):
    """https://stats.stackexchange.com/a/253620/36979"""
    dropped = [] if dropped is None else dropped.copy()
    X = df.assign(const=1)  # faster than add_constant from statsmodels
    X = remove_cols(X, dropped, [], debug=False)
    variables = list(range(X.shape[1]))
    vif = [variance_inflation_factor(X.iloc[:, variables].values, ix) for ix in range(X.iloc[:, variables].shape[1])]
    vif = vif[:-1]
    maxloc = vif.index(max(vif))
    if max(vif) > thresh:
        dropped.append(X.iloc[:, variables].columns[maxloc])
        print(f"Dropped: {dropped[::-1]}")
    return dropped


def run(
    d: hdict,
    high_is_positive,
    mn,
    t1=False,
    t2=False,
    just_df=False,
    vif=True,
    scheduler=True,
    printing=True,
    eeg=False,
    eegpow=False,
    malpha=False,
    mpathways=False,
    mspecies=False,
    msuper=False,
    metavars=None,
    targets_meta=None,
    targets_eeg1=None,
    targets_eeg2=None,
    stratifiedcv=True,
    path="data/",
    loc=True,
    rem=True,
    sync=False,
    verbose=False,
):
    # d.show()
    dct = d.dct.copy() if isinstance(d.dct, dict) else dict(d.dct)
    print(dct)
    dct["t1"] = t1
    dct["t2"] = t2
    dct["stratifiedcv"] = stratifiedcv
    d["dct"] = list(sorted(dct.items()))
    d.hosh.show()
    malpha1 = malpha2 = eegpow1 = eegpow2 = eeg1 = eeg2 = pathways1 = pathways2 = species1 = species2 = super1 = super2 = False
    if eeg:
        eeg1, eeg2 = t1, t2
    if eegpow:
        eegpow1, eegpow2 = t1, t2
    if malpha:
        malpha1, malpha2 = t1, t2
    if mpathways:
        pathways1, pathways2 = t1, t2
    if mspecies:
        species1, species2 = t1, t2
    if msuper:
        super1, super2 = t1, t2
    logname = f"{d.ids['dct']}-{t1=}{t2=}{malpha=}{mpathways=}{mspecies=}{msuper=}{eeg=}{eegpow=}{targets_meta=}{targets_eeg1=}{targets_eeg2=}{[metavars]=}{stratifiedcv=}"
    logname = Hosh(logname.encode()).id + logname[:200]
    if verbose:
        print(logname)
    with nullcontext() if printing else open(
        "out/output-" + logname[40:96] + f"-tgteeg1={bool(targets_eeg1)}-tgteeg2={bool(targets_eeg2)}-{targets_meta and targets_meta[0][-1]}-germina.txt", "w"
    ) as ctx:
        if not printing:
            old = sys.stdout
            sys.stdout = ctx
        print(f"Scenario: {t1=}, {t2=}, {malpha=}, {mpathways=}, {mspecies=}, {msuper=}, {eeg=}, {eegpow=},\n")
        if verbose:
            print(f"{metavars=},\n" f"{targets_meta=},\n" f"{targets_eeg1=},\n" f"{targets_eeg2=}")
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
            #################################################################################################################
            if malpha1:
                d = d >> apply(file2df, path + "data_microbiome___2023-06-18___alpha_diversity_n525.csv").microbiome_alpha1
                if vif:
                    d = drop_many_by_vif(d, "microbiome_alpha1", loc, rem, local, remote, sync)
            if pathways1:
                d = d >> apply(file2df, path + "data_microbiome___2023-07-04___vias_metabolicas_valor_absoluto_T1_n525.csv").microbiome_pathways1
                d = d >> apply(only_abundant, _.microbiome_pathways1).microbiome_pathways1
                if vif:
                    d = drop_many_by_vif(d, "microbiome_pathways1", loc, rem, local, remote, sync)
            if species1:
                d = d >> apply(file2df, path + "data_microbiome___2023-06-18___especies_3_meses_n525.csv").microbiome_species1
                d = d >> apply(only_abundant, _.microbiome_species1).microbiome_species1
                if vif:
                    d = drop_many_by_vif(d, "microbiome_species1", loc, rem, local, remote, sync)
            if super1:
                d = d >> apply(file2df, path + "data_microbiome___2023-07-04___T1_vias_relab_superpathways.csv").microbiome_super1
                if vif:
                    d = drop_many_by_vif(d, "microbiome_super1", loc, rem, local, remote, sync)
            if malpha2:
                d = d >> apply(file2df, path + "data_microbiome___2023-07-03___alpha_diversity_T2_n441.csv").microbiome_alpha2
                if vif:
                    d = drop_many_by_vif(d, "microbiome_alpha2", loc, rem, local, remote, sync)
            if pathways2:
                d = d >> apply(file2df, path + "data_microbiome___2023-07-04___vias_metabolicas_valor_absoluto_T2_n441.csv").microbiome_pathways2
                d = d >> apply(only_abundant, _.microbiome_pathways2).microbiome_pathways2
                if vif:
                    d = drop_many_by_vif(d, "microbiome_pathways2", loc, rem, local, remote, sync)
            if species2:
                d = d >> apply(file2df, path + "data_microbiome___2023-06-29___especies_6_meses_n441.csv").microbiome_species2
                d = d >> apply(only_abundant, _.microbiome_species2).microbiome_species2
                if vif:
                    d = drop_many_by_vif(d, "microbiome_species2", loc, rem, local, remote, sync)
            if super2:
                d = d >> apply(file2df, path + "data_microbiome___2023-07-04___T2_vias_relab_superpathways.csv").microbiome_super2
                if vif:
                    d = drop_many_by_vif(d, "microbiome_super2", loc, rem, local, remote, sync)

            ########################################################################################################################
            if (eeg1 and not targets_eeg2) or targets_eeg1:
                d = d >> apply(file2df, path + "data_eeg___2023-06-20___T1_RS_average_dwPLI_withEEGCovariates.csv").eeg1
                if not targets_eeg1 and vif:
                    d = drop_many_by_vif(d, "eeg1", loc, rem, local, remote, sync)
            if eegpow1 and not (targets_eeg1 or targets_eeg2):
                d = d >> apply(file2df, path + "data_eeg___2023-07-19___BRAINRISE_RS_3m_power.csv").eegpow1
                if vif:
                    d = drop_many_by_vif(d, "eegpow1", loc, rem, local, remote, sync)
            if targets_eeg1:
                d = d >> apply(DataFrame.__getitem__, _.eeg1, ["id_estudo"] + targets_eeg1).eeg1

            if (eeg2 and not targets_eeg1) or targets_eeg2:
                d = d >> apply(file2df, path + "data_eeg___2023-06-20___T2_RS_average_dwPLI_withEEGCovariates.csv").eeg2
                d = d >> apply(remove_nan_rows_cols, _.eeg2, keep=[]).eeg2
                if not targets_eeg2 and vif:
                    d = drop_many_by_vif(d, "eeg2", loc, rem, local, remote, sync)
            if eegpow2 and not (targets_eeg1 or targets_eeg2):
                d = d >> apply(file2df, path + "data_eeg___2023-07-19___BRAINRISE_RS_T2_Power.csv").eegpow2
                d = d >> apply(remove_nan_rows_cols, _.eegpow2, keep=[]).eegpow2
                if vif:
                    d = drop_many_by_vif(d, "eegpow2", loc, rem, local, remote, sync)
            if targets_eeg2:
                d = d >> apply(DataFrame.__getitem__, _.eeg2, ["id_estudo"] + targets_eeg2).eeg2
            d = ch(d, loc, rem, local, remote, sync)

            # join #######################################################################################################################
            new_or_join = lambda fi: field(fi) if "df" not in d else apply(join, other=field(fi))
            if t1:
                if malpha1:
                    d["df"] = new_or_join("microbiome_alpha1")
                if pathways1:
                    d["df"] = new_or_join("microbiome_pathways1")
                if species1:
                    d["df"] = new_or_join("microbiome_species1")
                if super1:
                    d["df"] = new_or_join("microbiome_super1")
                d = ch(d, loc, rem, local, remote, sync)
            if t2:
                if malpha2:
                    d["df"] = new_or_join("microbiome_alpha2")
                if pathways2:
                    d["df"] = new_or_join("microbiome_pathways2")
                if species2:
                    d["df"] = new_or_join("microbiome_species2")
                if super2:
                    d["df"] = new_or_join("microbiome_super2")
                d = ch(d, loc, rem, local, remote, sync)
            if eeg or targets_eeg1 or targets_eeg2:
                if (t1 and not targets_eeg2) or targets_eeg1:
                    if "df" not in d:
                        d["df"] = _.eeg1
                    else:
                        d = d >> apply(join, other=_.eeg1).df
                    if "eegpow1" in d and not (targets_eeg1 or targets_eeg2):
                        d = d >> apply(join, other=_.eegpow1).df
                if t2 or targets_eeg2:
                    if "df" not in d:
                        d["df"] = _.eeg2
                    else:
                        d = d >> apply(join, other=_.eeg2).df
                    if "eegpow2" in d and not (targets_eeg1 or targets_eeg2):
                        d = d >> apply(join, other=_.eegpow2).df
            # d = d >> apply(remove_nan_rows_cols, cols_at_a_time=0, keep=["id_estudo"] + targets).df
            d = ch(d, loc, rem, local, remote, sync)
            if verbose:
                print("Joined------------------------------------------------------------------------\n", d.df, "______________________________________________________\n")

            # Join metadata #############################################################################################################
            if metavars:
                d = d >> apply(file2df, path + "metadata___2023-07-17.csv").metadata
                d = d >> apply(DataFrame.__getitem__, _.metadata, metavars + ["id_estudo"]).metadata
                if verbose:
                    print("Format problematic attributes.")
                d = d >> apply(bina, _.metadata, attribute="antibiotic", positive_category="yes").metadata
                d = d >> apply(bina, _.metadata, attribute="EBF_3m", positive_category="EBF").metadata
                d = d >> apply(loga, _.metadata, attribute="renda_familiar_total_t0").metadata
                d["df"] = new_or_join("metadata")
                d = d >> apply(remove_nan_rows_cols, keep=["id_estudo"] + targets).df
                d = ch(d, loc, rem, local, remote, sync)
                if verbose:
                    print("Metadata----------------------------------------------------------------------\n", d.df, "______________________________________________________\n")

            ##############################   VIF    ######################################
            # d = d >> apply(remove_cols, cols=dropped, keep=[]).df
            if vif:
                d = drop_many_by_vif(d, "df", loc, rem, local, remote, sync)

            # Join targets ##############################################################################################################
            if targets_meta:
                d = d >> apply(file2df, path + "metadata___2023-06-18.csv").targets
                d = d >> apply(DataFrame.__getitem__, _.targets, targets + ["id_estudo"]).targets
                d = d >> apply(join, other=_.targets).df
                d = ch(d, loc, rem, local, remote, sync)
            if verbose:
                print("Dataset-----------------------------------------------------------------------\n", d.df, "______________________________________________________\n")

            # Remove NaNs ##################################################################################################################
            d = d >> apply(remove_nan_rows_cols, keep=["id_estudo"] + targets).df
            d = d >> apply(lambda df: df.columns.to_list()).columns
            d = ch(d, loc, rem, local, remote, sync)
            if verbose:
                print("Dataset without NaNs ------------------------------------------------------------\n", d.df, "______________________________________________________\n")

            # Visualize ####################################################################################################################
            print("Vars:", d.columns)
            # d.df.to_csv(f"/tmp/all.csv")
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
            d = d >> apply(StratifiedKFold, n_splits=3).cv3
            dfs = {}
            for target in targets:
                print("=======================================================")
                print(target)
                print("=======================================================")

                # Prepare dataset.
                d = d >> apply(getattr, _.df, target).t

                def qtl(x, mn):
                    q = quantile(x, [1 / 5, 4 / 5])
                    l = extract(x <= q[0], x)
                    h = extract(x >= q[1], x)
                    w = min(len(l), len(h)) if mn else max(len(l), len(h))
                    ix = argsort(x)
                    x.iloc[:] = 1
                    x.iloc[ix[:w]] = 0
                    x.iloc[ix[-w:]] = 2
                    return x.astype(int)

                d = d >> apply(qtl, _.t, mn).t
                d = d >> apply(lambda df, t: df[t != 1], _.df, _.t).dfcut
                d = d >> apply(remove_cols, _.dfcut, targets, keep=[]).X
                # print(targets)
                # print(d.X.columns.to_list())
                d = d >> apply(lambda t: t[t != 1]).t
                if high_is_positive:
                    d = d >> apply(lambda t: t // 2).y
                else:
                    d = d >> apply(lambda t: t // 2 ^ 1).y
                # print(d.y)

                d = ch(d, loc, rem, local, remote, sync)
                if just_df:
                    dfs[target] = d.X, d.y
                    continue

                d = d >> apply(lambda X: X.shape).Xshape
                d = d >> apply(lambda y: y.shape).yshape
                d = d >> apply(lambda y: np.unique(y, return_counts=True))("unique_labels", "counts")
                d = d >> apply(lambda y, counts: counts / len(y)).proportions
                d = ch(d, loc, rem, local, remote, sync)
                print("X:", d.Xshape)
                if verbose:
                    print("y:", d.yshape)
                print(f"{d.counts=}\t{d.proportions=}")

                clas_names = []
                clas = {
                    # DummyClassifier: {},
                    RFc: {},
                    XGBc: {},
                    # CATc: {"subsample": 0.1},
                    LGBMc: {},
                    ETc: {},
                }
                d["estimators"] = []
                for cla, kwargs in clas.items():
                    nam = cla.__name__
                    clas_names.append(nam)
                    d = d >> apply(cla, **kwargs)(nam)
                    d = d >> apply(lambda na, _: _.estimators + [(na, _[na])], nam).estimators
                d = ch(d, loc, rem, local, remote, sync)
                if d.stacking:
                    clas_names.append("StackingClassifier")
                    d = d >> apply(StackingClassifier, cv=_.stacking_cv, final_estimator=_.stacking_final_estimator).StackingClassifier

                # Interpretable.
                # clas_names.append("FIGSClassifier")
                # d = d >> apply(FIGSClassifier).FIGSClassifier

                scos = d.measures

                with sopen(schedule_uri) as db:
                    for m in scos:
                        print(m, end="\t")
                        d.hosh.show()
                        cor = (d.hoshes["dct"] * Hosh(m.encode())).ansi

                        # Prediction power.
                        jobs = [f"{cn:<25} {target} PERMs {m} {cor} {'+' if high_is_positive else ''} {'+mn' if mn else ''}" for cn in clas_names]
                        tasks = (Scheduler(db, timeout=20) << jobs) if scheduler else jobs
                        for task in tasks:
                            print(m, end="\t")
                            classifier_field = task.split(" ")[0]
                            scores_fi = f"{m}_{classifier_field}"
                            permscores_fi = f"perm_{scores_fi}"
                            pval_fi = f"pval_{scores_fi}"
                            # d = d >> apply(cross_val_score, field(classifier_field), _.X, _.y, cv=_.cv, scoring=m)(scores_fi)
                            d.hosh.show()  # todo: por que stacking não está cacheando no roc?
                            d = d >> apply(permutation_test_score, field(classifier_field), _.X, _.y, cv=_.cv, scoring=m)(scores_fi, permscores_fi, pval_fi)
                            d = ch(d, loc, rem, local, remote, sync)
                            print(f"classification\tp-value={d[pval_fi]:.6f}\t{mean(d[scores_fi]):.6f}\t{std(d[scores_fi]):.6f}\t{m:22}\t{classifier_field:24}\t{target:20}")

                        # ConfusionMatrix; importance
                        jobs = [f"{cn:<25} {target} importance {m} {cor} {'+' if high_is_positive else ''} {'+mn' if mn else ''}" for cn in clas_names]
                        tasks = (Scheduler(db, timeout=20) << jobs) if scheduler else jobs
                        for task in tasks:
                            print(m, end="\t")
                            d.hosh.show()
                            classifier_field = task.split(" ")[0]
                            if verbose:
                                print(classifier_field)
                            field_name_z = f"{classifier_field}_z"
                            field_name_p = f"{classifier_field}_p"
                            d = d >> apply(cross_val_predict, field(classifier_field), _.X, _.y, cv=_.cv)(field_name_z)
                            d = d >> apply(cross_val_predict, field(classifier_field), _.X, _.y, cv=_.cv, method="predict_proba")(field_name_p)
                            d = d >> apply(lambda y, z: confusion_matrix(y, z), z=_[field_name_z]).confusion_matrix
                            d = ch(d, loc, rem, local, remote, sync)

                            print(f"hit&miss,{classifier_field},{','.join(str(x) for x in (~(d[field_name_z].astype(bool) ^ d.y.astype(bool))).astype(int).tolist())}")
                            print("--------------------")

                            if verbose:
                                print(f"{d.confusion_matrix}")

                            # Importances
                            model = f"{target}_{classifier_field}_model"
                            d = d >> apply(lambda c, *args, **kwargs: clone(c).fit(*args, **kwargs), field(classifier_field), _.X, _.y)(model)
                            importances_field_name = f"{target}_{classifier_field}_importances"
                            d = d >> apply(permutation_importance, field(model), _.X, _.y, n_repeats=100, scoring=m, n_jobs=-1)(importances_field_name)
                            d = ch(d, loc, rem, local, remote, sync)
                            r = d[importances_field_name]
                            for i in r.importances_mean.argsort()[::-1]:
                                if r.importances_mean[i] - r.importances_std[i] > 0:
                                    print(f"importance   \t                 \t{r.importances_mean[i]:.6f}\t{r.importances_std[i]:.6f}\t{m:22}\t{classifier_field:24}\t{target:20}\t{d.columns[i]}")
                        print()
                    print()
    if not printing:
        sys.stdout = old
    print("Scenario finished")
    return dfs


def run_t1_t2(
    d: hdict,
    eeg=False,
    eegpow=False,
    malpha=False,
    mpathways=False,
    mspecies=False,
    msuper=False,
    metavars=None,
    stratifiedcv=True,
    path="data/",
    loc=True,
    rem=True,
    verbose=False,
    sync=False,
    **kwargs,
):
    kwargs |= dict(
        eeg=eeg,
        eegpow=eegpow,
        malpha=malpha,
        mpathways=mpathways,
        mspecies=mspecies,
        msuper=msuper,
        metavars=metavars,
        stratifiedcv=stratifiedcv,
        path=path,
        loc=loc,
        rem=rem,
        verbose=verbose,
        sync=sync,
    )
    #       t1 → t1
    run(hdict.fromdict(d, d.ids), t1=True, targets_meta=["ibq_reg_t1", "ibq_soot_t1", "ibq_dura_t1", "bayley_3_t1"], **kwargs)
    run(hdict.fromdict(d, d.ids), t1=True, targets_eeg1=["Beta_t1", "r_20hz_post_pre_waveleting_t1", "Number_Segs_Post_Seg_Rej_t1"], **kwargs)
    #       t1 → t2
    run(hdict.fromdict(d, d.ids), t1=True, targets_meta=["ibq_reg_t2", "ibq_soot_t2", "ibq_dura_t2", "bayley_3_t2"], **kwargs)
    run(hdict.fromdict(d, d.ids), t1=True, targets_eeg2=["Beta_t2", "r_20hz_post_pre_waveleting_t2", "Number_Segs_Post_Seg_Rej_t2"], **kwargs)
    #       t1+t2 → t2
    run(hdict.fromdict(d, d.ids), t1=True, t2=True, targets_meta=["ibq_reg_t2", "ibq_soot_t2", "ibq_dura_t2", "bayley_3_t2"], **kwargs)
    run(hdict.fromdict(d, d.ids), t1=True, t2=True, targets_eeg2=["Beta_t2", "r_20hz_post_pre_waveleting_t2", "Number_Segs_Post_Seg_Rej_t2"], **kwargs)

    """ 
                                # zs[classifier_field[:10]] = z
                                # hs[classifier_field[:10]] = (z == d.y).astype(int)
                    # d = d >> apply(ensemble_predict, *members_z).ensemble_z
                    # d = ch(d, loc, rem, local, remote, sync)
                    # 
                    # # Accuracy
                    # # for classifier_field in clas_names:
                    # #     field_name_z = f"{classifier_field}_z"
                    # #     fieldbalacc = f"{classifier_field}_balacc"
                    # #     d = d >> apply(balanced_accuracy_score, _.y, field(field_name_z), adjusted=True)(fieldbalacc)
                    # #     d = ch(d, loc, rem, local, remote, sync)
                    # #     print(f"{classifier_field:24} {d[fieldbalacc]:.6f} ")
                    # d = d >> apply(balanced_accuracy_score, _.y, _.ensemble_z).ensemble_balacc
                    # d = ch(d, loc, rem, local, remote, sync)
                    # print(f"ensemble5 {d.ensemble_balacc:.6f} ")
    
                    # if verbose:
                    #     print("Prediction:")
                    #     Z = array(list(zs.values()))
                    #     zs["   AND    "] = np.logical_and.reduce(Z, axis=0).astype(int)
                    #     zs["   OR     "] = np.logical_or.reduce(Z, axis=0).astype(int)
                    #     zs["   SUM    "] = np.sum(Z, axis=0).astype(int)
                    #     zs["   NOR    "] = np.logical_not(np.logical_or.reduce(Z, axis=0)).astype(int)
                    #     zs["   ==     "] = (np.logical_and.reduce(Z, axis=0) | np.logical_not(np.logical_or.reduce(Z, axis=0))).astype(int)
                    #     for k, z in zs.items():
                    #         if "AND" in k:
                    #             print()
                    #         # print(k, sum(z), ",".join(map(str, z)))
                    #     print()
                    #     print("Hit:")
                    #     H = array(list(hs.values()))
                    #     hs["   AND    "] = np.logical_and.reduce(H, axis=0).astype(int)
                    #     hs["   OR     "] = np.logical_or.reduce(H, axis=0).astype(int)
                    #     hs["   SUM    "] = np.sum(H, axis=0).astype(int)
                    #     hs["   NOR    "] = np.logical_not(np.logical_or.reduce(H, axis=0)).astype(int)
                    #     hs["   ==     "] = (np.logical_and.reduce(H, axis=0) | np.logical_not(np.logical_or.reduce(H, axis=0))).astype(int)
                    #     for k, h in hs.items():
                    #         if "AND" in k:
                    #             print()
                    #         # print(k, sum(h), "\t", ",".join(map(str, h)))
                    #     print()
    """
