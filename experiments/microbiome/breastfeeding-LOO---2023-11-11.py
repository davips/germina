if __name__ == '__main__':
    import numpy as np
    from dalex.model_explanations import VariableImportance
    import pandas as pd
    from multiprocessing import freeze_support
    import dalex as dx
    import warnings
    from datetime import datetime
    from itertools import repeat

    from pandas import DataFrame
    from shelchemy.scheduler import Scheduler
    from sklearn import clone
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.inspection import permutation_importance
    from xgboost import XGBClassifier

    warnings.filterwarnings('ignore')
    from pprint import pprint
    from sys import argv

    from argvsucks import handle_command_line
    from germina.runner import ch
    from hdict import hdict, apply, _
    from shelchemy import sopen
    from sklearn.experimental import enable_iterative_imputer

    from germina.config import local_cache_uri, remote_cache_uri, near_cache_uri, schedule_uri
    from germina.dataset import join
    from germina.loader import load_from_csv, clean_for_dalex, get_balance, train_xgb, importances, importances2, aaa, start_reses, ccc

    from sklearn.model_selection import LeaveOneOut, permutation_test_score, StratifiedKFold, cross_val_score

    import warnings

    warnings.filterwarnings('ignore')
    __ = enable_iterative_imputer
    dct = handle_command_line(argv, pvalruns=int, importanceruns=int, imputertrees=int, seed=int, target=str, trees=int, vif=False, nans=False, sched=False, up="", measures=list)
    print(datetime.now())
    pprint(dct, sort_dicts=False)
    print()
    path = "data/paper-breastfeeding/"
    d = hdict(
        n_permutations=dct["pvalruns"],
        n_repeats=dct["importanceruns"],
        imputation_trees=dct["imputertrees"],
        random_state=dct["seed"],
        target_var=dct["target"],
        measures=dct["measures"],
        max_iter=dct["trees"], n_estimators=dct["trees"],
        n_splits=5,
        shuffle=True,
        index="id_estudo", join="inner", n_jobs=20, return_name=False,
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
        d["res"] = {}
        d["res_importances"] = {}
        for measure in d.measures:
            d = d >> apply(start_reses, measure=measure)("res", "res_importances")
            d = ch(d, storages, storage_to_be_updated)

        for arq, field, oldidx in [("t_3-4_pathways_filtered", "pathways34", "Pathways"),
                                   ("t_3-4_species_filtered", "species34", "Species"),
                                   ("t_5-7_pathways_filtered", "pathways57", "Pathways"),
                                   ("t_5-7_species_filtered", "species57", "Species"),
                                   ("t_8-9_pathways_filtered", "pathways89", "Pathways"),
                                   ("t_8-9_species_filtered", "species89", "Species")]:
            d["field"] = field
            print(field, "=================================================================================")
            d = load_from_csv(d, storages, storage_to_be_updated, path, vif, arq, field, transpose=True, old_indexname=oldidx)
            d = load_from_csv(d, storages, storage_to_be_updated, path, False, "EBF_parto", "ebf", False)

            d = d >> apply(join, df=_.ebf, other=_[field]).df
            d = ch(d, storages, storage_to_be_updated)

            d = clean_for_dalex(d, storages, storage_to_be_updated)
            d = d >> apply(lambda X: X.copy(deep=True)).X0
            d = d >> apply(lambda y: y.copy(deep=True)).y0
            d = ch(d, storages, storage_to_be_updated)

            for parto in ["c_section", "vaginal"]:
                print(parto)
                d["parto"] = parto
                d = d >> apply(lambda X0, parto: X0[X0["delivery_mode"] == parto]).X
                d = d >> apply(lambda X: X.drop("delivery_mode", axis=1)).X
                d = d >> apply(lambda X0, y0, parto: y0[X0["delivery_mode"] == parto]).y
                d = ch(d, storages, storage_to_be_updated)
                d = get_balance(d, storages, storage_to_be_updated)

                params = {"max_depth": 5, "objective": "binary:logistic", "eval_metric": "auc"}
                loo = LeaveOneOut()
                runs = list(loo.split(d.X))
                d = d >> apply(RandomForestClassifier).alg

                for m in d.measures:
                    # calcula baseline score e p-values
                    d["scoring"] = m
                    d["field"] = field
                    score_field, permscores_field, pval_field = f"{m}_score", f"{m}_permscores", f"{m}_pval"

                    tasks = [(field, parto, f"{vif=}", m, f"trees={d.n_estimators}")]
                    for __, __, __, __, __ in (Scheduler(db, timeout=60) << tasks) if sched else tasks:
                        d = d >> apply(permutation_test_score, _.alg)(score_field, permscores_field, pval_field)
                        d = ch(d, storages, storage_to_be_updated)
                        d = d >> apply(ccc, d_score=_[score_field], d_pval=_[pval_field]).res
                        d = ch(d, storages, storage_to_be_updated)

                    d = d >> apply(lambda res: res).res
                    d = ch(d, storages, storage_to_be_updated)

                    # LOO importances
                    importances_mean, importances_std = [], []
                    tasks = zip(repeat((field, parto, f"{vif=}", m, f"trees={d.n_estimators}")), range(len(runs)))
                    d["contribs_accumulator"] = None
                    for (fi, pa, vi, __, __), i in (Scheduler(db, timeout=60) << tasks) if sched else tasks:
                        d["idxtr", "idxts"] = runs[i]
                        print(f"\t{i}\t{fi}\t{pa}\t{vi}\tts:{d.idxts}\t", datetime.now(), f"\t{100 * i / len(d.X):1.1f} %\t-----------------------------------")

                        d = d >> apply(lambda X, y, idxtr, idxts: (X.iloc[idxtr], y.iloc[idxtr], X.iloc[idxts], y.iloc[idxts]))("Xtr", "ytr", "Xts", "yts")
                        # # reminder: don't store 'model'
                        if d.yts.to_list()[0] == 1:
                            d = d >> apply(lambda alg, Xtr, ytr: clone(alg).fit(Xtr, ytr)).estimator
                            d = d >> apply(lambda estimator, Xts: estimator.predict(Xts)).prediction
                            if d.prediction.tolist()[0] == 1:
                                d = d >> apply(dx.Explainer, model=_.estimator, data=_.Xtr, y=_.ytr).explainer
                                # d = d >> apply(dx.Explainer.model_parts, _.explainer, processes=_.n_jobs).modelparts
                                # d = ch(d, storages, storage_to_be_updated)
                                # d.modelparts.plot(show=False).show()

                                d = d >> apply(dx.Explainer.predict_parts, _.explainer, new_observation=_.Xts, type="shap", processes=1).predictparts
                                d = ch(d, storages, storage_to_be_updated)
                                # d.predictparts.plot(min_max=[0, 1], show=False).show()
                                # predictparts: VariableImportance = d.predictparts
                                d = d >> apply(aaa).contribs_accumulator
                                d = ch(d, storages, storage_to_be_updated)

                    d = d >> apply(importances2, descr1=_.field, descr2=_.parto).res_importances
                    d = ch(d, storages, storage_to_be_updated)
                    # pprint(d.res_importances)
                    # d = ch(d, storages, storage_to_be_updated)

                print()
    print("Finished!")

    if not sched:
        for m in d.measures:
            df1 = DataFrame(d.res_importances[m])
            df2 = DataFrame(d.res[m])
            df = df1.merge(df2, on="description", how="left")
            df[["field", "delivery_mode", "measure"]] = df["description"].str.split('-', expand=True)
            del df["description"]
            df.sort_values("score", ascending=False, inplace=True)
            print(df)
            df.to_csv(f"/tmp/breastfeed-paper-scores-pvalues-importances-{vif}-{m}.csv")
