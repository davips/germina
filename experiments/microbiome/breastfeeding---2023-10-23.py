import warnings
from datetime import datetime
from itertools import repeat

from pandas import DataFrame
from shelchemy.scheduler import Scheduler
from sklearn import clone
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
from germina.loader import load_from_csv, clean_for_dalex, get_balance, train_xgb, build_explainer, explain_modelparts, explain_predictparts, importances

from sklearn.model_selection import LeaveOneOut, permutation_test_score, StratifiedKFold

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
    index="id_estudo", join="inner", n_jobs=-1, return_name=False,
    osf_filename="germina-osf-request---davi121023"
)
cfg = hdict(d)
for noncfg in ["index", "join", "n_jobs", "return_name", "osf_filename"]:
    del cfg[noncfg]
vif, nans, sched, storage_to_be_updated = dct["vif"], dct["nans"], dct["sched"], dct["up"]
with (sopen(local_cache_uri) as local_storage, sopen(near_cache_uri) as near_storage, sopen(remote_cache_uri) as remote_storage, sopen(schedule_uri) as db):
    storages = {
        # "remote": remote_storage,
        "near": near_storage,
        "local": local_storage,
    }

    d = d >> apply(StratifiedKFold).cv
    res = {}
    res_importances = {}
    for measure in d.measures:
        res[measure] = {"description": [], "score": [], "p-value": []}
        res_importances[measure] = {"description": [], "variable": [], "importance_mean": [], "importance_std": []}
    d["res_importances"] = res_importances

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

            # # LOO
            # tasks = zip(repeat((field, parto, f"{vif=}")), range(len(d.X)))
            # for (fi, pa, vi), i in (Scheduler(db, timeout=60) << tasks) if sched else tasks:
            #     idxtr, idxts = runs[i]
            #     print(f"\t{i}\t{fi}\t{pa}\t{vi}\tts:{idxts}\t", datetime.now(), f"\t{100 * i / len(d.X):1.1f} %\t-----------------------------------")
            #     d = d >> apply(train_xgb, params, idxtr=idxtr).classifier
            #     d = ch(d, storages, storage_to_be_updated)
            #
            #     d = d >> apply(build_explainer, idxtr=idxtr).explainer
            #     d = ch(d, storages, storage_to_be_updated)
            #
            #     d = d >> apply(explain_modelparts).modelparts
            #     d = ch(d, storages, storage_to_be_updated)
            #
            #     d = d >> apply(explain_predictparts, idxts=idxts).predictparts
            #     d = ch(d, storages, storage_to_be_updated)
            #
            #     # from dalex.model_explanations import VariableImportance
            #     # modelparts: VariableImportance = d.modelparts
            #     # pprint(modelparts.result[["variable", "contribution"]].to_dict())
            #
            #     # from dalex.model_explanations import VariableImportance
            #     # predictparts: VariableImportance = d.predictparts
            #     # varcontrib = dict(list(sorted(zip(predictparts.result["contribution"], predictparts.result["variable"]), key=lambda x: x[0]))[:5])
            #     # pprint(varcontrib)
            #
            #     # d.modelparts.plot(show=False).show()
            #     # d.predictparts.plot(min_max=[0, 1], show=False).show()
            #     # exit()

            # Entire dataset
            cfg["field"], cfg["parto"] = field, parto
            tasks = [(cfg.hosh, field, parto, f"{vif=}")]
            idxtr = range(len(d.X))
            for h, fi, pa, vi in (Scheduler(db, timeout=50) << tasks) if sched else tasks:
                if not sched:
                    print(f"\t{h.ansi}\t{fi}\t{pa}\t{vi}\t", datetime.now(), f"\t-----------------------------------")

                # # todo: esse xgb não tem o nr de trees ajustado
                # d = d >> apply(train_xgb, params, idxtr=idxtr).classifier
                # d = ch(d, storages, storage_to_be_updated)
                #
                # d = d >> apply(build_explainer, idxtr=idxtr).explainer
                # d = ch(d, storages, storage_to_be_updated)
                #
                # d = d >> apply(explain_modelparts).modelparts
                # d = ch(d, storages, storage_to_be_updated)

                #  CSV
                # from dalex.model_explanations import VariableImportance
                # modelparts: VariableImportance = d.modelparts
                # # vardroploss = dict(list(sorted(zip(modelparts.result["dropout_loss"], modelparts.result["variable"]), key=lambda x: x[0])))
                # df = modelparts.result[["dropout_loss", "variable"]].sort_values("dropout_loss", ascending=False)
                # tgtarq = f"/tmp/breastfeed-paper-{fi}-{pa}-{vi.split('=')[1]}.csv"
                # print(tgtarq)
                # df.to_csv(tgtarq)
                # pprint(df)

                # d.modelparts.plot(show=False).show()
                # exit()

                d = d >> apply(XGBClassifier).alg
                for m in d.measures:
                    d["scoring"] = m
                    rets = [f"{m}_scores", f"{m}_permscores", f"{m}_pval"]
                    d = d >> apply(permutation_test_score, _.alg)(*rets)
                    d = ch(d, storages, storage_to_be_updated)
                    res[m]["description"].append(f"{field}-{parto}-{m}")
                    res[m]["score"].append(d[rets[0]])
                    res[m]["p-value"].append(d[rets[2]])
                    print(f"{m:20} (p-value):\t{d[rets[0]]:.4f} ({d[rets[2]]:.4f})", flush=True)

                    # Importances
                    d = d >> apply(lambda alg, X, y: clone(alg).fit(X, y)).estimator
                    d = d >> apply(permutation_importance).importances
                    d = ch(d, storages, storage_to_be_updated)
                    d = d >> apply(importances, descr1=_.field, descr2=_.target_var).res_importances

            print()
print("Finished!")

if not sched:
    for m in d.measures:
        df = DataFrame(res[m])
        df[["field", "delivery_mode", "measure"]] = df["description"].str.split('-', expand=True)
        del df["description"]
        df.sort_values("p-value", inplace=True, kind="stable")
        print(df)
        df.to_csv(f"/tmp/breastfeed-paper-scores-pvalues-{vif}-{m}.csv")

        df = DataFrame(d.res_importances[m])
        df[["field", "delivery_mode", "measure"]] = df["description"].str.split('-', expand=True)
        del df["description"]
        df.sort_values("importance_mean", ascending=False, inplace=True, kind="stable")
        print(df)
        df.to_csv(f"/tmp/breastfeed-paper-importances-{vif}-{m}.csv")
