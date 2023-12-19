import warnings
from datetime import datetime
from itertools import repeat
from pprint import pprint
from sys import argv

import dalex as dx
import numpy as np
from argvsucks import handle_command_line
from lightgbm import LGBMClassifier as LGBMc
from pandas import DataFrame
from scipy import stats
from shelchemy import sopen
from shelchemy.scheduler import Scheduler
from sklearn import clone
from sklearn.ensemble import ExtraTreesClassifier as ETc, StackingClassifier, VotingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.experimental import enable_iterative_imputer
from sklearn.metrics import average_precision_score, make_scorer
from sklearn.model_selection import LeaveOneOut, permutation_test_score, StratifiedKFold, cross_val_predict
from sklearn.neural_network import MLPClassifier

from germina.config import local_cache_uri, remote_cache_uri, near_cache_uri, schedule_uri
from germina.dataset import join
from germina.loader import load_from_csv, clean_for_dalex, get_balance, start_reses, ccc
from germina.runner import ch
from hdict import hdict, apply, _

warnings.filterwarnings('ignore')
if __name__ == '__main__':
    load = argv[argv.index("load") + 1] if "load" in argv else False
    __ = enable_iterative_imputer
    dct = handle_command_line(argv, pvalruns=int, importanceruns=int, imputertrees=int, seed=int, target=str, trees=int, vif=False, nans=False, sched=False, up="", measures=list, algs=list, loo=False)
    print(datetime.now())
    pprint(dct, sort_dicts=False)
    print()
    path = "data/"
    d = hdict(
        algs=dct["algs"],
        n_permutations=dct["pvalruns"],
        n_repeats=dct["importanceruns"],
        imputation_trees=dct["imputertrees"],
        random_state=dct["seed"],
        target_vars=dct["target"],
        measures=dct["measures"],
        max_iter=dct["trees"], n_estimators=dct["trees"],
        n_splits=5,
        shuffle=True,
        index="id_estudo", join="inner", n_jobs=20, return_name=False, deterministic=True, force_row_wise=True,
        osf_filename="germina-osf-request---davi121023"
    )
    cfg = hdict(d)
    for noncfg in ["index", "join", "n_jobs", "return_name", "osf_filename"]:
        del cfg[noncfg]
    vif, nans, sched, storage_to_be_updated = dct["vif"], dct["nans"], dct["sched"], dct["up"]
    loo_flag = dct["loo"]
    with (sopen(local_cache_uri, ondup="skip") as local_storage, sopen(near_cache_uri, ondup="skip") as near_storage, sopen(remote_cache_uri, ondup="skip") as remote_storage, sopen(schedule_uri) as db):
        storages = {
            "remote": remote_storage,
            "near": near_storage,
            "local": local_storage,
        }
        if load:
            d = hdict.load(load, local_storage)
            print("Loaded!")
        else:
            d = d >> apply(StratifiedKFold).cv
            d["res"] = {}
            d["res_importances"] = {}
            for measure in d.measures:
                d = d >> apply(start_reses, measure=measure)("res", "res_importances")
                d = ch(d, storages, storage_to_be_updated)

            results = {}
            for arq, field, oldidx in [("t1_species_filtered", "species1", "Species"),
                                       ("t2_species_filtered", "species2", "Species")]:
                d["field"] = field
                results[field] = {}

                print(field, "=================================================================================")
                d = load_from_csv(d, storages, storage_to_be_updated, path, vif, arq, field, transpose=True, old_indexname=oldidx, verbose=False)
                d = load_from_csv(d, storages, storage_to_be_updated, path, False, "workshop111223", "osf0", False, verbose=False)
                for target_var in d.target_vars.split(","):
                    d["target_var"] = target_var
                    d["cols"] = ["idade_crianca_dias_t2", "idade_crianca_dias_t3", target_var]
                    d = d >> apply(lambda osf0, target_var, cols: osf0[cols]).osf
                    d = d >> apply(join, df=_.osf, other=_[field]).df
                    d = ch(d, storages, storage_to_be_updated)

                    d = clean_for_dalex(d, storages, storage_to_be_updated, target=d.target_var, alias=d.target_var, keep=d.cols)
                    d = d >> apply(lambda X: X.copy(deep=True)).X0
                    d = d >> apply(lambda y: y.copy(deep=True)).y0
                    d = ch(d, storages, storage_to_be_updated)

                    parto = "none"
                    d["parto"] = parto
                    print(parto)
                    results[field][parto] = {}
                    d["X"] = d.X0
                    d = ch(d, storages, storage_to_be_updated)
                    d = get_balance(d, storages, storage_to_be_updated)
                    d[f"X_{field}_{parto}"] = _.X
                    d[f"y_{field}_{parto}"] = _.y
                    print(f"X_{field}_{parto}", f"y_{field}_{parto}")

                    params = {"max_depth": 5, "objective": "binary:logistic", "eval_metric": "auc"}
                    loo = LeaveOneOut()
                    runs = list(loo.split(d.X))
                    algs = {"RFc": RandomForestClassifier, "LGBMc": LGBMc, "ETc": ETc, "Sc": StackingClassifier, "MVc": VotingClassifier, "hardMVc": VotingClassifier}
                    algs = {k: algs[k] for k in d.algs}
                    for alg_name, alg in algs.items():
                        print(alg_name, "<<<<<<<<<<<<<<<<<")
                        if alg_name == "Sc":
                            d["estimators"] = []
                            for na, al in list(algs.items())[:-3]:
                                d.apply(al, out=f"base_alg")
                                d["base_name"] = na
                                d.apply(lambda base_name, base_alg, estimators: estimators + [(base_name, base_alg)], out="estimators")
                            d = d >> apply(alg, final_estimator=MLPClassifier(random_state=0, max_iter=30, hidden_layer_sizes=(20,))).alg
                        elif alg_name == "MVc":
                            d["estimators"] = []
                            for na, al in list(algs.items())[:-3]:
                                d.apply(al, out=f"base_alg")
                                d["base_name"] = na
                                d.apply(lambda base_name, base_alg, estimators: estimators + [(base_name, base_alg)], out="estimators")
                            d = d >> apply(alg, voting="soft").alg
                        elif alg_name == "hardMVc":
                            d["estimators"] = []
                            for na, al in list(algs.items())[:-3]:
                                d.apply(al, out=f"base_alg")
                                d["base_name"] = na
                                d.apply(lambda base_name, base_alg, estimators: estimators + [(base_name, base_alg)], out="estimators")
                            d = d >> apply(alg, voting="hard").alg
                        else:
                            d = d >> apply(alg).alg

                        for m in d.measures:
                            results[field][parto][m] = {}
                            # calcula baseline score e p-values
                            if m == "average_precision_score":
                                if alg_name == "hardMVc":
                                    continue
                                d["scoring"] = make_scorer(average_precision_score, needs_proba=True)
                            else:
                                d["scoring"] = m
                            prefix = f"{field}_{parto}_{alg_name}_{m}"
                            score_field, permscores_field, pval_field = f"{prefix}_score", f"{prefix}_permscores", f"{prefix}_pval"
                            predictions_field = f"{field}_{parto}_{alg_name}_predictions"

                            tasks = [(field, parto, f"{vif=}", m, f"trees={d.n_estimators}_{alg_name}_{target_var}")]
                            print(f"Starting {field}_{parto}_{m}  ...", d.id)
                            for __, __, __, __, __ in (Scheduler(db, timeout=60) << tasks) if sched else tasks:
                                d = d >> apply(cross_val_predict, _.alg)(predictions_field)
                                d = ch(d, storages, storage_to_be_updated)
                                d = d >> apply(permutation_test_score, _.alg)(score_field, permscores_field, pval_field)
                                d = ch(d, storages, storage_to_be_updated)
                                d = d >> apply(ccc, d_score=_[score_field], d_pval=_[pval_field]).res
                                d = ch(d, storages, storage_to_be_updated)
                                print(f"score (p-value):\t{d.res[m]['score'][-1]:.4f} ({d.res[m]['p-value'][-1]:.4f})\t{d.res[m]['description'][-1]}={alg_name}", flush=True)

                            d = d >> apply(lambda res: res).res
                            d = ch(d, storages, storage_to_be_updated)

                            # LOO shaps @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
                            if not loo_flag:
                                continue
                            tasks = zip(repeat((field, parto, f"{vif=}", m, f"trees={d.n_estimators}_{alg_name}_{target_var}")), range(len(runs)))
                            d["contribs_accumulator"] = d["values_accumulator"] = None
                            print()
                            for (fi, pa, vi, __, __), i in (Scheduler(db, timeout=60) << tasks) if sched else tasks:
                                d["idxtr", "idxts"] = runs[i]
                                print(f"\r>>> {fi}\t{pa}\t{vi} ts:{d.idxts}\t{str(datetime.now()):19}\t{100 * i / len(d.X):1.1f} %", end="")

                                d = d >> apply(lambda X, y, idxtr, idxts: (X.iloc[idxtr], y.iloc[idxtr], X.iloc[idxts], y.iloc[idxts]))("Xtr", "ytr", "Xts", "yts")

                                d = d >> apply(lambda alg, Xtr, ytr: clone(alg).fit(Xtr, ytr)).estimator  # reminder: we won't store hundreds of models; skipping ch() call
                                d = d >> apply(lambda estimator, Xts: estimator.predict(Xts)).prediction

                                # if d.yts.to_list()[0] == d.prediction.tolist()[0] == 1:
                                d = d >> apply(dx.Explainer, model=_.estimator, data=_.Xtr, y=_.ytr).explainer
                                d = d >> apply(dx.Explainer.predict_parts, _.explainer, new_observation=_.Xts, type="shap", processes=1).predictparts
                                d = ch(d, storages, storage_to_be_updated)

                                var_valu_dct = {name_val.split(" = ")[0]: float(name_val.split(" = ")[1:][0]) for name_val in d.predictparts.result["variable"]}
                                var_shap_dct = dict(zip((k.split(" ")[0] for k in d.predictparts.result["variable"]), d.predictparts.result["contribution"]))
                                label = d.yts.to_list()[0]
                                prediction = d.prediction.tolist()[0]
                                results[field][parto][m][i] = {"target": label, "prediction": prediction, "var_shap_dct": var_shap_dct, "var_valu_dct": var_valu_dct}

                                # d = d >> apply(aaa).contribs_accumulator
                                # d = ch(d, storages, storage_to_be_updated)
                                # d = d >> apply(bbb).values_accumulator
                                # d = ch(d, storages, storage_to_be_updated)

                            # d = d >> apply(importances2, descr1=_.field, descr2=_.parto).res_importances
                            # # for storage in storages.values():
                            # #     del storage[d.ids["res_importances"]]
                            # d = ch(d, storages, storage_to_be_updated)

                    print()
                    d["results"] = results
                    for storage in storages.values():
                        d.save(storage)
                    print(f"{field}_{parto} finished!", d.id)
            d["results"] = results
            for storage in storages.values():
                d.save(storage)
            print(d.id)
            print("Finished!")

    # d.evaluated.show()
    if sched or not load:
        exit()

    lstfield, lstparto, lstm, lstbebe = [], [], [], []
    targets, predictions = [], []
    vars, shaps, valus = [], [], []
    for field, fielddct in d.results.items():
        for parto, partodct in fielddct.items():
            for m, mdct in partodct.items():
                for bebe, bebedct in mdct.items():
                    target, prediction, var_shap_dct, var_valu_dct = bebedct.values()
                    if var_shap_dct.keys() != var_valu_dct.keys():
                        raise Exception(f"")
                    for (var, shap), valu in zip(var_shap_dct.items(), var_valu_dct.values()):
                        var = var.replace(",", "-").replace(";", "-")
                        lstfield.append(field)
                        lstparto.append(parto)
                        lstm.append(m)
                        lstbebe.append(bebe)
                        targets.append(target)
                        predictions.append(prediction)

                        vars.append(var)
                        shaps.append(shap)
                        valus.append(valu)

    dfbig = DataFrame({"field": lstfield, "parto": lstparto, "score": lstm, "bebe": lstbebe, "target": targets, "prediction": predictions, "variable": vars, "value": valus, "SHAP": shaps})
    dfbig["description"] = dfbig["field"] + "-" + dfbig["parto"] + "-" + dfbig["score"]
    del dfbig["field"]
    del dfbig["parto"]
    del dfbig["score"]
    print(dfbig)
    print(dfbig.columns)
    for m in d.measures:
        print()
        print("============================-------------------")
        print(" ", m)
        print("============================-------------------")
        print()
        dfmodel = DataFrame(d.res[m])
        dfmodel.rename(columns={"p-value": "model_p-value"}, inplace=True)
        dfmodel[["type-age", "delivery_mode", "measure"]] = dfmodel["description"].str.split('-', expand=True)
        dfmodel["age"] = dfmodel["type-age"].str.slice(-2)
        dfmodel["type"] = dfmodel["type-age"].str.slice(0, -2)
        del dfmodel["type-age"]
        dfmodel.sort_values("score", ascending=False, inplace=True)
        dfmodel.rename(columns={"score": m}, inplace=True)
        dfmodel.to_csv(f"/home/davi/git/germina/results/model-performance-{m}-trees={d.n_estimators}-perms={d.n_permutations}-{d.id}--LOO.csv")
        print(dfmodel)
        print(dfmodel.columns)
        print("__________________________________")
        print()

        df = dfbig.merge(dfmodel, on="description", how="left")
        for descr in df["description"].unique():
            subdf = df[df["description"] == descr]
            del subdf["type"]
            del subdf["age"]
            del subdf["measure"]
            del subdf[m]
            del subdf["description"]
            del subdf["delivery_mode"]
            subdf.to_csv(f"/home/davi/git/germina/results/complete---{descr}---{m}-trees={d.n_estimators}-perms={d.n_permutations}-{d.id}--LOO.csv")
        print("++++++++++++++++++++++++++++++++++++++++++++++")
        print()

        df["helpfulness"] = np.where(df["target"] == 1, df["SHAP"], -df["SHAP"])
        grouped = df.groupby(["delivery_mode", "age", "type", "measure", "variable"]).agg({
            "helpfulness": [lambda x: np.mean(x), lambda x: np.std(x), lambda x: stats.ttest_1samp(x, popmean=0, alternative="greater")[1]],
            "SHAP": ["min", "max"],
            'value': ['mean', 'std'],
            "model_p-value": ["first"],
        })
        print(grouped)
        print(grouped.columns)
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        grouped.to_csv(f"/home/davi/git/germina/results/grouped--SHAPhelpfulness-p-value-{m}-trees={d.n_estimators}-perms={d.n_permutations}-{d.id}--LOO.csv")
