import copy
from datetime import datetime
import warnings
from datetime import datetime
from itertools import repeat
from pprint import pprint
from sys import argv

import dalex as dx
from argvsucks import handle_command_line
from pandas import DataFrame
from shelchemy import sopen
from shelchemy.scheduler import Scheduler
from sklearn import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.experimental import enable_iterative_imputer
from sklearn.model_selection import LeaveOneOut, permutation_test_score, StratifiedKFold

from germina.config import local_cache_uri, remote_cache_uri, near_cache_uri, schedule_uri
from germina.dataset import join
from germina.loader import load_from_csv, clean_for_dalex, get_balance, importances2, aaa, start_reses, ccc, bbb
from germina.runner import ch
from hdict import hdict, apply, _

warnings.filterwarnings('ignore')
if __name__ == '__main__':
    pulatudo = True
    pulatudo = False
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
        if pulatudo:
            d = hdict.load("LACpbgrFDL4dqebRTxOMd4R2o.PtioKtBuj87tbF", local_storage)
            print("Loaded!")
        else:
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
                        d["contribs_accumulator"] = d["values_accumulator"] = None
                        for (fi, pa, vi, __, __), i in (Scheduler(db, timeout=60) << tasks) if sched else tasks:
                            d["idxtr", "idxts"] = runs[i]
                            print(f"\t{i}\t{fi}\t{pa}\t{vi}\tts:{d.idxts}\t", datetime.now(), f"\t{100 * i / len(d.X):1.1f} %\t-----------------------------------")

                            d = d >> apply(lambda X, y, idxtr, idxts: (X.iloc[idxtr], y.iloc[idxtr], X.iloc[idxts], y.iloc[idxts]))("Xtr", "ytr", "Xts", "yts")
                            if d.yts.to_list()[0] == 1:
                                d = d >> apply(lambda alg, Xtr, ytr: clone(alg).fit(Xtr, ytr)).estimator  # reminder: don't store 'model'
                                d = d >> apply(lambda estimator, Xts: estimator.predict(Xts)).prediction
                                if d.prediction.tolist()[0] == 1:
                                    d = d >> apply(dx.Explainer, model=_.estimator, data=_.Xtr, y=_.ytr).explainer
                                    d = d >> apply(dx.Explainer.predict_parts, _.explainer, new_observation=_.Xts, type="shap", processes=1).predictparts
                                    d = ch(d, storages, storage_to_be_updated)
                                    d = d >> apply(aaa).contribs_accumulator
                                    d = ch(d, storages, storage_to_be_updated)
                                    d = d >> apply(bbb).values_accumulator
                                    d = ch(d, storages, storage_to_be_updated)

                        d = d >> apply(importances2, descr1=_.field, descr2=_.parto).res_importances
                        d = ch(d, storages, storage_to_be_updated)

                    print()
            d.save(local_storage)
            print("Finished!")

    d.show()
    if not sched:
        resimportances, res = copy.deepcopy(d.res_importances), d.res
        for m in d.measures:
            values_shaps = resimportances[m].pop("values_shaps")
            print(values_shaps)

            df1 = DataFrame(resimportances[m])
            df2 = DataFrame(res[m])
            df2.rename({"p-value": "model_p-value"}, inplace=True)

            df = df1.merge(df2, on="description", how="left")
            df[["field", "delivery_mode", "measure"]] = df["description"].str.split('-', expand=True)
            del df["description"]
            df.sort_values("score", ascending=False, inplace=True)
            print(df)
            df.to_csv(f"/home/davi/git/germina/breastfeed-paper-scores-pvalues-importances-{vif=}-{m}-trees={d.n_estimators}-{d.n_permutations=}-{d.ids['res']}-{d.ids['res_importances']}.csv")

"""
    _id: LACpbgrFDL4dqebRTxOMd4R2o.PtioKtBuj87tbF,
    _ids: {
        n_permutations: Ia59r57N4ZbP2QU6ukhHZyFTetb71a1MWr6wNv2e,
        n_repeats: M7HyZUgSF.ZSmBEMFcDkiZBQz00wU9pGF3DoRiDu,
        imputation_trees: M7HyZUgSF.ZSmBEMFcDkiZBQz00wU9pGF3DoRiDu,
        random_state: M7HyZUgSF.ZSmBEMFcDkiZBQz00wU9pGF3DoRiDu,
        target_var: 61e69319G35iXtmb4coJC7iarKv8nrGTbqe31QF7,
        measures: J4fWpvg0macuoHiXzsP6bHVzVmgmr0xUQZRD0Sb4,
        max_iter: 51.DDUrY5M40-CLBvPlHV-p04HKnsXzpoq7SVEir,
        n_estimators: 51.DDUrY5M40-CLBvPlHV-p04HKnsXzpoq7SVEir,
        n_splits: ecvgo-CBPi7wRWIxNzuo1HgHQCbdvR058xi6zmr2,
        shuffle: oK8X-7eG1Qp1WH7v6fokBDrQPdngKn.h86tlEnx4,
        index: LuboysuDvbHXYTrfFdElle0gArVEPFRxLpNYtqIb,
        join: mIYIXtS2dR60NlpaHeHkHgi-vm7AICh0xwXhs3fq,
        n_jobs: 3MDYtx6Fhsi0FuPsmIRfthQ-xbwUWEiYgQPRHO.7,
        return_name: ZDGiyqxLvVn3e7PvMeLgzGeYm9uuNyeVYe9Owi14,
        osf_filename: fP28-J7G9Rm49hTjJI0cPvbGAnt6J6dxnTsnW962,
        cv: nMEU4Fjfu1XyQs3RB2vid5fju7xWJzM-QfwFmxhg,
        res: zIaAb1W7hrQaNbWZzR6OLHIEE57TsG0-NR2hpWhr,
        res_importances: 5ddyciUdwaUZoilYv172KPuhv2Y3x2rg7AtVYpwi,
        field: 3TN0RrNqX.l283blsfGrpxcLZJhRRW.b-aHY4Wo4,
        pathways34: pErnqiQrE0dpYZYHlEaW.6jX3BpV.B1GwsZ.GRX1,
        ebf: ot3GKqCzu76fAQdug7uw8B1UTh8Ed8F64EHmgMVJ,
        df: hXLKU.tMp.qM7rkqJt.rvFAxD6YwejBhnZMOTeWC,
        X: EMWOHfdfL7SbGbcjIlhuxeC2vjqP8KGdJLqoNDLn,
        Xcols: 5sqz1zEsrWPLSICdrNXzp22XUr6TxewZfq5Xq1G5,
        y: bO3DG7Nraylu4UVT51FLrcDgDChXzS7x9Q8DQoC-,
        X0: YSd53GRdu.GAlMeze5SsQRhgqG.Bq2jIwX1Map9p,
        y0: 4l2gIviWbSxaPPM9xkzJL5Yj.mSGFM-xM4rptUjS,
        parto: sM3AhTsyZsJOEioev8lGED-K2aIvJVuyl7zB7ly9,
        Xshape: cGZusuWM8bKbDkZU-D4Dd3X0TDdU0SkR9CTyXgli,
        yshape: 58gPnEbCJEnxDk0rg.XRoUIxNsy3NenaYPvFoteX,
        unique_labels: MUJwW7LrRN1wbqcNZ-Wd-ttdqBrrYnqVKrasncZ6,
        counts: GEcsjrtrndDjjOXUibEwFm7KrhmFj63n6t3AbRnc,
        proportions: 13FK69Bt5zDUuYsv9fM.vLWOamm18Cm1e7xqHhdx,
        alg: Z4qBSOY42K0bPG1xDTko2wkZ.yMMzNFyu9wj0vSp,
        scoring: AIekppmFFWjb6KcgZXwiZgsBvjWWBgdWWHeSaeKs,
        balanced_accuracy_score: SlMNFEnwZGssN0Wcm3cKgnRnqnH02dFLe.PShb16,
        balanced_accuracy_permscores: 3XSbG4S6.7vIcRMTl9.V8dE347j5R8iOwV9pPL8n,
        balanced_accuracy_pval: 20yHBNNT6wkWDB-2pZMvqUj7CtcM6Pt8822Zfh5P,
        contribs_accumulator: okXi2ZAmtac-92AZOnkWpPOP8LBfaUw1SEr4iZyG,
        idxtr: BSZio5.U1sAyAmPcwAyB744yaBSrStCiF5jH.eW2,
        idxts: IsFH5tADZAyLEPxGMb6atZ-y0yFfwXuYaXeHPwel,
        Xtr: iGrFAXoL44.XyCnZUFrO2euu2R9-zK2G5nBIvOFr,
        ytr: GqkM6rO3lnyLI0zorV9xvx0p3U8hpgpIUpQoFkL1,
        Xts: KlBDhKOp5-Nc0U1G-Yaz-ERZWvyWB2GYKETs9PAc,
        yts: 1PlBKPRVbS0KSEQWBDpkmQYuCzn99l63BJjuhy3Z,
        estimator: JbuS-3KK7J4qNN72-6c7PJQXtOCB.9Gc.jZkzn9N,
        prediction: DXrxhAe7rpyDrW7jgIGkX3sDIImAHEIHXU1N3OA0,
        explainer: wnA.Nfg09IAaTBQPFaFlwGk2y0qe1d567ZFmYgdJ,
        predictparts: zQO90Ypaivk42m78SbH0Cv-JgLTko20kdcaizFJ7,
        species34: OPMDmtzPsZ-udScfGJqjwbk2pplXYFa6jfqeWk9f,
        pathways57: p5uscYiJ-C174-tLczeIyYzbKUYc6OWYF7SGlDvK,
        species57: hb5m9cqlmf1CsOK.pvsnDd2RZLxTC9FAl.NcVfmJ,
        pathways89: BqmRMgw4tNjkMWiW3huLP.7dJ.4mT.x2dbe4ELt1,
        species89: 0HktNaOIyMDI.R9XJyg0SgJfmFNeOsngCW1q6LeI
"""
