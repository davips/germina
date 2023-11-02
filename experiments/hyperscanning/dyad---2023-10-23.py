import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
from datetime import datetime
from pprint import pprint
from sys import argv

from argvsucks import handle_command_line
from catboost import CatBoostRegressor as CatBr
from catboost import CatBoostClassifier as CatBc
from hdict import hdict, apply, _
from lightgbm import LGBMClassifier as LGBMc, LGBMRegressor as LGBMr
from pandas import DataFrame
from shelchemy import sopen
from shelchemy.scheduler import Scheduler
from sklearn import clone
from sklearn.ensemble import ExtraTreesClassifier as ETc
from sklearn.ensemble import ExtraTreesRegressor as ETr
from sklearn.ensemble import HistGradientBoostingClassifier as HGBc
from sklearn.ensemble import HistGradientBoostingRegressor as HGBr
from sklearn.ensemble import RandomForestClassifier as RFc
from sklearn.ensemble import RandomForestRegressor as RFr
from sklearn.experimental import enable_iterative_imputer
from sklearn.inspection import permutation_importance
from sklearn.model_selection import permutation_test_score, StratifiedKFold, KFold
from sklearn.tree import DecisionTreeRegressor as DTr
from sklearn.tree import DecisionTreeClassifier as DTc
from xgboost import XGBClassifier as XGBc, XGBRegressor as XGBr

from germina.config import local_cache_uri, remote_cache_uri, near_cache_uri, schedule_uri
from germina.dataset import eeg_t2_vars, join
from germina.loader import load_from_csv, get_balance, importances, load_from_synapse, impute
from germina.runner import ch

__ = enable_iterative_imputer
dct = handle_command_line(argv, pvalruns=int, importanceruns=int, imputertrees=int, seed=int, target=str, trees=int, vif=False, nans=False, sched=False, up="", measures=list, targets=list, swap=False)
print(datetime.now())
pprint(dct, sort_dicts=False)
print()
path = "data/"
d = hdict(
    n_permutations=dct["pvalruns"],
    n_repeats=dct["importanceruns"],
    imputation_trees=dct["imputertrees"],
    random_state=dct["seed"],
    target=dct["target"],
    measures=dct["measures"], targets=dct["targets"],
    max_iter=dct["trees"], n_estimators=dct["trees"],
    n_splits=5,
    shuffle=True,
    index="id_estudo", join="inner", n_jobs=-1, return_name=False,
    osf_filename="germina-osf-request---davi121023",
    verbose=False, swap=dct["swap"]
)
cfg = hdict(d)
for noncfg in ["index", "join", "n_jobs", "return_name", "osf_filename", "verbose", "swap"]:
    del cfg[noncfg]
vif, nans, sched, storage_to_be_updated = dct["vif"], dct["nans"], dct["sched"], dct["up"]

res = {}
res_importances = {}
for measure in d.measures:
    res[measure] = {"description": [], "score": [], "p-value": []}
    res_importances[measure] = {"description": [], "variable": [], "importance-mean": [], "importance-stdev": []}
d["res_importances"] = res_importances

with (sopen(local_cache_uri) as local_storage, sopen(near_cache_uri) as near_storage, sopen(remote_cache_uri) as remote_storage, sopen(schedule_uri) as db):
    storages = {
        "remote": remote_storage,
        "near": near_storage,
        "local": local_storage,
    }

    d = d >> apply(StratifiedKFold).cv
    res = {}
    res_importances = {}
    for measure in d.measures:
        res[measure] = {"description": [], "score": [], "p-value": []}
        res_importances[measure] = {"description": [], "variable": [], "importance-mean": [], "importance-stdev": []}
    d["res_importances"] = res_importances

    d = load_from_synapse(d, storages, storage_to_be_updated, path, vif, "synapse/EEG-september-nosensorvars-nomother-nobaby", "Xdyadic")
    # d = load_from_synapse(d, storages, storage_to_be_updated, path, vif, "synapse/EEG-september-nosensorvars", "Xdyadic")
    d = load_from_synapse(d, storages, storage_to_be_updated, path, vif, "timedelta", "Xtime")
    d = d >> apply(lambda Xdyadic: Xdyadic.dropna()).Xdyadic
    d = d >> apply(join, df=_.Xdyadic, other=_.Xtime).Xdyadic_time
    print(f"Joined timedelta with dyadic  ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓ {d.Xdyadic_time.shape} ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑")

    d = d >> apply(lambda Xdyadic_time: Xdyadic_time.dropna()).Xdyadic_time
    d = ch(d, storages, storage_to_be_updated)
    print(f"Removed NaNs  ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓ {d.Xdyadic_time.shape} ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑")

    d = load_from_csv(d, storages, storage_to_be_updated, path, vif, d.osf_filename, "single", transpose=False, vars=eeg_t2_vars + ["risco_class"])
    d = ch(d, storages, storage_to_be_updated)

    print("Separate subset from dataset 'EEG single'  --------------------------------------------------------------------------------------------------------------------------------------------------------")
    d = d >> apply(lambda single, Xdyadic_time: single.loc[Xdyadic_time.index]).single_small
    d = d >> apply(lambda single, Xdyadic_time: single.loc[~single.index.isin(Xdyadic_time.index)]).single_large
    d = ch(d, storages, storage_to_be_updated)

    print(datetime.now(), f"Model imputation {d.n_estimators=} {d.imputation_trees=}--------------------------------------------------------------------------------------------------------------------------------------------------------")
    d = d >> apply(RFr, n_estimators=_.imputation_trees).imputation_alg
    d = d >> apply(impute).Xsingle
    d = ch(d, storages, storage_to_be_updated)
    print(f"X {d.Xsingle.shape} ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓\n", d.Xsingle, "↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑\n")

    d["risco_class"] = d.Xsingle[["risco_class"]]
    d = d >> apply(join, df=_.Xdyadic_time, other=_.risco_class).Xdyadic_time_risk
    d = ch(d, storages, storage_to_be_updated)
    print(f"Joined dyadic_time with risco_class  ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓ {d.Xdyadic_time_risk.shape} ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑")

    tasks = [(cfg.hosh * tgt.encode(), f"{vif=}", tgt) for tgt in d.targets]
    for h, vi, target in (Scheduler(db, timeout=50) << tasks) if False and sched else tasks:
        # if not sched:
        #     print(f"\t{h.ansi}\t{vi}\t{target}\t", datetime.now(), f"\t-----------------------------------")

        d = load_from_csv(d, storages, storage_to_be_updated, path, vif, d.osf_filename, "y", transpose=False, vars=[target], verbose=False)
        d = d >> apply(lambda y, Xdyadic_time_risk: y.loc[Xdyadic_time_risk.index].dropna()).y
        if "r2" not in d.measures:
            d = d >> apply(lambda y: (y > y.median()).astype(int)).y
            d = ch(d, storages, storage_to_be_updated)
            constructors = {"RFc": RFc, "DTc": DTc, "HGBc": HGBc, "ETc": ETc, "LGBMc": LGBMc, "XGBc": XGBc, "CatBc": CatBc}
        else:
            constructors = {"RFr": RFr, "DTr": DTr, "HGBr": HGBr, "ETr": ETr, "LGBMr": LGBMr, "XGBr": XGBr, "CatBr": CatBr}

        tasks = [(cfg.hosh * cons.encode() * target.encode(), f"{vif=}", target, cons) for cons in constructors]
        for h2, vi2, __, k in (Scheduler(db, timeout=50) << tasks) if sched else tasks:
            d["alg_name"] = k
            for Xvar in (["Xdyadic_time_risk", "Xsingle"] if d.swap else ["Xsingle", "Xdyadic_time_risk"]):
                if not sched:
                    print(f"\t{h2.ansi}\t{vi2}\t{target}\t{Xvar}\t{k}\t", datetime.now(), f"\t-----------------------------------")

                d = d >> apply(lambda X, y: X.loc[X.index.isin(y.index)], _[Xvar]).X
                d = d >> apply(KFold).cv

                if "r2" not in d.measures:
                    d = get_balance(d, storages, storage_to_be_updated)
                    d = ch(d, storages, storage_to_be_updated)

                d = d >> apply(constructors[k]).alg
                d = ch(d, storages, storage_to_be_updated)

                for m in d.measures:
                    d["scoring"] = m
                    rets = [f"{m}_scores", f"{m}_permscores", f"{m}_pval"]
                    d = d >> apply(permutation_test_score, _.alg)(*rets)
                    d = ch(d, storages, storage_to_be_updated)
                    res[m]["description"].append(f"{target}-{Xvar}-{m}-{k}")
                    res[m]["score"].append(d[rets[0]])
                    res[m]["p-value"].append(d[rets[2]])
                    print(f"{m:20} (p-value):\t{d[rets[0]]:.4f} ({d[rets[2]]:.4f})")

                    # Importances
                    d = d >> apply(lambda alg, X, y: clone(alg).fit(X, y)).estimator
                    d = d >> apply(permutation_importance).importances
                    d = ch(d, storages, storage_to_be_updated)
                    d = d >> apply(importances, descr1=target, descr2=Xvar).res_importances
                    d = ch(d, storages, storage_to_be_updated)

                    print()

    print("All finished")

if not sched:
    for m in d.measures:
        df = DataFrame(res[m])
        df[["target", "eeg_type", "measure", "algorithm"]] = df["description"].str.split('-', expand=True)
        del df["description"]
        df.sort_values("p-value", inplace=True)
        print(df)
        df.to_csv(f"/tmp/dyad-paper--{cfg.id}--scores-pvalues-{vif}-{m}.csv")

        df = DataFrame(d.res_importances[m])
        df[["target", "eeg_type", "measure", "algorithm"]] = df["description"].str.split('-', expand=True)
        del df["description"]
        df.sort_values("importance-mean", ascending=False, inplace=True)
        print(df)
        df.to_csv(f"/tmp/dyad-paper--{cfg.id}--importances-{vif}-{m}.csv")
