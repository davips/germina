import copy
from functools import partial
from itertools import repeat
from pprint import pprint
from sys import argv

from sklearn.ensemble import ExtraTreesClassifier as ETc, ExtraTreesRegressor as ETr
import numpy as np
import pandas as pd
from argvsucks import handle_command_line
from lange import ap
from pandas import read_csv, DataFrame
from scipy.stats import ttest_1samp
from shelchemy import sopen
from shelchemy.scheduler import Scheduler
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier

from germina.config import local_cache_uri, remote_cache_uri, near_cache_uri, schedule_uri
from germina.dataset import eeg_t2_vars
from germina.pairwise import pairwise_diff, pairwise_hstack
from germina.runner import ch, sgid2estudoid
from hdict import hdict
from sklearn.ensemble import RandomForestClassifier as RFc, RandomForestRegressor as RFr
from lightgbm import LGBMClassifier as LGBMc, LGBMRegressor as LGBMr
from xgboost import XGBClassifier as XGBc, XGBRegressor as XGBr

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer


def interpolate_for_classification(targets, conditions):
    """
    :param targets:
        sorted
    :param conditions:
        `1` means the resulting value should be greater than the corresponding target.
        `0` means the resulting value should be equal than the corresponding target. (`0` is not usually needed)
        `-1` means the resulting value should be lesser than the corresponding target.
    :return:

    # >>> tgts = np.array([77,88,81,84,88,90,95,100,103,105,110,112,115,120])
    # >>> conds = np.array([1,1,-1,1,1,-1,1,-1,-1,1,-1,-1,1,-1])
    # >>> interpolate(tgts, conds)
    93.25
    """
    first = 2 * targets[0] - targets[1]
    last = 2 * targets[-1] - targets[-2]
    targets = np.hstack([np.array([first]), targets, np.array([last])])
    conditions = np.hstack([np.array([1]), conditions, np.array([-1])])
    acc = np.cumsum(conditions)
    mx_mask = acc == np.max(acc)
    mx_idxs = np.flatnonzero(mx_mask)
    neighbors_before = targets[mx_idxs]
    neighbors_after = targets[mx_idxs + 1]
    candidates = (neighbors_before + neighbors_after) / 2
    return np.mean(candidates)


def interpolate_for_regression(targets, conditions):
    candidates = targets + conditions
    return np.mean(candidates)


def substep(df, idx, trees, delta, diff, handle_last_as_y, algname):
    algname = "argment passed just to mark which is the algorithm"
    filter = lambda tmp: (tmp[:, -1] < -delta) | (tmp[:, -1] >= delta)
    if diff:
        hstack = lambda a, b: pairwise_diff(a, b, pct=handle_last_as_y == "%")
    else:
        hstack = lambda a, b: pairwise_hstack(a, b, handle_last_as_y=handle_last_as_y)

    # current baby
    baby = df.loc[[idx], :].to_numpy()
    baby_x = baby[:, :-1]
    baby_y = baby[0, -1]

    # training set
    # Xy_tr = df.to_numpy()
    Xy_tr = df.drop(idx, axis="rows").to_numpy()
    tmp = hstack(Xy_tr, Xy_tr)
    pairs_Xy_tr = tmp[filter(tmp)]
    pairs_X_tr = pairs_Xy_tr[:, :-1]

    pairs_y_tr_c = (pairs_Xy_tr[:, -1] >= 0).astype(int)
    pairs_y_tr_r = pairs_Xy_tr[:, -1]

    alg_c = algclass_c()
    alg_r = algclass_r()
    alg_c.fit(pairs_X_tr, pairs_y_tr_c)
    alg_r.fit(pairs_X_tr, pairs_y_tr_r)

    # test set
    tmp = hstack(baby, Xy_tr)
    fltr = filter(tmp)
    pairs_Xy_ts = tmp[fltr]
    pairs_X_ts = pairs_Xy_ts[:, :-1]

    pairs_y_ts_c = (pairs_Xy_ts[:, -1] >= 0).astype(int)
    pairs_y_ts_r = pairs_Xy_ts[:, -1]

    # predictions
    pairs_z_ts_c = alg_c.predict(pairs_X_ts)
    pairs_z_ts_r = alg_r.predict(pairs_X_ts)

    # interpolation
    targets = Xy_tr[fltr, -1]
    baby_z_c = interpolate_for_classification(targets, conditions=2 * pairs_z_ts_c - 1)
    baby_z_r = interpolate_for_regression(targets, conditions=pairs_z_ts_r)

    return baby_y, baby_z_c, baby_z_r


def step(d, db, storages, sched):
    y = d.df[d.target_variable].to_numpy()
    hits_c, hits_r = {0: 0, 1: 0}, {0: 0, 1: 0}
    tot = {0: 0, 1: 0}
    z_lst_c, z_lst_r = [], []
    tasks = zip(repeat(d.id), d.df.index)
    ansi = d.hosh.ansi
    for c, (id, idx) in enumerate((Scheduler(db, timeout=60) << tasks) if sched else tasks):
        if not sched:
            print(f"\r Permutation: {d.i:8}\t\t{ansi} baby {idx}: {100 * c / len(d.df):8.5f}%", end="", flush=True)
        d.apply(substep, idx=idx, out="substep")
        d = ch(d, storages)
        if sched:
            continue

        baby_y, baby_z_c, baby_z_r = d.substep
        z_lst_c.append(baby_z_c)
        z_lst_r.append(baby_z_r)

        expected = int(baby_y >= 100)
        predicted_c = int(baby_z_c >= 100)
        predicted_r = int(baby_z_r >= 100)

        hits_c[expected] += int(expected == predicted_c)
        hits_r[expected] += int(expected == predicted_r)
        tot[expected] += 1

    if sched:
        return
    z_c = np.array(z_lst_c)
    z_r = np.array(z_lst_r)

    acc0 = hits_c[0] / tot[0]
    acc1 = hits_c[1] / tot[1]
    bacc_c = (acc0 + acc1) / 2

    acc0 = hits_r[0] / tot[0]
    acc1 = hits_r[1] / tot[1]
    bacc_r = (acc0 + acc1) / 2

    r2_c = r2_score(y, z_c)
    r2_r = r2_score(y, z_r)
    return bacc_c, bacc_r, r2_c, r2_r, hits_c, hits_r, tot


dct = handle_command_line(argv, delta=float, trees=int, pct=False, diff=False, demo=False, sched=False, perms=1, targetvar=str, jobs=int, alg=str, seed=0, prefix=str, sufix=str)
pprint(dct)
if "geneeg" in argv:
    extra = ["idade_crianca_dias_t2", dct["targetvar"]]
    # Single
    dfwor = read_csv("data/workshop111223.csv")
    dfwor.set_index("id_estudo", inplace=True)
    dfsin = dfwor[eeg_t2_vars + extra]
    dfsin = dfsin.dropna(axis="rows", how="any")
    dfsin.to_csv(f"/home/davi/git/germina/results/single_or_dyadic_is1_{dct['targetvar']}.csv")

    # Dyadic
    dfsyn = read_csv("data/eeg-synapse-reduced.csv")
    dfsyn = sgid2estudoid(dfsyn)
    dfdya: DataFrame = dfsyn.join(dfwor[extra], how="inner")
    # dfdya.drop([458,427,455,501], axis="rows", inplace=True)
    dfdya.drop([458, 501, 455, 427], axis="rows", inplace=True)
    idx = dfdya.count(axis="rows").sort_values() > 47  # Accept 10% of babies with NaN for a single variable
    dfdya = dfdya.loc[:, idx]
    # print(dfdya.count(axis="columns").sort_values())
    # print(dfdya.count(axis="rows").sort_values().tolist())
    # dfdya.fillna()
    dfdya.to_csv(f"/home/davi/git/germina/results/single_or_dyadic_is2_{dct['targetvar']}.csv")
    exit()

trees, delta, pct, diff, demo, sched, perms, targetvar, jobs, alg, seed, prefix, sufix = dct["trees"], dct["delta"], dct["pct"], dct["diff"], dct["demo"], dct["sched"], dct["perms"], dct["targetvar"], dct["jobs"], dct["alg"], dct["seed"], dct["prefix"], dct["sufix"]

rnd = np.random.default_rng(0)
handle_last_as_y = "%" if pct else True
if alg == "rf":
    algclass_c = partial(RFc, n_estimators=trees, random_state=seed, n_jobs=jobs)
    algclass_r = partial(RFr, n_estimators=trees, random_state=seed, n_jobs=jobs)
elif alg == "lgbm":
    algclass_c = partial(LGBMc, n_estimators=trees, random_state=seed, n_jobs=jobs)
    algclass_r = partial(LGBMr, n_estimators=trees, random_state=seed, n_jobs=jobs)
elif alg == "et":
    algclass_c = partial(ETc, n_estimators=trees, random_state=seed, n_jobs=jobs)
    algclass_r = partial(ETr, n_estimators=trees, random_state=seed, n_jobs=jobs)
elif alg == "xg":
    algclass_c = partial(XGBc, n_estimators=trees, random_state=seed, n_jobs=jobs)
    algclass_r = partial(XGBr, n_estimators=trees, random_state=seed, n_jobs=jobs)
elif alg == "cart":
    algclass_c = partial(DecisionTreeClassifier, random_state=seed, n_jobs=jobs)
    algclass_r = partial(DecisionTreeRegressor, random_state=seed, n_jobs=jobs)
elif alg.endswith("nn"):
    k = int(alg[:-2])
    algclass_c = partial(KNeighborsClassifier, n_neighbors=k, n_jobs=jobs)
    algclass_r = partial(KNeighborsRegressor, n_neighbors=k, n_jobs=jobs)
elif alg.endswith("NN"):
    k = int(alg[:-2])
    algclass_c = partial(Pipeline, steps=[('scaler', StandardScaler()), ('svc', KNeighborsClassifier(n_neighbors=k, n_jobs=jobs))])
    algclass_r = partial(Pipeline, steps=[('scaler', StandardScaler()), ('svc', KNeighborsRegressor(n_neighbors=k, n_jobs=jobs))])
else:
    raise Exception(f"Unknown algorithm {alg}. Options: rf,lgbm")

with (sopen(local_cache_uri, ondup="skip") as local_storage, sopen(near_cache_uri, ondup="skip") as near_storage, sopen(remote_cache_uri, ondup="skip") as remote_storage, sopen(schedule_uri) as db):
    storages = {
        # "remote": remote_storage,
        "near": near_storage,
        "local": local_storage,
    }
    for sp in [1, 2]:
        print(f"{sp=} <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
        df = read_csv(f"{prefix}{sp}{sufix}", index_col="id_estudo")
        df.sort_values(targetvar, inplace=True, ascending=True, kind="stable")
        if demo:
            df = pd.concat([df.iloc[:30], df.iloc[-30:]], axis="rows")
        age = df["idade_crianca_dias_t2"]

        d = hdict(sp=sp, df=df, handle_last_as_y=handle_last_as_y, trees=trees, target_variable=targetvar, delta=delta, diff=diff, i=0, algname=alg)
        d.hosh.show()

        ret = step(d, db, storages, sched)
        if ret:
            bacc_c0, bacc_r0, r2_c0, r2_r0, hits_c0, hits_r0, tot0 = ret
            print(f"\r{sp=} {delta=} {trees=} {bacc_c0=:4.3f} {bacc_r0=:4.3f} {r2_c0=:4.3f} {r2_r0=:4.3f} {hits_c0=}  {hits_r0=} {tot0=}\t{d.hosh.ansi}", flush=True)

        # permutation test
        scores_dct = dict(bacc_c=[], bacc_r=[], r2_c=[], r2_r=[])
        for i in ap[1, 2, ..., perms]:
            df_shuffled = df.copy()
            df_shuffled[targetvar] = rnd.permutation(df[targetvar].values)
            d["i", "df"] = i, df_shuffled
            ret = step(d, db, storages, sched)
            if ret:
                bacc_c, bacc_r, r2_c, r2_r, hits_c, hits_r, tot = ret
                scores_dct["bacc_c"].append(bacc_c - bacc_c0)
                scores_dct["bacc_r"].append(bacc_r - bacc_r0)
                scores_dct["r2_c"].append(r2_c - r2_c0)
                scores_dct["r2_r"].append(r2_r - r2_r0)

        if sched:
            print("Run again without providing flag `sched`.")
            continue

        if scores_dct:
            print(f"\n{sp=} p-values: ", end="")
            for measure, scores in scores_dct.items():
                p = ttest_1samp(scores, popmean=0, alternative="greater")[1]
                print(f"  {measure}={p:4.3f}", end="")
        print("\n")

"""
# filtered species
p=0;s="sched";t=1200;a=lgbm;pre="results/datasetr_species";suf="_bayley_8_t2.csv"; ps; time poetry run python experiments/microbiome/pairwise.py prefix=$pre sufix=$suf targetvar=bayley_8_t2 alg=$a delta=7.5 trees=$t jobs=-1 perms=$p diff $s; ps; time poetry run python experiments/microbiome/pairwise.py prefix=$pre sufix=$suf targetvar=bayley_8_t2 alg=$a delta=7.5 trees=$t jobs=-1 perms=$p $s; time poetry run python experiments/microbiome/pairwise.py prefix=$pre sufix=$suf targetvar=bayley_8_t2 alg=$a delta=0.2 trees=$t jobs=-1 perms=$p pct $s; time poetry run python experiments/microbiome/pairwise.py prefix=$pre sufix=$suf targetvar=bayley_8_t2 alg=$a delta=0.2 trees=$t jobs=-1 perms=$p pct diff $s; # filtered species
p=0;s="";t=1200;a=lgbm;pre="results/datasetr_species";suf="_bayley_8_t2.csv"; ps; time poetry run python experiments/microbiome/pairwise.py prefix=$pre sufix=$suf targetvar=bayley_8_t2 alg=$a delta=7.5 trees=$t jobs=1 perms=$p diff $s; ps; time poetry run python experiments/microbiome/pairwise.py prefix=$pre sufix=$suf targetvar=bayley_8_t2 alg=$a delta=7.5 trees=$t jobs=1 perms=$p $s; time poetry run python experiments/microbiome/pairwise.py prefix=$pre sufix=$suf targetvar=bayley_8_t2 alg=$a delta=0.2 trees=$t jobs=1 perms=$p pct $s; time poetry run python experiments/microbiome/pairwise.py prefix=$pre sufix=$suf targetvar=bayley_8_t2 alg=$a delta=0.2 trees=$t jobs=1 perms=$p pct diff $s; # filtered species

# full species
p=0;s="sched";t=1200;a=lgbm;pre="results/datasetr_fromtsv_species";suf="_bayley_8_t2.csv"; ps; time poetry run python experiments/microbiome/pairwise.py prefix=$pre sufix=$suf targetvar=bayley_8_t2 alg=$a delta=7.5 trees=$t jobs=-1 perms=$p diff $s; ps; time poetry run python experiments/microbiome/pairwise.py prefix=$pre sufix=$suf targetvar=bayley_8_t2 alg=$a delta=7.5 trees=$t jobs=-1 perms=$p $s; time poetry run python experiments/microbiome/pairwise.py prefix=$pre sufix=$suf targetvar=bayley_8_t2 alg=$a delta=0.2 trees=$t jobs=-1 perms=$p pct $s; time poetry run python experiments/microbiome/pairwise.py prefix=$pre sufix=$suf targetvar=bayley_8_t2 alg=$a delta=0.2 trees=$t jobs=-1 perms=$p pct diff $s; # full species
p=0;s="";t=1200;a=lgbm;pre="results/datasetr_fromtsv_species";suf="_bayley_8_t2.csv"; ps; time poetry run python experiments/microbiome/pairwise.py prefix=$pre sufix=$suf targetvar=bayley_8_t2 alg=$a delta=7.5 trees=$t jobs=1 perms=$p diff $s; ps; time poetry run python experiments/microbiome/pairwise.py prefix=$pre sufix=$suf targetvar=bayley_8_t2 alg=$a delta=7.5 trees=$t jobs=1 perms=$p $s; time poetry run python experiments/microbiome/pairwise.py prefix=$pre sufix=$suf targetvar=bayley_8_t2 alg=$a delta=0.2 trees=$t jobs=1 perms=$p pct $s; time poetry run python experiments/microbiome/pairwise.py prefix=$pre sufix=$suf targetvar=bayley_8_t2 alg=$a delta=0.2 trees=$t jobs=1 perms=$p pct diff $s; # full species

# gera EEG CSVs: single=1 dyadic=2
poetry run python experiments/microbiome/pairwise.py geneeg targetvar=bayley_8_t2
"""
