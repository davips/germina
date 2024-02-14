import copy
from itertools import repeat
from pprint import pprint
from sys import argv

import numpy as np
import pandas as pd
from argvsucks import handle_command_line
from lange import ap
from pandas import read_csv
from scipy.stats import ttest_1samp
from shelchemy import sopen
from shelchemy.scheduler import Scheduler
from sklearn.metrics import r2_score

from germina.config import local_cache_uri, remote_cache_uri, near_cache_uri, schedule_uri
from germina.pairwise import pairwise_diff, pairwise_hstack
from germina.runner import ch
from hdict import hdict

if "rf" in argv:
    from sklearn.ensemble import RandomForestClassifier as LGBMc, RandomForestRegressor as LGBMr

    # REMINDER: we are using LGBMc,LGBMr as aliases to the actual selected algorithms to avoid changing the caching of `substep()` previous results
    rf = True
else:
    from lightgbm import LGBMClassifier as LGBMc, LGBMRegressor as LGBMr

    rf = False


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


def substep(df, idx, z_lst_c, z_lst_r, hits_c, hits_r, tot, trees, delta, diff, handle_last_as_y):
    filter = lambda tmp: (tmp[:, -1] < -delta) | (tmp[:, -1] >= delta)
    if diff:
        hstack = lambda a, b: pairwise_diff(a, b, pct=handle_last_as_y == "%")
    else:
        hstack = lambda a, b: pairwise_hstack(a, b, handle_last_as_y=handle_last_as_y)
    z_lst_c, z_lst_r = copy.deepcopy(z_lst_c), copy.deepcopy(z_lst_r)
    hits_c, hits_r = copy.deepcopy(hits_c), copy.deepcopy(hits_r)
    tot = copy.deepcopy(tot)

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

    alg_c = LGBMc(n_estimators=trees, n_jobs=-1, random_state=0)
    alg_r = LGBMr(n_estimators=trees, n_jobs=-1, random_state=0)
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
    z_lst_c.append(baby_z_c)
    z_lst_r.append(baby_z_r)

    expected = int(baby_y >= 100)
    predicted_c = int(baby_z_c >= 100)
    predicted_r = int(baby_z_r >= 100)
    hits_c[expected] += int(expected == predicted_c)
    hits_r[expected] += int(expected == predicted_r)
    tot[expected] += 1
    return z_lst_c, z_lst_r, hits_c, hits_r, tot


def step(d, db, storages, sched):
    y = d.df[d.target_variable].to_numpy()
    hits_c, hits_r = {0: 0, 1: 0}, {0: 0, 1: 0}
    tot = {0: 0, 1: 0}
    z_lst_c, z_lst_r = [], []
    d = d >> dict(z_lst_c=z_lst_c, z_lst_r=z_lst_r, hits_c=hits_c, hits_r=hits_r, tot=tot)
    tasks = zip(repeat(d.id + ("_RF" if rf else "")), d.df.index)
    ansi = d.hosh.ansi
    for c, (id, idx) in enumerate((Scheduler(db, timeout=60) << tasks) if sched else tasks):
        if not sched:
            print(f" permutation: {d.i:8}\t\t{ansi} babies: {100 * c / len(d.df):8.5f}%", end="\n", flush=True)
        d.apply(substep, idx=idx, out=("z_lst_c", "z_lst_r", "hits_c", "hits_r", "tot"))
        d = ch(d, storages)
    if sched:
        return
    z_c = np.array(d.z_lst_c)
    z_r = np.array(d.z_lst_r)

    acc0 = d.hits_c[0] / d.tot[0]
    acc1 = d.hits_c[1] / d.tot[1]
    bacc_c = (acc0 + acc1) / 2

    acc0 = d.hits_r[0] / d.tot[0]
    acc1 = d.hits_r[1] / d.tot[1]
    bacc_r = (acc0 + acc1) / 2

    r2_c = r2_score(y, z_c)
    r2_r = r2_score(y, z_r)
    return bacc_c, bacc_r, r2_c, r2_r, d.hits_c, d.hits_r, d.tot


dct = handle_command_line(argv, delta=float, trees=int, pct=False, diff=False, demo=False, sched=False, perms=1, targetvar=str)
pprint(dct)
trees, delta, pct, diff, demo, sched, perms, targetvar = dct["trees"], dct["delta"], dct["pct"], dct["diff"], dct["demo"], dct["sched"], dct["perms"], dct["targetvar"]
rnd = np.random.default_rng(0)
handle_last_as_y = "%" if pct else True

with (sopen(local_cache_uri, ondup="skip") as local_storage, sopen(near_cache_uri, ondup="skip") as near_storage, sopen(remote_cache_uri, ondup="skip") as remote_storage, sopen(schedule_uri) as db):
    storages = {
        # "remote": remote_storage,
        "near": near_storage,
        "local": local_storage,
    }
    for sp in [1, 2]:
        print(f"{sp=} <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
        df = read_csv(f"results/datasetr_species{sp}_bayley_8_t2.csv", index_col="id_estudo")
        df.sort_values(targetvar, inplace=True, ascending=True, kind="stable")
        if demo:
            df = pd.concat([df.iloc[:15], df.iloc[-15:]], axis="rows")
        age = df["idade_crianca_dias_t2"]

        d = hdict(sp=sp, df=df, handle_last_as_y=handle_last_as_y, trees=trees, target_variable=targetvar, delta=delta, diff=diff, i=0)
        d.show()

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

        print(f"\r{sp=} p-values: ", end="")
        for measure, scores in scores_dct.items():
            p = ttest_1samp(scores, popmean=0, alternative="greater")[1]
            print(f"\t{measure}={p:4.3f}", end="")
        print()
