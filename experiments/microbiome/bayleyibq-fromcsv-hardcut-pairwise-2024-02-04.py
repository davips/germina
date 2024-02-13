from pprint import pprint
from sys import argv

from argvsucks import handle_command_line
import numpy as np
import pandas as pd
from lange import ap
from lightgbm import LGBMClassifier as LGBMc
from pandas import read_csv, DataFrame
from sklearn.metrics import r2_score
from sklearn.model_selection import LeaveOneOut

from germina.pairwise import pairwise_diff, pairwise_hstack

from lightgbm import LGBMClassifier as LGBMc, LGBMRegressor as LGBMr
import pandas as pd
from lange import ap, gp
from pandas import read_csv
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from sklearn.model_selection import permutation_test_score, LeaveOneOut, StratifiedKFold
from xgboost import XGBClassifier


def interpolate(targets, conditions):
    """
    :param targets:
        sorted
    :param conditions:
        `1` means the resulting value should be greater than the corresponding target.
        `0` means the resulting value should be equal than the corresponding target. (`0` is not usually needed)
        `-1` means the resulting value should be lesser than the corresponding target.
    :return:

    >>> tgts = np.array([77,88,81,84,88,90,95,100,103,105,110,112,115,120])
    >>> conds = np.array([1,1,-1,1,1,-1,1,-1,-1,1,-1,-1,1,-1])
    >>> interpolate(tgts, conds)
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


dct = handle_command_line(argv, delta=float, trees=int, jobs=int)
pprint(dct)
trees, delta, jobs = dct["trees"], dct["delta"], dct["jobs"]
for sp in [1, 2]:
    df = read_csv(f"results/datasetr_species{sp}_bayley_8_t2.csv", index_col="id_estudo")
    # df = df[:160]
    df.sort_values("bayley_8_t2", inplace=True, ascending=True)
    # print(df.shape)
    age = df["idade_crianca_dias_t2"]
    y = df["bayley_8_t2"]
    for delta in ap[3,6,...,15]:
        filter = lambda tmp: (tmp[:, -1] < -delta) | (tmp[:, -1] > delta)
        hits_ = {0: 0, 1: 0}
        tot_ = hits_.copy()
        hits = {0: 0, 1: 0}
        tot = hits.copy()
        z_lst = []
        c = 0
        for idx in df.index:
            # if c % 15 == 0:
            #     print(f"\r {100 * c / len(df):8.5f}%")
            c += 1

            # current baby
            baby = df.loc[[idx], :].to_numpy()
            baby_x = baby[:, :-1]
            baby_y = baby[0, -1]

            # training set
            # Xy_tr = df.to_numpy()
            Xy_tr = df.drop(idx, axis="rows").to_numpy()
            tmp = pairwise_hstack(Xy_tr, Xy_tr, handle_last_as_y=True)
            pairs_Xy_tr = tmp[filter(tmp)]
            pairs_X_tr = pairs_Xy_tr[:, :-1]
            pairs_y_tr = (pairs_Xy_tr[:, -1] >= 0).astype(int)

            alg = LGBMc(n_estimators=trees, n_jobs=jobs)
            alg.fit(pairs_X_tr, pairs_y_tr)

            # test set
            tmp = pairwise_hstack(baby, Xy_tr, handle_last_as_y=True)
            fltr = filter(tmp)
            pairs_Xy_ts = tmp[fltr]
            pairs_X_ts = pairs_Xy_ts[:, :-1]
            pairs_y_ts = (pairs_Xy_ts[:, -1] >= 0).astype(int)

            # predictions
            pairs_z_ts = alg.predict(pairs_X_ts)

            # interpolation
            targets = Xy_tr[fltr, -1]
            conditions = 2 * pairs_z_ts - 1
            baby_z = interpolate(targets, conditions)
            z_lst.append(baby_z)

            expected = int(baby_y >= 100)
            predicted = int(baby_z >= 100)
            hits[expected] += int(expected == predicted)
            tot[expected] += 1
        z = np.array(z_lst)
        acc0 = hits[0] / tot[0]
        acc1 = hits[1] / tot[1]
        balacc = (acc0 + acc1) / 2
        r2 = r2_score(y, z)
        # for y0, z0 in zip(y, z):
        #     print(y0, z0)

        print(f"{sp=} {delta=} {trees=} {balacc=} {r2=:5.3f} {hits=} {tot=}")
