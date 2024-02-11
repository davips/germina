from sys import argv

from argvsucks import handle_command_line
import numpy as np
import pandas as pd
from lange import ap
from lightgbm import LGBMClassifier as LGBMc
from pandas import read_csv, DataFrame
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

dct = handle_command_line(argv, delta=int, trees=int)
trees, delta = dct["trees"], dct["delta"]
for sp in [1, 2]:
    df = read_csv(f"/home/davi/git/germina/results/datasetr_species{sp}_bayley_8_t2.csv", index_col="id_estudo")
    # df = df[:60]
    print(df.shape)
    age = df["idade_crianca_dias_t2"]
    # X0 = df.drop("bayley_8_t2", axis="columns")
    # y0 = df["bayley_8_t2"]
    hits = {0: 0, 1: 0}
    tot = hits.copy()
    print()
    c = 0
    for idx in df.index:
        print(f"\r {100 * c / len(df):8.5f}%")
        c += 1

        # current baby
        xy = df.loc[[idx], :].to_numpy()

        # training set
        Xy0 = df.drop(idx, axis="rows").to_numpy()
        Xy_tr = pairwise_hstack(Xy0, Xy0, handle_last_as_y=True)
        Xy_tr = Xy_tr[(Xy_tr[:, -1] < -delta) | (Xy_tr[:, -1] > delta)]
        X_tr = Xy_tr[:, :-1]
        y_tr = (Xy_tr[:, -1] >= 0).astype(int)

        alg = LGBMc(n_estimators=trees, n_jobs=-1)
        alg.fit(X_tr, y_tr)

        Xy_ts = pairwise_hstack(xy, Xy0, handle_last_as_y=True)
        Xy_ts = Xy_ts[(Xy_ts[:, -1] < -delta) | (Xy_ts[:, -1] > delta)]
        X_ts = Xy_ts[:, :-1]
        y_ts = (Xy_ts[:, -1] >= 0).astype(int)
        zs = alg.predict(X_ts)
        for z, l in zip(zs, y_ts):
            hits[l] += int(z == l)
            tot[l] += 1

    acc0 = hits[0] / tot[0]
    acc1 = hits[1] / tot[1]
    score = (acc0 + acc1) / 2

    print(f"\t", trees, "score:", score, sep="\t")
