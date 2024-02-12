from pprint import pprint

from lightgbm import LGBMClassifier as LGBMc, LGBMRegressor as LGBMr
import pandas as pd
from lange import ap, gp
from matplotlib import pyplot as plt
from pandas import read_csv, DataFrame
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from sklearn.model_selection import permutation_test_score, LeaveOneOut, StratifiedKFold
from xgboost import XGBClassifier
import numpy as np

from itertools import product

import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.ensemble import VotingClassifier
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

n_wrongmodels = [np.array([0])] * 3
n_wrongmodels[1] = np.array([(5.3 * (float(v) + 1)) ** 1.43 for v in """5
8
3
3
3
5
9
5
10
7
4
2
4
4
3
0
1
0
7
6
5
9
4
6
9
4
1
0
6
6
1
9
1
9
8
6
1
3
8
3
0
7
1
2
8
1
1
5
2
8
6
0
1
6
3
2
1
4
2
8
7
4
4
10
0
3
8
0
8
0
0
8
1
4
6
1
4
3
4
2
5
9
0
1
5
6
3
10
4
5
2
10
4
5
0
8
9
5
0
0
3
4
7
5
4
3
4
3
0
7
1
0
3
3
3
8
9
0
0
2
6
3
5
2
0
5
8
7
2
4
8
8
9
2
4
3
1
3
1
4
10
4
8
9
3
6
4
5
1
2
2
3
8
2
6
6
4
0
2
0
7
4
4
8
6
2
7
5
5
4
5
3
1
2
1
7
3
1
4
4
9
1
2
10
6
8
2
2
5
2
0
5
2
9
8
9
4
4
7
2
0
6
2
4
1
4
3
2
9
8
4
8
8
2
7
6
4
8
10
5
4
6
1
4
5
9
9
7
5
8
5
8
6
6
7
6
10
8
8
6
5
9
0
9
5
7
10
6
10
7
5
3
9
2
5
3
2
0
7
3
0
7
0
7
10
7
2
1
8
1
10
7
4
7
2
10
5
1
0
6
6
8
3
6""".split("\n")])
n_wrongmodels[2] = np.array([(5.3 * (float(v) + 1)) ** 1.43 for v in """8
4
2
10
6
5
2
4
3
4
9
9
8
1
4
3
5
9
10
3
0
6
4
1
6
9
0
3
3
3
6
4
1
8
3
0
0
4
7
2
2
7
1
3
1
1
10
8
5
3
0
10
0
6
0
8
1
8
0
3
3
7
6
0
3
5
8
4
1
2
2
5
5
3
1
7
6
4
3
2
6
7
1
2
8
1
1
1
5
3
6
1
3
1
0
6
8
8
3
0
2
4
0
5
7
4
10
2
5
3
4
5
2
1
5
10
5
4
0
0
4
2
0
4
8
0
9
0
6
2
3
1
8
2
3
2
0
3
5
6
3
1
7
4
0
8
5
3
8
5
4
2
2
6
5
8
7
5
5
0
9
6
9
8
5
3
4
3
6
0
0
9
4
0
6
5
3
2
7
2
10
0
3
2
7
2
6
7
7
6
3
2
6
10
2
4
7
10
8
0
7
4
5
8
1
6
10
6
10
6
4
5
9
9
10
8
4
9
1
0
4
0
9
3
8
6
1
0
9
7
6
2
6
3
3
0
7
7
1
3
3
3
4
7
0
1
6
7
2
4
1
0
0
4""".split("\n")])
ax = axd = None
rejcolor = "gray"
for sp in [1, 2]:
    scolab = f"Balanced Accuracy T{sp}"
    df = read_csv(f"/home/davi/git/germina/results/datasetr_species{sp}_bayley_8_t2.csv", index_col="id_estudo")
    # df.drop(a, axis="columns", inplace=True)
    # df.drop(b, axis="columns", inplace=True)
    print(df.shape)
    # print(df.shape)
    # print("---------------")
    # df.sort_values("idade_crianca_dias_t2", inplace=True)  # age at bayley test
    age = df["idade_crianca_dias_t2"]
    yr = df["bayley_8_t2"]

    # hiidx = df.index[yr >= 100.0]  # non arbitrary scale-based
    # loidx = df.index[yr < 100.0]
    # hiidx = df.index[yr >= 107.5]  # scale-based
    # loidx = df.index[yr <= 92.5]
    hiidx = df.index[yr >= 107.86]  # sample-based
    loidx = df.index[yr <= 99.06]

    print("sp:", sp, "balance:", len(loidx), len(hiidx))
    X = pd.concat([df.loc[loidx], df.loc[hiidx]])
    del X["bayley_8_t2"]
    # del X["idade_crianca_dias_t2"]
    hiy = yr[hiidx].astype(int) * 0 + 1
    loy = yr[loidx].astype(int) * 0
    y = pd.concat([loy, hiy])
    X = MDS(random_state=0).fit_transform(X)

    # Training classifiers ###################################################
    # clf1 = DecisionTreeClassifier(max_depth=4)
    # clf2 = KNeighborsClassifier(n_neighbors=7)
    # clf3 = SVC(gamma=0.1, kernel="rbf", probability=True)
    # eclf = VotingClassifier(
    #     estimators=[("dt", clf1), ("knn", clf2), ("svc", clf3)],
    #     voting="soft",
    #     weights=[2, 1, 2],
    # )
    # clf1.fit(X, y)
    # clf2.fit(X, y)
    # clf3.fit(X, y)
    # eclf.fit(X, y)

    # clfs = [clf1, clf2, clf3, eclf]
    # f, axarr = plt.subplots(2, 2, sharex="col", sharey="row", figsize=(10, 8))
    clfs = [LGBMc(n_estimators=64, n_jobs=8).fit(X, y)]
    f, axarr = plt.subplots(1, 1, sharex="col", sharey="row", figsize=(10, 8))

    for idx, clf, tt in zip(product([0, 1], [0, 1]), clfs, ["LGBM", "Decision Tree (depth=4)", "KNN (k=7)", "Kernel SVM", "Soft Voting"], ):
        # DecisionBoundaryDisplay.from_estimator(clf, X, alpha=0.4, ax=axarr[idx[0], idx[1]], response_method="predict")
        # axarr[idx[0], idx[1]].scatter(X[:, 0], X[:, 1], c=y, s=20, edgecolor="k")
        # axarr[idx[0], idx[1]].set_title(tt)
        DecisionBoundaryDisplay.from_estimator(clf, X, alpha=0.4, ax=axarr, response_method="predict")
        b = y.astype(bool)
        axarr.scatter(X[b, 0], X[b, 1], label="High", c="green", s=n_wrongmodels[sp][b], edgecolor="k", alpha=0.6)
        axarr.scatter(X[~b, 0], X[~b, 1], label="Low", c="purple", s=n_wrongmodels[sp][~b], edgecolor="k", alpha=0.6)
        axarr.set_title(f"Decision Boundary for LightGBM classifier T{sp}. Size: # of models missing the target")
    plt.legend()
    plt.show()
