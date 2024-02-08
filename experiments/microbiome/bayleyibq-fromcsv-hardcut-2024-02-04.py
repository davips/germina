from lightgbm import LGBMClassifier as LGBMc, LGBMRegressor as LGBMr
import pandas as pd
from lange import ap, gp
from pandas import read_csv
from sklearn.model_selection import permutation_test_score
from xgboost import XGBClassifier

for i in [1, 2]:
    df = read_csv(f"/home/davi/git/germina/results/datasetr_species{i}_bayley_8_t2.csv", index_col="id_estudo")
    # print(df.shape)
    # print("---------------")
    # df.sort_values("idade_crianca_dias_t2", inplace=True)  # age at bayley test
    age = df["idade_crianca_dias_t2"]
    yr = df["bayley_8_t2"]

    for delta in ap[8, 9, 10, 11, 12]:
        hiidx = df.index[yr > 100 + delta]
        loidx = df.index[yr < 100 - delta]
        print("s:", i, "d:", delta, "balance:", len(loidx), len(hiidx), end="\t")
        X = pd.concat([df.loc[loidx], df.loc[hiidx]])
        del X["bayley_8_t2"]
        del X["idade_crianca_dias_t2"]
        hiy = yr[hiidx].astype(int) * 0 + 1
        loy = yr[loidx].astype(int) * 0
        y = pd.concat([loy, hiy])

        # t = PCA(n_components=10)
        # X = MDS(n_components=70).fit_transform(X)
        # X = TSNE(n_components=50, method="exact").fit_transform(X)

        for trees in gp[30000]:
            # alg = RandomForestClassifier(n_estimators=trees, n_jobs=-1)
            # alg = CatBc(n_estimators=trees)
            alg = LGBMc(n_estimators=trees, n_jobs=-1)
            # alg = XGBClassifier(n_estimators=trees, n_jobs=-1)
            # score, permscores, pval = permutation_test_score(rf, X=X, y=y, scoring="r2", n_permutations=1, n_jobs=-1)
            score, permscores, pval = permutation_test_score(alg, X=X, y=y, scoring="balanced_accuracy", n_permutations=1, n_jobs=-1)
            print("\t\t", trees, "score:", score, pval, sep="\t")
