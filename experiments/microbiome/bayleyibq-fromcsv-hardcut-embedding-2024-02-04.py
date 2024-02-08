from lightgbm import LGBMClassifier as LGBMc, LGBMRegressor as LGBMr
import pandas as pd
from lange import ap, gp
from pandas import read_csv, DataFrame
from sklearn.model_selection import permutation_test_score
from xgboost import XGBClassifier

from sortedness.embedding.sortedness_ import balanced_embedding

for d in gp[1, 2, ..., 80]:
    for i in [1, 2]:
        df = read_csv(f"/home/davi/git/germina/results/datasetr_species{i}_bayley_8_t2.csv", index_col="id_estudo")
        # print(df.shape)
        # print("---------------")
        # df.sort_values("idade_crianca_dias_t2", inplace=True)  # age at bayley test
        age = df["idade_crianca_dias_t2"]
        yr = df["bayley_8_t2"]

        Xe = df.copy()
        del Xe["bayley_8_t2"]
        del Xe["idade_crianca_dias_t2"]
        Xe = DataFrame(balanced_embedding(Xe.to_numpy(), d=d), index=Xe.index)

        print()
        for delta in ap[3, 4, ..., 10]:
            hiidx = df.index[yr > 100 + delta]
            loidx = df.index[yr < 100 - delta]
            print("d:", d, "s:", i, "delta:", delta, "balance:", len(loidx), len(hiidx), end="\t")
            X = pd.concat([Xe.loc[loidx], Xe.loc[hiidx]])
            hiy = yr[hiidx].astype(int) * 0 + 1
            loy = yr[loidx].astype(int) * 0
            y = pd.concat([loy, hiy])

            # t = PCA(n_components=10)
            # X = MDS(n_components=70).fit_transform(X)
            # X = TSNE(n_components=50, method="exact").fit_transform(X)

            for trees in gp[1000]:
                # alg = RandomForestClassifier(n_estimators=trees, n_jobs=-1)
                # alg = CatBc(n_estimators=trees)
                alg = LGBMc(n_estimators=trees, n_jobs=-1)
                # alg = XGBClassifier(n_estimators=trees, n_jobs=-1)
                # score, permscores, pval = permutation_test_score(rf, X=X, y=y, scoring="r2", n_permutations=1, n_jobs=-1)
                score, permscores, pval = permutation_test_score(alg, X=X, y=y, scoring="balanced_accuracy", n_permutations=1, n_jobs=-1)
                print("\t\t", trees, "score:", score, pval, sep="\t")
