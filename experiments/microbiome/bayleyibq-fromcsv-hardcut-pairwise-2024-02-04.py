from sys import argv

import numpy as np
import pandas as pd
from lange import ap
from lightgbm import LGBMClassifier as LGBMc
from pandas import read_csv, DataFrame
from sklearn.model_selection import LeaveOneOut

from germina.pairwise import pairwise_diff, pairwise_hstack


def cut(df, low, high, low2=0, high2=999):
    if high is None:
        hiidx = df.index[yr > 100]
        loidx = df.index[yr <= 100]
    else:
        hiidx = df.index[(yr > high) & (yr < high2)]
        loidx = df.index[(yr < low) & (yr > low2)]
    X = pd.concat([df.loc[loidx], df.loc[hiidx]])
    del X["bayley_8_t2"]
    del X["idade_crianca_dias_t2"]
    hiy = yr[hiidx].astype(int) * 0 + 1
    loy = yr[loidx].astype(int) * 0
    y = pd.concat([loy, hiy])
    return X, y, len(hiidx), len(loidx)


M, delta, trees, sp = 5, 7.5, 64, int(argv[1])
# print("sp", sp, flush=True)
# print(f"id,label,1-label,ma,mb,mc,md,vote,mean,ma,mb,mc,md,vote", flush=True)
df = read_csv(f"results/datasetr_species{sp}_bayley_8_t2.csv", index_col="id_estudo")
age = df["idade_crianca_dias_t2"]
yr = df["bayley_8_t2"]

X0, y0, _, _ = cut(df, None, None)
X, y, s0, s1 = [DataFrame()] * M, [DataFrame()] * M, [DataFrame()] * M, [DataFrame()] * M
start = 100 - delta / 2
step = delta / (M - 1)
for i in ap[0, 1, ..., M - 1]:
    # print(start + i * step - delta / 2, start + i * step + delta / 2)
    X[i], y[i], s0[i], s1[i] = cut(df, start + i * step - delta / 2, start + i * step + delta / 2)
# exit()
loo = LeaveOneOut()
for _, its in loo.split(X0):
    # idxtr = X0.index[itr]
    idxts = X0.index[its]
    Xbaby = X0.loc[idxts]
    ybaby = y0.loc[idxts]
    id = idxts.tolist()[0]
    label = ybaby.tolist()[0]
    preds, probs, hits = [], [], []
    for m in ap[0, 1, ..., M - 1]:
        alg = LGBMc(n_estimators=trees, n_jobs=1)  # todo X_SHAP_values
        if id in X[m].index:
            X_ = X[m].drop([id], axis="rows")
            y_ = y[m].drop([id], axis="rows")
        else:
            X_ = X[m]
            y_ = y[m]

        # Make 0 the first label.
        y_.sort_values(inplace=True)
        X_ = X_.loc[y_.index]

        alg.fit(X_, y_)
        pred = alg.predict(Xbaby)[0]
        preds.append(pred)
        probs.append(alg.predict_proba(Xbaby).tolist())
        hits.append(str(int(label == pred)))
    avg = float(np.mean(preds))
    vote = int(round(avg))
    print(f"{id},{label},{1 - label},{','.join(str(p) for p in preds)},{vote},{avg:.5f},{','.join(hits)},{int(label == vote)}", flush=True)
