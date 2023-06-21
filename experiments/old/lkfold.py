from statistics import mean

from matplotlib import pyplot as plt
from pandas import DataFrame
from sklearn.ensemble import RandomForestClassifier as RFC
from datasets import load_dataset
from lange import ap
from sklearn.model_selection import StratifiedKFold as SKF, cross_val_score, KFold as KF

from germina.config import local_cache_uri, remote_cache_uri
from hdict import hdict, apply, cache, _
from hdict.dataset.dataset import df2Xy, nom2bin
from hdict.dataset.pandas_handling import explode_df, file2df
from shelchemy import sopen


def scores2lists(scores):
    mxs, accs, mns = [], [], []
    for k, scores in scores.items():
        mxs.append(max(scores))
        accs.append(mean(scores))
        mns.append(min(scores))
    return mxs, accs, mns


d = hdict(filename="/home/davi/research/dataset/abalone-3class.arff", n=480, trees=10, scoring=None)
with sopen(remote_cache_uri) as remote, sopen(local_cache_uri) as local:
    # caches = cache(remote) >> cache(local)
    caches = cache(local)
    d >>= apply(file2df)("df", "name") >> caches
    d >>= apply(DataFrame.sample, _.df).df >> caches
    d >>= apply(nom2bin, _.df, nomcols=[0]).df >> caches
    d >>= apply(df2Xy, _.df, "c")("X", "y") >> caches

    d["ks"] = [2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 24, 30, 40, 60, 120]
    scores = {}
    for k in d.ks:
        l = d.n // k
        scores[k] = []
        for seed in ap[1, 2, ..., l]:
            d["k", "seed"] = k, seed
            d >>= apply(SKF, n_splits=_.k, shuffle=True, random_state=_.seed).cv >> caches
            d >>= apply(RFC, n_estimators=_.trees).estimator >> caches
            d >>= apply(cross_val_score, n_jobs=-1).scores >> caches
            scores[k].extend(d.scores)
    d["scores"] = scores
    d >>= apply(scores2lists, _.scores)("mxs", "accs", "mns") >> caches
    fig, ax = plt.subplots()
    ax.plot(d.ks, d.accs)
    ax.fill_between(d.ks, d.mns, d.mxs, color='b', alpha=.1)
    plt.show()
