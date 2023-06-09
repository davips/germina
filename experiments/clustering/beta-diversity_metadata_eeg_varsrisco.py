import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from numpy import mean, std, quantile
from sklearn.ensemble import RandomForestClassifier as RFc
from sklearn.manifold import TSNE
from sklearn.model_selection import KFold, cross_val_predict
from sortedness.local import sortedness

from germina.config import local_cache_uri
from germina.data import clean
from hdict import _, apply, cache
from shelchemy import sopen

files = [
    ("data_microbiome___2023-05-10___beta_diversity_distance_matrix_T1.csv", None),
    ("data_eeg___2023-03-15___VEP-N1---covariates-et-al---Average-et-al.csv", None),
    ("metadata___2023-05-08-fup5afixed.csv", ["id_estudo", "ibq_reg_t1", "ibq_reg_t2"]),
    ("nathalia170523_variaveis_risco___2023-06-08.csv", None)
]
targets = ["risk_score"] #, "ibq_reg_t1", "ibq_reg_t2", "ibq_reg_t2-ibq_reg_t1"]
d = clean(targets, "data/", files, [local_cache_uri])
cm = "coolwarm"
for target in targets:
    if "-" in target:
        a, b = target.split("-")
        labels = d.targets[a] - d.targets[b]
        colors = labels.copy()
        colors = np.where(colors <= -1, -3, colors)
        # labels = np.where((labels > -1) & (labels < 1), 0, labels)
        # labels = np.where(labels >= 1, 1, labels)
        colors *= -1
        scale = 150
    else:
        labels = d.targets[target]
        colors = labels.copy()
        if target.startswith("ibq_reg_t"):
            qmn, qmx = quantile(colors, [1 / 4, 3 / 4])
            menores = colors[colors <= qmn].index
            maiores = colors[colors >= qmx].index
            pd.options.mode.chained_assignment = None
            colors.loc[menores] = -1
            colors.loc[maiores] = 1
            colors.loc[list(set(colors.index).difference(maiores, menores))] = 0
            pd.options.mode.chained_assignment = "warn"
            scale = 150
        else:
            colors = np.where(colors == 0, 20, colors)
            colors = np.where(colors == 1, 10, colors)
            colors = np.where(colors == 2, 30, colors)
            scale = 200
    labels = [int(l) for l in labels]
    # print(labels)
    mn, mx = min(colors), max(colors)
    cv = KFold(n_splits=20, random_state=0, shuffle=True)
    for ndims in [2, 3]:
        with sopen(local_cache_uri) as local:  # , sopen(remote_cache_uri) as remote:
            d >>= (
                    apply(RFc, n_estimators=1000, random_state=0, n_jobs=-1).rfc
                    >> apply(cross_val_predict, _.rfc, X=_.raw_df, y=labels, cv=cv, n_jobs=-1).preds
                    >> apply(TSNE, n_components=ndims, random_state=0, method='exact', n_jobs=-1).tsne >> apply(TSNE.fit_transform, _.tsne, X=_.data100_df).proj
                    >> apply(sortedness, _.std_df, _.proj).q
                    >> cache(local)
            )
            d.evaluate()
        r, q = d.proj, d.q
        print(f"{target} {ndims}d".ljust(25), mean(q), std(q), sep="\t")
        t0 = -1
        t1 = 0.1
        bad = (q >= t0) & (q <= t1)
        sizes = np.where(bad, 10, q * scale)
        if ndims == 2:
            fig = plt.figure()
            plt.title(f"{target} ({ndims})")
            ax = fig.add_subplot(111)

            ax.scatter(r[:, 0], r[:, 1], vmin=mn, vmax=mx, c=colors, cmap=cm, s=sizes)

            miss = (d.preds != labels) #& (labels != 0)
            ax.scatter(r[miss, 0], r[miss, 1], c="gray", s=1000)

            if "-" in target:
                mask = colors == 3
                colors = colors[mask]
                sizes = sizes[mask]
                # mask = mask.reshape(-1, 1) # & [True, True]
                r = r[mask, :]
            ax.scatter(r[:, 0], r[:, 1], vmin=mn, vmax=mx, c=colors, cmap=cm, s=sizes)
        # elif ndims == 3:
        #     fig = plt.figure()
        #     plt.title(f"{target} ({ndims})")
        #     ax = fig.add_subplot(111, projection="3d")
        #     ax.scatter(r[:, 0], r[:, 1], r[:, 2], vmin=mn, vmax=mx, c=colors, cmap=cm, s=sizes)

plt.show()
