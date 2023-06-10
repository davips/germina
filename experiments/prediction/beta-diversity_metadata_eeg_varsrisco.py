from sklearn.ensemble import RandomForestClassifier as RF
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from numpy import mean, std, quantile
from sklearn.manifold import TSNE
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
targets = ["risco_class", "ibq_reg_t1", "ibq_reg_t2", "ibq_reg_t2-ibq_reg_t1"]
d = clean(targets, "data/", files, [local_cache_uri])
cm = "coolwarm"
for target in targets:
    if "-" in target:
        a, b = target.split("-")
        labels = d.targets[a] - d.targets[b]
        labels = np.where(labels <= -1, -3, labels)
    else:
        labels = d.targets[target]
        if target.startswith("ibq_reg_t"):
            qmn, qmx = quantile(labels, [1 / 4, 3 / 4])
            menores = labels[labels <= qmn].index
            maiores = labels[labels >= qmx].index
            pd.options.mode.chained_assignment = None
            labels.loc[menores] = -1
            labels.loc[maiores] = 1
            labels.loc[list(set(labels.index).difference(maiores, menores))] = 0
        else:
            labels = np.where(labels == 0, 20, labels)
            labels = np.where(labels == 1, 10, labels)
            labels = np.where(labels == 2, 30, labels)

    with sopen(local_cache_uri) as local:  # , sopen(remote_cache_uri) as remote:
        d >>= (
                apply(RF, n_estimators=1000, random_state=0, n_jobs=-1).rf
                >> apply(RF.fit, _.rf, X=_.p, y=labels).none
                >> apply(RF.predict, _.rf, X=_.p).r
                >> cache(local)
        )
        print(d.r)
