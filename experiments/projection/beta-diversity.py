import numpy as np
import pathos.multiprocessing as mp
import seaborn as sns
from matplotlib import pyplot as plt
from numpy import mean, array, ndarray
from numpy.random import shuffle
from pandas import DataFrame
from scipy.spatial.distance import cdist
from scipy.stats import weightedtau
from sklearn.manifold import smacof, TSNE
from sortedness import global_pwsortedness
from sortedness.local import remove_diagonal, pwsortedness, sortedness as sortedness0

from hdict import _

sns.color_palette("colorblind")
vars_path = "/data/data_microbiome___2023-05-10___beta_diversity_distance_matrix_T1.csv"
outcome_path = "/data/metadata___2023-05-08-fup5afixed.csv"

targets = ["ibq_reg_t2"]

dm = _.fromfile(vars_path)
m: DataFrame = dm.df
m.set_index("d", inplace=True)
print(m)

dout = _.fromfile(outcome_path)
dfo: DataFrame = dout.df
dfo.set_index("id_estudo", inplace=True)
dfo = dfo.loc[m.index]

# Reconstruct from DM.
d:ndarray = smacof(m, metric=False, n_components=513, n_jobs=18, random_state=0, normalized_stress=True)[0]


def tsne(n):
    return TSNE(n_components=n, random_state=0,method='exact', n_jobs=18).fit_transform(d)


def sortedness(dist_X, dist_X_):
    f = weightedtau
    kwargs = {"rank": None}
    result, pvalues = [], []
    scores_X = -remove_diagonal(dist_X)
    scores_X_ = -remove_diagonal(dist_X_)
    for i in range(len(dist_X)):
        corr, pvalue = f(scores_X[i], scores_X_[i], **kwargs)
        result.append(round(corr, 12))
        pvalues.append(round(pvalue, 12))
    result = np.array(result, dtype=float)
    return result

d2 = d.copy()
shuffle(d2)
# cm = "gray"
cm = "coolwarm"
for target in  targets:
    menores = dfo[target][dfo[target] <= mean(dfo[target])].index
    maiores = dfo[target][dfo[target] > mean(dfo[target])].index
    dfo[target].loc[menores] = -1
    dfo[target].loc[maiores] = 1

    labels = dfo[target]
    ps = {
        #"NMDS 2": smacof(m, metric=False, n_components=2, n_jobs=18, random_state=0, normalized_stress=True)[0],
        #"NMDS 3": smacof(m, metric=False, n_components=3, n_jobs=18, random_state=0, normalized_stress=True)[0],
        "t-SNE 2": tsne(2),
        "t-SNE 3": tsne(3)
    }

    for k, p in ps.items():
        #print(global_pwsortedness(d, p), mean(pwsortedness(d, p)), mean(sortedness0(d, p)), sep="\n")
        q = sortedness0(d, p)
        # q = sortedness0(d, d2)
        print(mean(q))
        # q = sortedness(m.to_numpy(), cdist(p, p))
        t0=-0
        t1=0.4
        colors = np.where((q > t0) & (q < t1), 0, q)
        colors = np.where(colors > t1, 1, colors)
        colors = np.where(colors < t0, -1, colors)
        if k.endswith("2"):
            fig = plt.figure()
            plt.title(f"{target} ({k})")
            ax = fig.add_subplot(111)
            ax.scatter(p[:, 0], p[:, 1],  vmin=-1, vmax=1, c=-colors, cmap=cm)
        else:
            fig = plt.figure()
            plt.title(f"{target} ({k})")
            ax = fig.add_subplot(111, projection="3d")
            ax.scatter(p[:, 0], p[:, 1], p[:, 2],  vmin=-1, vmax=1, c=-colors, cmap=cm)
    plt.show()
