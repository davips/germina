import warnings
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
import plotly
from germina.runner import ch
from plotly.tools import mpl_to_plotly
from shelchemy import sopen
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from germina.config import local_cache_uri, near_cache_uri, remote_cache_uri, schedule_uri
from hdict import hdict, _

# 3-4 c-section
warnings.filterwarnings('ignore')
# with (sopen(local_cache_uri, ondup="skip") as local_storage):
with (sopen(local_cache_uri, ondup="skip") as local_storage, sopen(near_cache_uri, ondup="skip") as near_storage, sopen(remote_cache_uri, ondup="skip") as remote_storage, sopen(schedule_uri) as db):
    storages = {
        "remote": remote_storage,
        "near": near_storage,
        "local": local_storage,
    }

    id = "P5n7VEl0Dw6Rt.m7mHjJG2MoZaf7u.WsSTT2xJUN"  # U4w8lX0KIr6KXT1EL3Ks.LBMrTERm-ydj6wICe-o"
    d = hdict.load(id, local_storage)
    d.show()
    exit()

    d["X"] = _.X_species34_c_section
    d["y"] = _.y_species34_c_section
    d["y"] = d.y.to_numpy()
    d.apply(StandardScaler, out="stsc")
    d.apply(StandardScaler.fit_transform, _.stsc, out="X")
    d.apply(PCA, out="pca")
    d.apply(PCA.fit_transform, _.pca, out="X")
    d.apply(MinMaxScaler, out="mmsc")
    d.apply(MinMaxScaler.fit_transform, _.mmsc, out="X")
    d = ch(d, storages, to_be_updated="")

    prefix = "species34_c_section"
    algnames = ["RFc", "LGBMc", "ETc", "Sc"]
    measures = ["average_precision_score", "balanced_accuracy"]
    predictions, scores, pvals = {}, {}, {}
    for measure in measures:
        predictions[measure] = {}
        scores[measure] = {}
        pvals[measure] = {}
        for algname in algnames:
            predictions[measure][algname] = d[f"{prefix}_{algname}_predictions"]
            prefix2 = f"{prefix}_{algname}_{measure}"
            scores[measure][algname] = d[f"{prefix2}_score"]
            pvals[measure][algname] = d[f"{prefix2}_pval"]

        # SCORE #####################################################################################################
        print(f"{measure}")
        pprint(scores[measure])
        print(f"p-value")
        pprint(pvals[measure])

        atleast1hit, atleast1miss = set(), set()
        for name, y_ in predictions[measure].items():
            eq = d.y == y_
            atleast1hit.update(np.nonzero(eq)[0])
            atleast1miss.update(np.nonzero(~eq)[0])
        allhit = list(set(range(len(d.X))).difference(atleast1miss))
        allmiss = list(set(range(len(d.X))).difference(atleast1hit))

        # Hit / miss
        n = len(d.y)
        nhit = len(allhit)
        nmiss = len(allmiss)
        print(f"All correct:\t{nhit:3}/{n}\t=\t{100 * nhit / n:0.3f}%")
        print(f"Missed by all:\t{nmiss:3}/{n}\t=\t{100 * nmiss / n:0.3f}%")

        # Voting
        a = np.vstack(list(predictions[measure].values()))
        vote_predictions = np.round(np.mean(a, axis=0))
        vote_hit = d.y == vote_predictions
        vote_nhit = np.count_nonzero(vote_hit)
        print(f"Correct by majority voting:\t{vote_nhit:3}/{n}\t=\t{100 * vote_nhit / n:0.3f}%")
        ntpos = np.count_nonzero(vote_hit == 1)
        ntneg = np.count_nonzero(vote_hit == 0)
        npos = np.count_nonzero(d.y == 1)
        nneg = np.count_nonzero(d.y == 0)
        balacc = (ntpos / npos + ntneg / nneg) / 2
        print(f"Balanced Accuracy by majority voting:\t{balacc:0.3f}")

        # MLP

        exit()

        # PLOT #########################################################################################################
        ################## ################## ################## ################## ################## ##################
        ax = plt.subplot(1, 2, 1)
        for (c, m, s, a), (name, y_) in zip([("black", "+", 500, .9), ("green", "^", 100, .9), ("gray", "o", 300, .2)], predictions[measure].items()):
            idx = list(set(np.nonzero(d.y == y_)[0].tolist()).difference(allhit).difference(allmiss))
            X, y = d.X[idx], d.y[idx]
            ax.scatter(X[:, 0], X[:, 1], color=c, label=name, alpha=a, s=s, marker=m)
        ax.scatter(d.X[allhit, 0], d.X[allhit, 1], color="blue", label="all hit", alpha=0.3, s=80, marker=".")
        ax.scatter(d.X[allmiss, 0], d.X[allmiss, 1], color="red", label="all miss", alpha=0.3, s=80, marker="x")
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.0])
        plt.grid()
        plt.xlabel('PC 1')
        plt.ylabel('PC 2')
        plt.title("Correct Predictions")
        plt.legend()
        mng = plt.get_current_fig_manager()
        # mng.resize(*mng.window.maxsize())
        mng.resize(1800, 900)

        ################## ################## ################## ################## ################## ##################
        ax = plt.subplot(1, 2, 2, sharex=ax, sharey=ax)
        for (c, m, s, a), (name, y_) in zip([("black", "+", 500, .9), ("green", "^", 100, .9), ("gray", "o", 300, .2)], predictions[measure].items()):
            idx = list(set(np.nonzero(d.y != y_)[0].tolist()).difference(allhit).difference(allmiss))
            X, y = d.X[idx], d.y[idx]
            ax.scatter(X[:, 0], X[:, 1], color=c, label=name, alpha=a, s=s, marker=m)
        ax.scatter(d.X[allhit, 0], d.X[allhit, 1], color="blue", label="all hit", alpha=0.3, s=80, marker=".")
        ax.scatter(d.X[allmiss, 0], d.X[allmiss, 1], color="red", label="all miss", alpha=0.3, s=80, marker="x")
        plt.grid()
        plt.xlabel('PC 1')
        plt.ylabel('PC 2')
        plt.title("Incorrect Predictions")
        plt.legend()
        mng = plt.get_current_fig_manager()
        # mng.resize(*mng.window.maxsize())
        mng.resize(1800, 900)

        plt.show()
