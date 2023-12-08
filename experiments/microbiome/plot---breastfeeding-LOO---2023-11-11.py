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
    id = "U4w8lX0KIr6KXT1EL3Ks.LBMrTERm-ydj6wICe-o"
    d = hdict.load(id, local_storage)

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

    predictions = {"RFc": d.species34_c_section_RFc_predictions, "LGBMc": d.species34_c_section_LGBMc_predictions, "ETc": d.species34_c_section_ETc_predictions}
    scores = {"RFc": d.species34_c_section_RFc_average_precision_score_score, "LGBMc": d.species34_c_section_LGBMc_average_precision_score_score, "ETc": d.species34_c_section_ETc_average_precision_score_score}
    pvals = {"RFc": d.species34_c_section_RFc_average_precision_score_pval, "LGBMc": d.species34_c_section_LGBMc_average_precision_score_pval, "ETc": d.species34_c_section_ETc_average_precision_score_pval}
    balacc_scores = {"RFc": d.species34_c_section_RFc_balanced_accuracy_score, "LGBMc": d.species34_c_section_LGBMc_balanced_accuracy_score, "ETc": d.species34_c_section_ETc_balanced_accuracy_score}
    balacc_pvals = {"RFc": d.species34_c_section_RFc_balanced_accuracy_pval, "LGBMc": d.species34_c_section_LGBMc_balanced_accuracy_pval, "ETc": d.species34_c_section_ETc_balanced_accuracy_pval}

    # SCORE #####################################################################################################
    pprint(scores)
    pprint(pvals)
    pprint(balacc_scores)
    pprint(balacc_pvals)

    atleast1hit, atleast1miss = set(), set()
    for name, y_ in predictions.items():
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
    a = np.vstack(list(predictions.values()))
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
    for (c, m, s, a), (name, y_) in zip([("black", "+", 500, .9), ("green", "^", 100, .9), ("gray", "o", 300, .2)], predictions.items()):
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
    for (c, m, s, a), (name, y_) in zip([("black", "+", 500, .9), ("green", "^", 100, .9), ("gray", "o", 300, .2)], predictions.items()):
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
