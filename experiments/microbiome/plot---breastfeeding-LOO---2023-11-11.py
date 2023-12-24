import warnings
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
from lightgbm import LGBMClassifier as LGBMc
from shelchemy import sopen
from sklearn import clone
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier as ETc, StackingClassifier, VotingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.tree import DecisionTreeClassifier

from germina.config import local_cache_uri, near_cache_uri, remote_cache_uri, schedule_uri
from germina.runner import ch
from hdict import hdict, _, apply

warnings.filterwarnings('ignore')
algs = {"RFc": RandomForestClassifier, "LGBMc": LGBMc, "ETc": ETc, "Sc": StackingClassifier, "MVc": VotingClassifier, "hardMVc": VotingClassifier, "CART": DecisionTreeClassifier, "Perceptron": Perceptron, "Dummy": DummyClassifier}

# exp, id = "species34_c_section", "jIgdwdP1oDI416bOLM0ZQrhm.U8blbJTNIy5UNzN"  # EBF
# exp, id = "species2_bayley_8_t2", "6pmEcmVAWWWfLJnCMb8vo4ZoYl0AHHD-2g4oJ28N"  # cognition
# exp, id = "species2_bayley_8_t2", "rKecIDcHJIFNBfKi26hmK-k2Tsim2wAW3QWtdrCN"  # cognition
# exp, id = "species2_bayley_8_t2", "jIMFNh5FvZQQl4bzhKsbloQDvbFSia4LVidZrnZX"  # cognition
exp, id = "species2_bayley_8_t2", "t7-iOQHf6wI16IpDFG15l.cbX2zc8w49kwpiMTID"  # cognition

with (sopen(local_cache_uri, ondup="skip") as local_storage, sopen(near_cache_uri, ondup="skip") as near_storage, sopen(remote_cache_uri, ondup="skip") as remote_storage, sopen(schedule_uri) as db):
    storages = {
        "remote": remote_storage,
        "near": near_storage,
        "local": local_storage,
    }
    d = hdict.load(id, local_storage)
    d.show()

    d["X"] = _[f"X_{exp}"]
    d["y"] = _[f"y_{exp}"]
    d["y0"] = d.y
    d["y"] = d.y.to_numpy()
    d.apply(StandardScaler, out="stsc")
    d.apply(StandardScaler.fit_transform, _.stsc, out="X")
    d.apply(PCA, out="pca")
    d.apply(PCA.fit_transform, _.pca, out="X")
    d.apply(MinMaxScaler, out="mmsc")
    d.apply(MinMaxScaler.fit_transform, _.mmsc, out="X")
    d = ch(d, storages, to_be_updated="")
    pos = d.y == 1
    neg = d.y == 0

    #################################################################################################################
    # Confusion Matrix
    #################################################################################################################
    predictions = {}
    algs = {k: algs[k] for k in d.algs}
    for algname, algclass in algs.items():
        print(algname)
        predictions[algname] = d[f"{exp}_{algname}_predictions"]

        # Andre
        # print(type(d.y0))
        # pprint(list(sorted(zip(d.y, predictions[algname], d.y0.index))))
        # exit()

        d.apply(algclass, out="alg")
        d.apply(lambda alg, X, y: clone(alg).fit(X, y), out="model_tr")
        d.apply(lambda model_tr: model_tr.classes_, out="display_labels")
        d.apply(ConfusionMatrixDisplay.from_estimator, _.model_tr, out="cm_tr")
        d.apply(ConfusionMatrixDisplay.from_predictions, y_true=_.y, y_pred=_[f"{exp}_{algname}_predictions"], out="cm_ts")
        d.cm_ts.plot()
    plt.show()
    # exit()

    #################################################################################################################
    # For each measure...
    #################################################################################################################
    scores, pvals = {}, {}
    for measure in d.measures:
        scores[measure] = {}
        pvals[measure] = {}
        for algname in d.algs:
            prefix2 = f"{exp}_{algname}_{measure}"
            scores[measure][algname] = d[f"{prefix2}_score"]
            pvals[measure][algname] = d[f"{prefix2}_pval"]

        # SCORE #####################################################################################################
        print(f"{measure}")
        pprint(scores[measure])
        print(f"p-value")
        pprint(pvals[measure])

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
        ntpos = np.count_nonzero((vote_hit == 1) & (d.y == 1))
        ntneg = np.count_nonzero((vote_hit == 1) & (d.y == 0))
        npos = np.count_nonzero(d.y == 1)
        nneg = np.count_nonzero(d.y == 0)
        balacc = (ntpos / npos + ntneg / nneg) / 2
        print(f"Balanced Accuracy by majority voting:\t{balacc:0.3f}")

        # continue

        # PLOT #########################################################################################################
        ################## ################## ################## ################## ################## ##################
        ax = plt.subplot(1, 2, 1)
        for (c, m, s, a), (name, y_) in zip([("black", "+", 500, .9), ("green", "^", 100, .9), ("gray", "o", 300, .2)], predictions.items()):
            idx = list(set(np.nonzero(d.y == y_)[0].tolist()).difference(allhit).difference(allmiss))
        X, y = d.X[idx], d.y[idx]
        # ax.scatter(X[:, 0], X[:, 1], color=c, label=name, alpha=a, s=s, marker=m)
        ax.scatter(d.X[allhit, 0], d.X[allhit, 1], color="blue", label="all hit", alpha=0.99, s=70, marker=".")
        ax.scatter(d.X[allmiss, 0], d.X[allmiss, 1], color="red", label="all miss", alpha=0.7, s=50, marker="x")
        ax.scatter(d.X[pos, 0], d.X[pos, 1], color="green", label="high", alpha=0.2, s=150, marker="o")
        ax.scatter(d.X[neg, 0], d.X[neg, 1], color="yellow", label="low", alpha=0.4, s=150, marker="o")
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

    """
    100 trees
    score (p-value):        0.4270 (0.9610) species2-none-balanced_accuracy=RFc
    score (p-value):        0.4558 (0.8751) species2-none-balanced_accuracy=RFc
    score (p-value):        0.4725 (0.7702) species2-none-balanced_accuracy=RFc
    score (p-value):        0.4911 (0.5884) species2-none-balanced_accuracy=RFc
    score (p-value):        0.4914 (0.6394) species2-none-balanced_accuracy=RFc
    score (p-value):        0.4984 (0.4965) species1-none-balanced_accuracy=RFc
    score (p-value):        0.5008 (0.5025) species1-none-balanced_accuracy=RFc
    score (p-value):        0.5071 (0.4166) species2-none-balanced_accuracy=RFc
    score (p-value):        0.5137 (0.3487) species1-none-balanced_accuracy=RFc
    score (p-value):        0.5143 (0.3327) species1-none-balanced_accuracy=RFc
    score (p-value):        0.5173 (0.3427) species1-none-balanced_accuracy=RFc
    score (p-value):        0.5191 (0.2987) species1-none-balanced_accuracy=RFc
    score (p-value):        0.5237 (0.1289) species1-none-balanced_accuracy=RFc
    score (p-value):        0.5352 (0.1948) species2-none-balanced_accuracy=RFc
    score (p-value):        0.5380 (0.1159) species1-none-balanced_accuracy=RFc
    score (p-value):        0.5467 (0.0949) species1-none-balanced_accuracy=RFc
    score (p-value):        0.5475 (0.1119) species2-none-balanced_accuracy=RFc
    score (p-value):        0.5509 (0.0569) species1-none-balanced_accuracy=RFc
    score (p-value):        0.5948 (0.0050) species2-none-balanced_accuracy=RFc
    score (p-value):        0.6297 (0.0010) species2-none-balanced_accuracy=RFc
    
    1000 trees
    score (p-value):        0.4357 (0.9301) species2-none-balanced_accuracy=RFc
    score (p-value):        0.4662 (0.8042) species2-none-balanced_accuracy=RFc
    score (p-value):        0.4671 (0.8092) species1-none-balanced_accuracy=RFc
    score (p-value):        0.4927 (0.5784) species1-none-balanced_accuracy=RFc
    score (p-value):        0.4928 (0.6074) species1-none-balanced_accuracy=RFc
    score (p-value):        0.5057 (0.4555) species2-none-balanced_accuracy=RFc
    score (p-value):        0.5065 (0.4366) species1-none-balanced_accuracy=RFc
    score (p-value):        0.5072 (0.3846) species2-none-balanced_accuracy=RFc
    score (p-value):        0.5073 (0.4236) species2-none-balanced_accuracy=RFc
    score (p-value):        0.5090 (0.4066) species2-none-balanced_accuracy=RFc
    score (p-value):        0.5213 (0.2837) species1-none-balanced_accuracy=RFc
    score (p-value):        0.5220 (0.3047) species1-none-balanced_accuracy=RFc
    score (p-value):        0.5228 (0.2797) species2-none-balanced_accuracy=RFc
    score (p-value):        0.5373 (0.1159) species1-none-balanced_accuracy=RFc
    score (p-value):        0.5387 (0.1598) species1-none-balanced_accuracy=RFc
    score (p-value):        0.5438 (0.0969) species1-none-balanced_accuracy=RFc
    score (p-value):        0.5464 (0.1039) species1-none-balanced_accuracy=RFc
    score (p-value):        0.5735 (0.0140) species2-none-balanced_accuracy=RFc
    score (p-value):        0.5766 (0.0200) species2-none-balanced_accuracy=RFc
    score (p-value):        0.6170 (0.0030) species2-none-balanced_accuracy=RFc
    """
