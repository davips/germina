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

# exp, id = "species34_c_section", "?????????"  # EBF
exp, id = "species1_ibq_dura_t3", "BFLlB6IUDyIp6FffZkw9zicVY604y3s5aj6cdOaU"  # cognition
div = 3
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
    d["yor"] = _[exp]
    d["y"] = d.y.to_numpy()
    d.apply(StandardScaler, out="stsc")
    d.apply(StandardScaler.fit_transform, _.stsc, out="X")
    d.apply(PCA, out="pca")
    d.apply(PCA.fit_transform, _.pca, out="X")
    d.apply(MinMaxScaler, out="mmsc")
    d.apply(MinMaxScaler.fit_transform, _.mmsc, out="X")
    d = ch(d, storages, to_be_updated="")
    pos = d.y == 1
    neu = d.y == 0
    neg = neu if div == 2 else (d.y == -1)

    #################################################################################################################
    # Confusion Matrix
    #################################################################################################################
    predictions = {}
    algs = {k: algs[k] for k in d.algs}
    for algname, algclass in algs.items():
        print(algname)
        predictions[algname] = d[f"{exp}_{algname}_predictions"]

        # Andre
        # print(type(d.yor))
        # pprint(list(sorted(zip(d.y, predictions[algname], d.yor.index))))
        # exit()

        d.apply(algclass, out="alg")
        d.apply(lambda alg, X, y: clone(alg).fit(X, y), out="model_tr")
        d.apply(lambda model_tr: model_tr.classes_, out="display_labels")
        d.apply(ConfusionMatrixDisplay.from_estimator, _.model_tr, out="cm_tr")
        d.apply(ConfusionMatrixDisplay.from_predictions, y_true=_.y, y_pred=_[f"{exp}_{algname}_predictions"], out="cm_ts")
        # d.cm_tr.plot()
        # d.cm_ts.plot()
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
        for algname in d.algs:
            y_ = predictions[algname]
            for tgt in range(3):
                print("target", tgt)
                ax = plt.subplot(1, 3, tgt + 1)
                idxhit = (d.y == y_) & (d.y == tgt)
                idxmiss = (d.y != y_) & (d.y == tgt)
                Xh, yorh = d.X[idxhit], d.yor[idxhit]
                Xm, yorm = d.X[idxmiss], d.yor[idxmiss]
                print(yorm)
                ax.scatter(Xh[:, 0], Xh[:, 1], c=yorh, label=f"hit class {tgt}",  s=100, marker="o", alpha=.2)
                ax.scatter(Xm[:, 0], Xm[:, 1], c=yorm, label=f"miss class {tgt}", s=150, marker="x")
                ax.set_xlim([0.0, 1.0])
                ax.set_ylim([0.0, 1.0])
                plt.grid()
                plt.xlabel('PC 1')
                plt.ylabel('PC 2')
                plt.title("Correct Predictions")
                plt.legend()

        # ax = plt.subplot(1, 2, 1)
        # for (c, m, s, a), (name, y_) in zip([("black", "+", 500, .9), ("green", "^", 100, .9), ("gray", "o", 300, .2)], predictions.items()):
        #     idx = list(set(np.nonzero(d.y == y_)[0].tolist()).difference(allhit).difference(allmiss))
        # X, y = d.X[idx], d.y[idx]
        # ax.scatter(d.X[allhit, 0], d.X[allhit, 1], color="blue", label="all hit", alpha=0.99, s=70, marker=".")
        # ax.scatter(d.X[allmiss, 0], d.X[allmiss, 1], color="red", label="all miss", alpha=0.7, s=50, marker="x")
        # ax.scatter(d.X[pos, 0], d.X[pos, 1], color="green", label="high", alpha=0.2, s=150, marker="o")
        # ax.scatter(d.X[neg, 0], d.X[neg, 1], color="yellow", label="low", alpha=0.4, s=150, marker="o")
        # ax.set_xlim([0.0, 1.0])
        # ax.set_ylim([0.0, 1.0])
        # plt.grid()
        # plt.xlabel('PC 1')
        # plt.ylabel('PC 2')
        # plt.title("Correct Predictions")
        # plt.legend()

        mng = plt.get_current_fig_manager()
        # mng.resize(*mng.window.maxsize())
        mng.resize(1800, 900)

        plt.show()
