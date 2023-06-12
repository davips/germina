from sklearn.dummy import DummyClassifier
from sklearn.model_selection import KFold, cross_val_score
import numpy as np
import pandas as pd
from numpy import quantile, mean, std
from sklearn.ensemble import RandomForestClassifier as RFc
from sklearn.ensemble import RandomForestRegressor as RFr

from germina.config import local_cache_uri
from germina.data import clean
from hdict import _, apply, cache
from shelchemy import sopen

files = [
    ("data_microbiome___2023-05-10___beta_diversity_distance_matrix_T1.csv", None),
    # ("data_eeg___2023-03-15___VEP-N1---covariates-et-al---Average-et-al.csv", None),
    ("metadata___2023-05-08-fup5afixed.csv", ["id_estudo", "ibq_reg_t1", "ibq_reg_t2"]),
    # ("metadata___2023-05-08-fup5afixed.csv", None),
    ("nathalia170523_variaveis_risco___2023-06-08.csv", None)
]
targets = ["risco_class", "ibq_reg_t1", "ibq_reg_t2", "ibq_reg_t2-ibq_reg_t1"]
# targets = ["risco_class"]
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

    cv = KFold(n_splits=10, random_state=0, shuffle=True)
    with sopen(local_cache_uri) as local:
        d >>= (
                apply(RFc, n_estimators=1000, random_state=0, n_jobs=-1).rfc
                >> apply(cross_val_score, _.rfc, X=_.raw_df, y=labels, cv=cv, n_jobs=-1).scoresc
                >> apply(DummyClassifier).dc
                >> apply(cross_val_score, _.dc, X=_.raw_df, y=labels, cv=cv, n_jobs=-1).baseline
                >> apply(RFr, n_estimators=1000, random_state=0, n_jobs=-1).rfr
                >> apply(cross_val_score, _.rfr, X=_.raw_df, y=labels, cv=cv, scoring="r2", n_jobs=-1).scoresr
                >> cache(local)
        )
        print(
            f"{target.ljust(20)}\t"
            f"baseline:\t{mean(d.baseline):.2f} ± {std(d.baseline):.2f}\t\t"
            f"RFcla:\t{mean(d.scoresc):.2f} ± {std(d.scoresc):.2f}\t\t"
            f"RFreg:\t{mean(d.scoresr):.2f} ± {std(d.scoresr):.2f}\t\t"
        )
        print(",".join(map(str, d.baseline)))
        print(",".join(map(str, d.scoresc)))
        # print(",".join(d.scoresr))
    print()

""" EEG + Microbiome + SocioDemographic
risco_class         	baseline:	0.80 ± 0.03		RFcla:	0.82 ± 0.03		RFreg:	0.76 ± 0.13		
0.8048780487804879,0.8292682926829268,0.7804878048780488,0.8048780487804879,0.8048780487804879,0.8292682926829268,0.7804878048780488,0.8536585365853658,0.8048780487804879,0.7560975609756098
0.8292682926829268,0.8536585365853658,0.7804878048780488,0.8292682926829268,0.8048780487804879,0.8536585365853658,0.8292682926829268,0.8780487804878049,0.7804878048780488,0.8048780487804879

ibq_reg_t1          	baseline:	0.50 ± 0.07		RFcla:	0.56 ± 0.06		RFreg:	0.71 ± 0.06		
0.43902439024390244,0.5609756097560976,0.43902439024390244,0.5853658536585366,0.43902439024390244,0.4146341463414634,0.4146341463414634,0.5121951219512195,0.5853658536585366,0.5853658536585366
0.4878048780487805,0.6585365853658537,0.4634146341463415,0.5609756097560976,0.5365853658536586,0.5609756097560976,0.5853658536585366,0.6097560975609756,0.4878048780487805,0.6097560975609756

ibq_reg_t2          	baseline:	0.50 ± 0.07		RFcla:	0.53 ± 0.07		RFreg:	0.66 ± 0.09		
0.3902439024390244,0.5365853658536586,0.5365853658536586,0.6341463414634146,0.5365853658536586,0.43902439024390244,0.5121951219512195,0.3902439024390244,0.5365853658536586,0.4634146341463415
0.36585365853658536,0.5365853658536586,0.6097560975609756,0.5853658536585366,0.5365853658536586,0.6097560975609756,0.5853658536585366,0.4634146341463415,0.5609756097560976,0.4634146341463415
"""

""" Microbiome
risco_class         	baseline:	0.80 ± 0.04		RFcla:	0.80 ± 0.05		RFreg:	0.69 ± 0.12		
0.7755102040816326,0.7755102040816326,0.8367346938775511,0.7551020408163265,0.875,0.8333333333333334,0.7916666666666666,0.8541666666666666,0.7291666666666666,0.7708333333333334
0.7959183673469388,0.7346938775510204,0.8571428571428571,0.7551020408163265,0.8541666666666666,0.8333333333333334,0.7916666666666666,0.8541666666666666,0.7291666666666666,0.7708333333333334

ibq_reg_t1          	baseline:	0.50 ± 0.08		RFcla:	0.43 ± 0.06		RFreg:	-0.12 ± 0.06		
0.40816326530612246,0.5102040816326531,0.6326530612244898,0.5510204081632653,0.3333333333333333,0.5625,0.4791666666666667,0.4791666666666667,0.4583333333333333,0.5625
0.3877551020408163,0.42857142857142855,0.5102040816326531,0.46938775510204084,0.2916666666666667,0.4375,0.4791666666666667,0.4583333333333333,0.3541666666666667,0.4791666666666667

ibq_reg_t2          	baseline:	0.50 ± 0.07		RFcla:	0.47 ± 0.07		RFreg:	-0.12 ± 0.08		
0.5714285714285714,0.5510204081632653,0.4489795918367347,0.6122448979591837,0.4791666666666667,0.4166666666666667,0.5625,0.4583333333333333,0.4583333333333333,0.4166666666666667
0.5510204081632653,0.5510204081632653,0.40816326530612246,0.5918367346938775,0.4166666666666667,0.375,0.4583333333333333,0.4375,0.5,0.3958333333333333

ibq_reg_t2-ibq_reg_t1	baseline:	0.50 ± 0.09		RFcla:	0.46 ± 0.07		RFreg:	-0.17 ± 0.12		
0.4897959183673469,0.46938775510204084,0.5510204081632653,0.4897959183673469,0.4166666666666667,0.3958333333333333,0.6458333333333334,0.6458333333333334,0.3958333333333333,0.5416666666666666
0.46938775510204084,0.4489795918367347,0.5306122448979592,0.4489795918367347,0.4375,0.2916666666666667,0.5416666666666666,0.5416666666666666,0.3958333333333333,0.4791666666666667
"""
