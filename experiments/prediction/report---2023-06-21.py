import pandas as pd
from mdscuda import MDS
from numpy import log, ndarray
from pandas import DataFrame
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from hdict import _, apply, cache, hdict
from shelchemy import sopen, memory
from pprint import pprint

from matplotlib import pyplot as plt
from pandas import Series, DataFrame
from sklearn.dummy import DummyClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold, cross_val_score, cross_val_predict
import numpy as np
import pandas as pd
from numpy import quantile, mean, std
from sklearn.ensemble import RandomForestClassifier as RFc
from sklearn.ensemble import RandomForestRegressor as RFr

from germina.config import local_cache_uri, remote_cache_uri
from germina.data import clean
from germina.nan import remove_nan_rows, backup_cols, hasNaN, remove_nan_cols, remove_cols, bina, loga
from hdict import _, apply, cache, hdict
from shelchemy import sopen

"""
'ebia_tot_t2', 'ebia_2c_t2', 'bayley_1_t2', 'bayley_2_t2',
       'bayley_3_t2', 'bayley_6_t2', 'bayley_16_t2', 'bayley_7_t2',
       'bayley_17_t2', 'bayley_18_t2', 'bayley_8_t2', 'bayley_11_t2',
       'bayley_19_t2', 'bayley_12_t2', 'bayley_20_t2', 'bayley_21_t2',
       'bayley_13_t2', 'bayley_22_t2', 'bayley_23_t2', 'bayley_24_t2',
       'risco_total_t0'
"""

# Build a single dataset.
filenames = [
    "data_microbiome___2023-06-18___alpha_diversity_n525.csv",
    "data_microbiome___2023-06-20___vias_metabolicas_3_meses_n525.csv",
    "data_microbiome___2023-06-18___especies_3_meses_n525.csv",
    "data_eeg___2023-06-20___T1_RS_average_dwPLI_withEEGCovariates.csv",
    "data_eeg___2023-06-20___T2_RS_average_dwPLI_withEEGCovariates.csv",
    # "metadata___2023-06-18.csv", ["id_estudo", "ibq_reg_t1", "ibq_reg_t2", "risco_class"]
    "metadata___2023-06-18.csv"
]
path = "data/"
dfs_noidx = [_.fromfile(path + filename).df for filename in filenames]
dfs = [df.set_index("id_estudo") for df in dfs_noidx]
df: DataFrame = dfs[0].join(dfs[1:], how="outer")
print("Original shape:", df.shape)
with sopen(local_cache_uri) as local, sopen(remote_cache_uri) as remote:
    d = hdict(df=df, random_state=0)

    print("Remove undesired attributes...", end="\t")
    undesired = [
        "risco_class",  # Colocar de volta depois de conferido por Pedro.
        "final_8_t1", "final_10_t1", "e01_t1",
        "ibq*",
        "bayley*"
    ]
    keep = ["ibq_reg_t1", "ibq_reg_t2"]
    d >>= apply(remove_cols, cols=undesired, keep=keep).df

    print("Remove NaNs.", end="\t")
    while hasNaN(d.df):
        d >>= apply(remove_nan_cols, keep=[], debug=False).df
        d >>= apply(remove_nan_rows, debug=False).df
    print("New shape:", d.df.shape)
    print(d.df.columns.to_numpy().tolist())

    print("Format problematic attributes.")
    if "antibiotic" in df:
        d >>= apply(bina, attribute="antibiotic", positive_category="yes").df
    if "EBF_3m" in df:
        d >>= apply(bina, attribute="EBF_3m", positive_category="EBF").df
    if "renda_familiar_total_t0" in df:
        d >>= apply(loga, attribute="renda_familiar_total_t0").df

    # Cache.
    d >>= cache(remote) >> cache(local)
    exit()
    exit()
    exit()
    exit()
    exit()
    exit()

    # Reduce dimensionality before applying t-SNE.
    todf = lambda X, cols: DataFrame(X, columns=cols)
    d["n_dims"] = 100
    cols = list(df.columns)
    with sopen(local_cache_uri) as local, sopen(remote_cache_uri) as remote:
        d >>= apply(StandardScaler.fit_transform, StandardScaler(), _.raw_df).std >> apply(todf, _.std, cols).std_df
        d >>= apply(cdist, XA=_.std_df, XB=_.std_df, metric="sqeuclidean").delta
        d >>= apply(PCA, n_components=min(*df.shape)).pca_model >> apply(PCA.fit_transform, _.pca_model, _.std_df).pca >> apply(todf, _.pca, None).pca_df
        # d >>= apply(MDS).mds >> apply(MDS.fit, _.mds).data100 >> apply(todf, _.data100, None).data100_df
        d >>= cache(remote) >> cache(local)
        d.evaluate()

    pprint([col[:80] for col in d.raw_df.columns])

    cm = "coolwarm"

    if "-" in target:
        a, b = target.split("-")
        labels = d.targets[a] - d.targets[b]
    else:
        labels = d.targets[target]
        if target.startswith("ibq_reg_t"):
            # qmn, qmx = quantile(labels, [1 / 4, 3 / 4])
            qmn, qmx = quantile(labels, [1 / 2, 1 / 2])
            menores = labels[labels <= qmn].index
            maiores = labels[labels >= qmx].index
            pd.options.mode.chained_assignment = None
            labels.loc[menores] = -1
            labels.loc[maiores] = 1
            labels.loc[list(set(labels.index).difference(maiores, menores))] = 0

    X = d.raw_df
    y = labels  # np.where(labels == 2, 1, labels)
    cv = KFold(n_splits=5, random_state=0, shuffle=True)
    with sopen(local_cache_uri) as local, sopen(remote_cache_uri) as remote:
        d >>= apply(RFc, n_estimators=1000, random_state=0, n_jobs=-1).rfc
        d >>= apply(cross_val_score, _.rfc, X=X, y=y, cv=cv, n_jobs=-1).scoresc
        d >>= apply(DummyClassifier).dc
        d >>= apply(cross_val_score, _.dc, X=X, y=y, cv=cv, n_jobs=-1).baseline
        d >>= apply(RFr, n_estimators=1000, random_state=0, n_jobs=-1).rfr
        d >>= apply(cross_val_score, _.rfr, X=_.raw_df, y=labels, cv=cv, scoring="r2", n_jobs=-1).scoresr
        d >>= cache(remote) >> cache(local)

        print(confusion_matrix(y, cross_val_predict(d.rfc, X, y)))
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

    # daf: DataFrame = d.df["idade_crianca_meses_t1"]
    # th = 3.6
    # lo, hi = daf[daf <= th], daf[daf > th]
    # print(lo.count(), hi.count())
    # # mn, mx = quantile(out, [1 / 3, 2 / 3])
    #
    #
    # outcome: DataFrame = d.df["ibq_reg_t2"]
    # print(list(outcome.sort_values()))
    #
    # tho = outcome.median()
    # a, b = daf[outcome <= tho], daf[outcome > tho]
    #
    # # plt.hist([a, b], bins=12)
    #
    #
    # outcome2: DataFrame = d.df["bayley_3_t2"]
    # outcome2.hist(bins=22)
    # plt.show()
