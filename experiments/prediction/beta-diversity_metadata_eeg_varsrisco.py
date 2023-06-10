from math import inf

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mdscuda import MDS
from numpy import mean, log, std, ndarray, quantile
from pandas import DataFrame
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sortedness.local import sortedness

from germina.config import local_cache_uri
from hdict import _, apply, cache, hdict
from shelchemy import sopen

path = "data/"
m = _.fromfile(path + "data_microbiome___2023-05-10___beta_diversity_distance_matrix_T1.csv").df
m.set_index("d", inplace=True)
d = hdict(delta=m, random_state=0, sqform=True, n_dims=484)
with sopen(local_cache_uri) as local:  # , sopen(remote_cache_uri) as remote:
    d >>= apply(MDS)("mds") >> apply(MDS.fit, _.mds)("p") >> cache(local)
    p = d.p
df_betadiv = DataFrame(p)
biomeattrs = [f"m{c}" for c in range(d.n_dims)]
df_betadiv.columns = biomeattrs
data = {
    # "eeg": _.fromfile(path + "data_eeg___2023-03-15___VEP-N1---covariates-et-al---Average-et-al.csv").df,
    "metadata": _.fromfile(path + "metadata___2023-05-08-fup5afixed.csv").df[["id_estudo", "ibq_reg_t1", "ibq_reg_t2"]],
    "varsrisco": _.fromfile(path + "nathalia170523_variaveis_risco___2023-06-08.csv").df,
}

dfs = [df.set_index("id_estudo") for df in data.values()]
df: DataFrame = df_betadiv.join(dfs, how="outer")
print("shape with NaNs", df.shape)


# Remove NaNs preserving maximum amount of data.
def nans(df):
    nans_hist = df.isna().sum()
    print("Removing NaNs...", df.shape, end="\t\t")
    print(nans_hist.to_numpy().tolist())
    nans_ = sum(nans_hist)
    return nans_


while nans(df):
    # Remove rows.
    s = df.isna().sum(axis=1)
    df = df[s.ne(s.max()) | s.eq(0)]
    print("After removing worst rows:", df.shape)

    # Backup targets.
    bkp = {tgt: df[tgt] for tgt in ["risco_class", "ibq_reg_t1", "ibq_reg_t2"] + biomeattrs}

    # Remove columns.
    s = df.isna().sum(axis=0)
    df = df.loc[:, s.ne(s.max()) | s.eq(0)]

    # Recover targets. (and microbioma)
    for tgt, col in bkp.items():
        if tgt not in list(df.columns):
            df = pd.concat((df, col), axis=1)

    print("After removing worst columns:", df.shape)
    print()

print("shape after NaN cleaning", df.shape)

if "e01_t1" in df:
    del df["e01_t1"]
if "antibiotic" in df:
    df["antibiotic"] = (df["antibiotic"] == "yes").astype(int)
if "EBF_3m" in df:
    df["EBF_3m"] = (df["EBF_3m"] == "EBF").astype(int)
if "renda_familiar_total_t0" in df:
    df["renda_familiar_total_t0"] = log(df["renda_familiar_total_t0"])
print("shape after some problematic attributes removal", df.shape)

# Drop targets.
df.sort_values(by="risco_class", inplace=True)
df0 = df.copy()
targets = []
for c in df.columns.values.tolist():
    if c.startswith("ibq_") or c.startswith("bayley_") or c.startswith("risco_"):
        targets.append(str(df.pop(c).name))
targets = ["risco_class", "ibq_reg_t1", "ibq_reg_t2", "ibq_reg_t2-ibq_reg_t1"]
print()
print(targets)
print("final shape after targets removal", df.shape)

# Standardize.
st = StandardScaler()
s: ndarray = st.fit_transform(df)
pca = PCA(n_components=df.shape[1])
s = pca.fit_transform(s)
df = DataFrame(s, columns=list(df.columns))
d["targets"] = targets
print()

d["df"] = df
d["n_dims"] = 100
with sopen(local_cache_uri) as local:  # , sopen(remote_cache_uri) as remote:
    d >>= (
            apply(cdist, XA=_.df, XB=_.df, metric="sqeuclidean").delta
            >> apply(MDS).mds >> apply(MDS.fit, _.mds)("p")
            >> cache(local)
    )
    d.evaluate()

cm = "Set1_r"
cm = "viridis"
cm = "coolwarm"
for target in targets:
    if "-" in target:
        a, b = target.split("-")
        labels = df0[a] - df0[b]
        labels = np.where(labels <= -1, -3, labels)
        # labels = np.where((labels > -1) & (labels < 1), 0, labels)
        # labels = np.where(labels >= 1, 1, labels)
        labels *= -1
        scale = 150
    else:
        labels = df0[target]
        if target.startswith("ibq_reg_t"):
            qmn, qmx = quantile(labels, [1 / 4, 3 / 4])
            menores = labels[labels <= qmn].index
            maiores = labels[labels >= qmx].index
            pd.options.mode.chained_assignment = None
            labels.loc[menores] = -1
            labels.loc[maiores] = 1
            labels.loc[list(set(labels.index).difference(maiores, menores))] = 0
            pd.options.mode.chained_assignment = "warn"
            scale = 150
        else:
            labels = np.where(labels == 0, 20, labels)
            labels = np.where(labels == 1, 10, labels)
            labels = np.where(labels == 2, 30, labels)
            scale = 200
    mn, mx = min(labels), max(labels)
    for ndims in [2, 3][:]:
        with sopen(local_cache_uri) as local:  # , sopen(remote_cache_uri) as remote:
            d >>= (
                    apply(TSNE, n_components=ndims, random_state=0, method='exact', n_jobs=-1).tsne >> apply(TSNE.fit_transform, X=_.p).r
                    >> apply(sortedness, _.df, _.r).q
                    >> cache(local)
            )
        r, q = d.r, d.q
        print(f"{target} {ndims}d".ljust(25), mean(q), std(q), sep="\t")
        t0 = -1
        t1 = 0.1
        bad = (q >= t0) & (q <= t1)
        sizes = np.where(bad, 10, q * scale)
        colors = labels
        if ndims == 2:
            fig = plt.figure()
            plt.title(f"{target} ({ndims})")
            ax = fig.add_subplot(111)
            ax.scatter(r[:, 0], r[:, 1], vmin=mn, vmax=mx, c=colors, cmap=cm, s=sizes)
            if "-" in target:
                mask = colors == 3
                colors = colors[mask]
                sizes = sizes[mask]
                # mask = mask.reshape(-1, 1) # & [True, True]
                r = r[mask, :]
            ax.scatter(r[:, 0], r[:, 1], vmin=mn, vmax=mx, c=colors, cmap=cm, s=sizes)
        elif ndims == 3:
            fig = plt.figure()
            plt.title(f"{target} ({ndims})")
            ax = fig.add_subplot(111, projection="3d")
            ax.scatter(r[:, 0], r[:, 1], r[:, 2], vmin=mn, vmax=mx, c=colors, cmap=cm, s=sizes)

plt.show()
