from lange import ap
from sklearn.decomposition import PCA
import numpy as np
from matplotlib import pyplot as plt
from numpy import mean, log, std, ndarray, quantile, median
from pandas import DataFrame
from sklearn.manifold import smacof, TSNE
from sklearn.preprocessing import StandardScaler
from sortedness import global_pwsortedness
from sortedness.local import sortedness, pwsortedness

from germina.config import local_cache_uri, remote_cache_uri
from hdict import _, hdict, apply, cache
from shelchemy import sopen

path = "data/"
m = _.fromfile(path + "data_microbiome___2023-05-10___beta_diversity_distance_matrix_T1.csv").df
m.set_index("d", inplace=True)
p = smacof(m, metric=False, n_components=len(m) // 2, n_jobs=18, random_state=0, normalized_stress=True)[0]
df_betadiv = DataFrame(p)
df_betadiv.columns = [f"m{c}" for c in range(len(m) // 2)]
data = {
    "eeg": _.fromfile(path + "data_eeg___2023-03-15___VEP-N1---covariates-et-al---Average-et-al.csv").df,
    "metadata": _.fromfile(path + "metadata___2023-05-08-fup5afixed.csv").df,
}

dfs = [df.set_index("id_estudo") for df in data.values()]
df: DataFrame = df_betadiv.join(dfs, how="outer")
print("shape with NaNs", df.shape)

# Remove NaNs preserving maximum amount of data.
while sum(df.isna().sum()) > 0:
    s = df.isna().sum(axis=1)
    df = df[s.ne(s.max()) | s.eq(0)]
    s = df.isna().sum(axis=0)
    df = df.loc[:, s.ne(s.max()) | s.eq(0)]
print("shape", df.shape)

del df["e01_t1"]
df["antibiotic"] = (df["antibiotic"] == "yes").astype(int)
df["EBF_3m"] = (df["EBF_3m"] == "EBF").astype(int)
df["renda_familiar_total_t0"] = log(df["renda_familiar_total_t0"])
print("final shape", df.shape)


# print(df.to_numpy().tolist())


def tsne(n):
    return TSNE(n_components=n, random_state=0, method='exact', n_jobs=18).fit_transform(df)


df0 = df.copy()
targets = []
for c in df.columns.values.tolist():
    if c.startswith("ibq_") or c.startswith("bayley_"):
        targets.append(str(df.pop(c).name))
targets = ["ibq_reg_t1"]  # , "ibq_reg_t2"]
print(targets)
# colors = df["EBF_3m"]
colors = df["infant_ethinicity"]
# colors = df.pop("ibq_reg_t1")
# colors = df.pop("ibq_reg_t2")
# del df["maternal_ethinicity"]

st = StandardScaler()
s: ndarray = st.fit_transform(df)

pca = PCA(n_components=df.shape[1])
s = pca.fit_transform(s)

explained = DataFrame(pca.components_[:4], columns=list(df.columns))
print(explained[explained >= 0.05].transpose().dropna(axis="rows", how="all").to_string())

df = DataFrame(s, columns=list(df.columns))

# fig = plt.figure()
# ax = fig.add_subplot(111, projection="3d")
# ax.scatter(p[:, 0], p[:, 1], p[:, 2], c=colors, cmap="coolwarm")
# plt.show()

# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.scatter(p[:, 0], p[:, 1], c=colors, cmap="coolwarm")
# plt.show()

d = hdict(df=df, targets=targets)
cm = "coolwarm"
for target in targets:
    out = df0[target]
    # mn, mx = quantile(out, [1 / 3, 2 / 3])
    # mn, mx = quantile(out, [1 / 5, 4 / 5])
    # qmn, qmx = quantile(out, [1 / 6, 5 / 6])
    # qmn, qmx = quantile(out, [1 / 10, 9 / 10])

    # print(qmn, qmx)
    # menores = out[out <= qmn].index
    # maiores = out[out >= qmx].index
    # out.loc[menores] = -1
    # out.loc[maiores] = 1
    # out.loc[list(set(out.index).difference(maiores, menores))] = 0

    mn, mx = min(out), max(out)
    zero = (mn + mx) / 2

    # mn, mx = -1, 1
    labels = out

    for ndims in [2, 3][:1]:
        with sopen(local_cache_uri) as local:  # , sopen(remote_cache_uri) as remote:
            d >>= apply(tsne, ndims).p >> apply(sortedness, _.df, _.p).q >> cache(local)
            p, q = d.p, d.q
        # print(global_pwsortedness(df, p), mean(pwsortedness(df, p)), mean(q), sep="\n")
        print(mean(q), std(q))
        t0 = -1
        t1 = 0.3
        colors = np.where((q >= t0) & (q <= t1), zero, labels)
        # colors = np.where(colors > t1, 1, colors)
        # colors = np.where(colors < t0, -1, colors)
        if ndims == 2:
            fig = plt.figure()
            plt.title(f"{target} ({ndims})")
            ax = fig.add_subplot(111)
            ax.scatter(p[:, 0], p[:, 1], vmin=mn, vmax=mx, c=colors, cmap=cm, s=50)
        elif ndims == 3:
            fig = plt.figure()
            plt.title(f"{target} ({ndims})")
            ax = fig.add_subplot(111, projection="3d")
            ax.scatter(p[:, 0], p[:, 1], p[:, 2], vmin=mn, vmax=mx, c=colors, cmap=cm, s=50)

plt.show()
