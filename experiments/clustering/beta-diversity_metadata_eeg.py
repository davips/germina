from sklearn.dummy import DummyClassifier
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as shc
from numpy import mean, log, std, ndarray
from pandas import DataFrame
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.manifold import smacof
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler

from hdict import _

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

# st = StandardScaler()
# s: ndarray = st.fit_transform(df)
# df = DataFrame(s, columns=list(df.columns))
#
# meta0 = set(list(data["metadata"].columns))
# all = set(list(df.columns))
# attrs = list(all.difference(meta0))
# outcomes = list(all.intersection([m for m in meta0 if m.startswith("ibq_") or m.startswith("bayley_")]))
# meta = df[list(all.intersection(meta0))]
# outcome = df[outcomes]
#
# plt.figure(figsize=(10, 7))
# plt.title("Dendrogram")
# Z = shc.linkage(meta, method='ward', metric="euclidean")
# labels = shc.fcluster(Z, 3, criterion='maxclust')
# shc.dendrogram(Z, color_threshold=50)
#
# # cluster_instances = {}
# # for i, label in enumerate(labels):
# #     if label not in cluster_instances:
# #         cluster_instances[label] = []
# #     cluster_instances[label].append(meta.iloc[i])
#
# X = df[attrs]
# yc = labels
# cv = KFold(n_splits=10, random_state=1, shuffle=True)
# algs = [
#     [DummyClassifier(random_state=0), yc, None],
#     [RandomForestClassifier(n_estimators=500, random_state=0), yc, None],
# ]
# fst = True
# for alg, y, score in algs:
#     if not fst:
#         print(end=",")
#     fst = False
#     model = alg.fit(X, y)
#     scores = cross_val_score(model, X, y, scoring=score, cv=cv, n_jobs=18)
#     print(f"{mean(scores):.2f}|{std(scores):.2f}", end=",\t")
# print()
#
#
#
# plt.figure(figsize=(10, 7))
# plt.title("Dendrogram")
# Z = shc.linkage(outcome, method='ward', metric="euclidean")
# labels = shc.fcluster(Z, 3, criterion='maxclust')
# shc.dendrogram(Z, color_threshold=50)
#
# X = df[list(all.difference(outcomes))]
# yc = labels
# cv = KFold(n_splits=10, random_state=1, shuffle=True)
# algs = [
#     [DummyClassifier(random_state=0), yc, None],
#     [RandomForestClassifier(n_estimators=500, random_state=0), yc, None],
# ]
# fst = True
# for alg, y, score in algs:
#     if not fst:
#         print(end=",")
#     fst = False
#     model = alg.fit(X, y)
#     scores = cross_val_score(model, X, y, scoring=score, cv=cv, n_jobs=18)
#     print(f"{mean(scores):.2f}|{std(scores):.2f}", end=",\t")
# print()
#
# plt.show()

print(df["ibq_reg_t1"])
df["ibq_reg_t1"].hist()
df["ibq_reg_t2"].hist()
plt.show()
"""
histograma dos desfechos 10-10   5-5

https://scholar.google.com.br/scholar?q=bayley+scale+machine+learning+classification+prediction&hl=en&as_sdt=0&as_vis=1&oi=scholart

https://www.parentprojectmd.org/wp-content/uploads/2019/06/06_Thurs_1120_Wagner.pdf
https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6242003/

https://onlinelibrary.wiley.com/doi/full/10.1111/apa.16037

Ver se Dustin fez classificação no gdrive.

"""
