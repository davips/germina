from statistics import correlation

import numpy as np
import seaborn as sns
from lange import ap
from matplotlib import pyplot as plt
from pandas import read_csv, DataFrame
from scipy.stats import rankdata, kendalltau
from sklearn.decomposition import PCA
from sklearn.manifold import MDS

from sortedness.embedding.sortedness_ import balanced_embedding

targets_df = read_csv(f"data/nathalia260224-targets.csv", index_col="id_estudo")
alltargets = [
    'bayley_3_t1', 'bayley_3_t2', 'bayley_3_t3', 'bayley_3_t4',
    'bayley_8_t1', 'bayley_8_t2', 'bayley_8_t3', 'bayley_8_t4',
    'ibq_soot_t1', 'ibq_soot_t2', 'ibq_soot_t3', 'ibq_dura_t1',
    'ibq_dura_t2', 'ibq_dura_t3',
    # 'ecbq_atf_t4', 'ecbq_ats_t4', 'ecbq_inh_t4', 'ecbq_sth_t4',
    # 'ecbq_effco_t4'
    # 'ibq_reg_t1', 'ibq_reg_t2', 'ibq_reg_t3',
]
targets = [tgt for tgt in alltargets if tgt[-1] == "2"]
Y_df = targets_df[targets]
Y_df = Y_df.dropna(axis="rows")
m0 = Y_df.to_numpy()
m = np.vstack([m0[:-1], [[0] * m0.shape[1]]]) - np.vstack([m0[1:], [[0] * m0.shape[1]]])
c = lambda x: abs(m[:, x]) / m0[:, x] > 0.1
l = ap[0, 1, ..., m.shape[1] - 1]
m = m0[c(0) & c(1) & c(2) & c(3)]
# m = DataFrame(m).dropna(axis="rows").to_numpy()

# m = rankdata(m, axis=0, method="average")
print(m)
print(m.shape)
# p = balanced_embedding(m, epochs=20)
p = MDS(n_components=2).fit_transform(m)
# p = PCA(n_components=2).fit_transform(m)
# for i in ap[0, 1, ..., m.shape[1] - 1]:
#     for j in ap[i, i + 1, ..., m.shape[1] - 1]:
#         print(kendalltau(m[:, i], m[:, j])[0], correlation(m[:, i], m[:, j]))
print(kendalltau(p[:, 0], p[:, 1])[0], correlation(p[:, 0], p[:, 1]))
pdf = DataFrame(p)
corr = pdf.corr()
sns.heatmap(corr,
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)
# plt.show()

sns.pairplot(pdf)
Y_df = pdf
s = 100 * (0.01 + Y_df.iloc[:, 1] - Y_df.iloc[:, 1].min()) / (Y_df.iloc[:, 1].max() - Y_df.iloc[:, 1].min())
DataFrame(p, columns=["x", "y"]).plot.scatter("x", "y", c=Y_df.iloc[:, 0], s=s)
plt.show()
