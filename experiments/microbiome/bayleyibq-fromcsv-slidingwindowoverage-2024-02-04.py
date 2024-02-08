import pandas as pd
from matplotlib import pyplot as plt
from pandas import read_csv, DataFrame, get_dummies, concat
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.manifold import MDS, TSNE
from sklearn.model_selection import permutation_test_score

df = read_csv("/home/davi/git/germina/results/datasetr_species2_bayley_8_t2.csv", index_col="id_estudo")
print(df.shape)
print("---------------")
df.sort_values("idade_crianca_dias_t2", inplace=True)  # age at bayley test
age_mx = df["idade_crianca_dias_t2"].max()
age_mn = df["idade_crianca_dias_t2"].min()
w_days = 4
l = []
nl, n_mn, div = [], 10, 1
for idx in df.index:
    baby = df.loc[idx]
    baby.setindex = [idx]
    age = baby["idade_crianca_dias_t2"]
    label = baby["bayley_8_t2"]
    print(age, end="\t")
    if not (age_mn + w_days < age < age_mx - w_days):  # will skip 34 extreme-age babies
        print("extreme age, skipped.")
        continue
    window = df[abs(age - df["idade_crianca_dias_t2"]) <= w_days]  # select between 36 and 192 babies
    n = window.shape[0]
    if n < n_mn:
        print("small sample, skipped.")
        continue
    mean_label = window["bayley_8_t2"].mean()
    stdev_label = window["bayley_8_t2"].std()
    if mean_label - stdev_label / div < label < mean_label + stdev_label / div:  # normal = within 68% = +- 1 stdev
        print("normal label, skipped.")
        continue
    ybool = label > mean_label
    print("----------------------")
    # baby = DataFrame([baby])
    baby["y"] = int(ybool)
    l.append(baby)
    nl.append(n)
print(max(nl), min(nl))
df = DataFrame(l)
print(df.shape)
del df["idade_crianca_dias_t2"]
del df["bayley_8_t2"]
# print(df.columns)
rf = RandomForestClassifier(n_estimators=1000)
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# t = PCA(n_components=10)
# X = MDS(n_components=70).fit_transform(X)
# X = TSNE(n_components=50, method="exact").fit_transform(X)

print(X.shape)
score, permscores, pval = permutation_test_score(rf, X=X, y=y, n_permutations=1, n_jobs=-1)
print(score, pval)
