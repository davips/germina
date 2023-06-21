aimport pandas as pd
from numpy import mean, absolute, std
from sklearn.dummy import DummyRegressor, DummyClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, cross_val_score

from hdict import hdict
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
from numpy import log, ndarray
from pandas import DataFrame
from sklearn.decomposition import PCA
from sklearn.manifold import smacof
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from hdict import _

path = "/home/davi/git/germina/data/metadata___2023-05-08-fup5afixed.csv"
attrs = ["delivery_mode", "EBF_3m", "renda_familiar_total_t0", "maternal_ethinicity", "infant_ethinicity", "antibiotic", "ahmed_c3_t1"]
targets = ["ibq_reg_t1", "ibq_reg_t2"]

d = _.fromfile(path)
df: DataFrame = d.df
df.set_index("id_estudo", inplace=True)
del df["e01_t1"]
df["antibiotic"] = df["antibiotic"] == "yes"
df["EBF_3m"] = df["EBF_3m"] == "EBF"
df["renda_familiar_total_t0"] = log(df["renda_familiar_total_t0"])

df = df[attrs+targets]
# df = df.loc[ids]
print("df:", df.shape)
print(df)
print()
print()
print()
nans = df.isna().sum(axis=1)
print(nans.sort_values().to_string())

print("Drop NaN rows")
print(df)
df.dropna(inplace=True)
print(df)

print(df.to_string())

from sklearn.decomposition import PCA
colors = df["EBF_3m"]
# colors = df["infant_ethinicity"]
# colors = df.pop("ibq_reg_t1")
# colors = df.pop("ibq_reg_t2")
# del df["maternal_ethinicity"]
st = StandardScaler()
s: ndarray = st.fit_transform(df)
pca = PCA(n_components=5)
p = pca.fit_transform(s)

explained = DataFrame(pca.components_, columns=list(df.columns))
print(explained[explained >= 0.02].transpose().to_string())

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.scatter(p[:, 0], p[:, 1], p[:, 2], c=colors, cmap="coolwarm")
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(p[:, 0], p[:, 1], c=colors, cmap="coolwarm")
plt.show()


exit()


# m = DataFrame(s).transpose().cov()
m = df.transpose().cov()
p: ndarray = smacof(m.fillna(0), metric=not True, n_components=3, random_state=0)[0]
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.scatter(p[:, 0], p[:, 1], p[:, 2], c=colors, cmap="coolwarm")
plt.show()



print(
    "Outcome,         "
    "     BaselineC, "
    "            RF, "
    "     BaselineR, "
    # "          CART, "
    "  RandomForest, "
    # "       XGBoost, "
    # "          LGBM, "
    # "      CatBoost"
)
for target in targets:
    X = df[attrs]
    yc = df[target] > mean(df[target])
    yc.replace(True, "high", inplace=True)
    yc.replace(False, "low", inplace=True)
    yr = df[target]
    # print(",".join(str(x) for x in yr))

    cv = KFold(n_splits=10, random_state=1, shuffle=True)

    algs = [
        [DummyClassifier(random_state=0), yc, None],
        [RandomForestClassifier(n_estimators=500), yc, None],
        [DummyRegressor(strategy="median"), yr, "r2"],
        # [DecisionTreeRegressor(max_depth=3, min_samples_split=4, min_samples_leaf=3, random_state=0, max_leaf_nodes=20), yr, "r2"],
        [RandomForestRegressor(n_estimators=500), yr, "r2"],
        # [XGBRegressor(), yr, "r2"],
        # [LGBMRegressor(), yr, "r2"],
        # [CatBoostRegressor(verbose=False), yr, "r2"]
    ]
    print(f"{target + ',':16}", end="")
    fst = True
    for alg, y, score in algs:
        if not fst:
            print(end=",")
        fst = False
        model = alg.fit(X, y)
        scores = cross_val_score(model, X, y, scoring=score, cv=cv, n_jobs=-1)
        # print(f"{mean(scores):15.2f}", end="")
        print(f"{mean(scores):.2f}|{std(scores):.2f}", end=",\t")
    print()
