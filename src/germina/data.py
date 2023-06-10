import pandas as pd
from mdscuda import MDS
from numpy import log, ndarray
from pandas import DataFrame
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from hdict import _, apply, cache, hdict
from shelchemy import sopen, memory


def clean(targets, path, filenames, cache_uris):
    """
    files = [(microbiome_filename, None), ···, (targets_filename, ["id_estudo", "ibq_reg_t1", "ibq_reg_t2"])]
    """
    first_file = filenames[0]
    local_cache_uri, remote_cache_uri = cache_uris if len(cache_uris) == 2 else (cache_uris[0], memory)
    m = _.fromfile(path + first_file[0]).df
    m.set_index("d", inplace=True)
    d = hdict(delta=m, random_state=0, sqform=True, n_dims=484)
    with sopen(local_cache_uri) as local, sopen(remote_cache_uri) as remote:
        d >>= (
                apply(MDS)("mds") >> apply(MDS.fit, _.mds)("p")
                >> cache(remote) >> cache(local)
        )
        df_betadiv = DataFrame(d.p)
    biomeattrs = [f"m{c}" for c in range(d.n_dims)]
    df_betadiv.columns = biomeattrs
    data = [_.fromfile(path + filename).df[cols] if cols else _.fromfile(path + filename).df for filename, cols in filenames[1:]]
    dfs = [df.set_index("id_estudo") for df in data]
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
    fltr = [tgt for tgt in targets if "-" not in tgt]
    d["targets"] = df[fltr]
    for c in df.columns.values.tolist():
        if c in targets:
            del df[c]
    print("final shape after targets removal", df.shape)

    # Standardize.
    st = StandardScaler()
    s: ndarray = st.fit_transform(df)
    pca = PCA(n_components=df.shape[1])
    s = pca.fit_transform(s)
    df = DataFrame(s, columns=list(df.columns))
    print()

    # Reduce dimensionality before applying t-SNE.
    d["df"] = df
    d["n_dims"] = 100
    with sopen(local_cache_uri) as local:  # , sopen(remote_cache_uri) as remote:
        d >>= (
                apply(cdist, XA=_.df, XB=_.df, metric="sqeuclidean").delta
                >> apply(MDS).mds >> apply(MDS.fit, _.mds)("p")
                >> cache(local)
        )
        d.evaluate()
    return d
