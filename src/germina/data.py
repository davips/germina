import pandas as pd
from mdscuda import MDS
from numpy import log, ndarray
from pandas import DataFrame
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from hdict import _, apply, cache, hdict
from shelchemy import sopen, memory


def clean(targets, path, filenames, cache_uris, mds_on_first):
    """
    files = [(microbiome_filename, None), ···, (targets_filename, ["id_estudo", "ibq_reg_t1", "ibq_reg_t2"])]
    """
    first_file = filenames[0]
    local_cache_uri, remote_cache_uri = cache_uris if len(cache_uris) == 2 else (cache_uris[0], memory)

    if mds_on_first:
        m = _.fromfile(path + first_file[0]).df
        m.set_index("d", inplace=True)
        d = hdict(delta=m, random_state=0, sqform=True, n_dims=484)
        with sopen(local_cache_uri) as local, sopen(remote_cache_uri) as remote:
            d >>= (
                    apply(MDS)("mds") >> apply(MDS.fit, _.mds).betadiv
                    >> cache(remote) >> cache(local)
            )
            df_betadiv = DataFrame(d.betadiv)
        biomeattrs = [f"m{c}" for c in range(d.n_dims)]
        df_betadiv.columns = biomeattrs
        data = [_.fromfile(path + filename).df[cols] if cols else _.fromfile(path + filename).df for filename, cols in filenames[1:]]
        dfs = [df.set_index("id_estudo") for df in data]
        df: DataFrame = df_betadiv.join(dfs, how="outer")
    else:
        data = [_.fromfile(path + filename).df[cols] if cols else _.fromfile(path + filename).df for filename, cols in filenames]
        dfs = [df.set_index("id_estudo") for df in data]
        df: DataFrame = dfs[0].join(dfs[1:], how="outer")
        d = hdict(df=df, random_state=0)
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

        # Backup targets (and microbiome as temporary a workaround?).
        bkp = {tgt: df[tgt] for tgt in targets if "-" not in tgt}

        # # Remove columns.
        # s = df.isna().sum(axis=0)
        # df = df.loc[:, s.ne(s.max()) | s.eq(0)]

        # Recover targets. (and microbioma)
        for tgt, col in bkp.items():
            if tgt not in list(df.columns):
                df = pd.concat((df, col), axis=1)

        print("After removing worst columns:", df.shape)
        print()

    print("shape after NaN cleaning", df.shape)

    if "final_8_t1" in df:
        del df["final_8_t1"]
    if "final_10_t1" in df:
        del df["final_10_t1"]
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
    # df.sort_values(by="risco_class", inplace=True)
    fltr = [tgt for tgt in targets if "-" not in tgt]
    d["targets"] = df[fltr]
    for c in df.columns.values.tolist():
        if c in targets:
            del df[c]
    print("final shape after targets removal", df.shape)
    print()
    d["raw_df"] = df

    # Reduce dimensionality before applying t-SNE.
    todf = lambda X, cols: DataFrame(X, columns=cols)
    d["n_dims"] = 100
    cols = list(df.columns)
    with sopen(local_cache_uri) as local:  # , sopen(remote_cache_uri) as remote:
        d >>= apply(StandardScaler.fit_transform, StandardScaler(), _.raw_df).std >> apply(todf, _.std, cols).std_df
        d >>= apply(cdist, XA=_.std_df, XB=_.std_df, metric="sqeuclidean").delta
        d >>= apply(PCA, n_components=min(*df.shape)).pca_model >> apply(PCA.fit_transform, _.pca_model, _.std_df).pca >> apply(todf, _.pca, None).pca_df
        # d >>= apply(MDS).mds >> apply(MDS.fit, _.mds).data100 >> apply(todf, _.data100, None).data100_df
        d >>= cache(local)
        d.evaluate()

    return d
