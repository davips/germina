from datasets import load_dataset

from germina.config import local_cache_uri, remote_cache_uri
from hdict import hdict, apply, cache, _
from hdict.dataset.dataset import df2Xy
from hdict.dataset.pandas_handling import explode_df
from shelchemy import sopen

d = hdict(dataset="scikit-learn/adult-census-income")
with sopen(remote_cache_uri) as remote, sopen(local_cache_uri) as local:
    # caches = cache(remote) >> cache(local)
    caches = cache(local)
    d >>= (
            apply(lambda dataset: load_dataset(dataset).data["train"].to_pandas()).df
            >> apply(df2Xy, _.df, "income")("X", "y")
            >> caches
    )
    d.evaluate()
    d.show()

    # y = dataset.pop("income")
    # X = dataset
    #
    # cla = RF()
    #
    # cv = cross_val_score(cla, X, y, cv=5)
    # print(cv)
