from dataclasses import dataclass

import numpy as np
from pandas import DataFrame
from scipy.stats import ttest_1samp


@dataclass
class SHAPs:
    center: int = 0

    def __post_init__(self):
        self.values = {}
        self.shaps = {}
        self.toshaps = {}  # Target-Oriented SHAP values

    def add(self, a: np.ndarray, b: np.ndarray = None, val_shap: dict = None):
        """
        >>> import numpy as np
        >>> shaps = SHAPs()
        >>> shaps.add(np.array([[90]]), np.array([[120]]), {"a": (1.7, 0.1), "b": (45.1, 0.2), "c": (13.1, -0.3)})
        >>> shaps.add(np.array([[95]]), np.array([[96]]), {"a": (2.3, 0.15), "b": (35.7, -0.02), "c": (13.5, -0.5)})
        >>> shaps.add(np.array([[101]]), np.array([[70]]), {"a": (2.1, -0.05), "b": (30.7, 0.01), "c": (12.5, -0.5)})
        >>> shaps.values_shaps_toshaps
        {'a': [(1.7, 0.1, -0.1), (2.3, 0.15, -0.15), (2.1, -0.05, -0.05)], 'b': [(45.1, 0.2, -0.2), (35.7, -0.02, 0.02), (30.7, 0.01, 0.01)], 'c': [(13.1, -0.3, 0.3), (13.5, -0.5, 0.5), (12.5, -0.5, -0.5)]}

        :param a:
        :param b:
        :param val_shap:
        :return:
        """
        for k, (v, s) in val_shap.items():
            if k not in self.values:
                self.values[k] = []
                self.shaps[k] = []
                self.toshaps[k] = []
            self.values[k].append(v)
            self.shaps[k].append(s)
            if b is None:
                self.toshaps[k].append(s if a[0, -1] >= self.center else -s)
            else:
                self.toshaps[k].append(s if a[0, -1] > b[0, -1] else -s)

    def relevance(self):
        """
        >>> import numpy as np
        >>> shaps = SHAPs()
        >>> shaps.add(np.array([[90]]), np.array([[120]]), {"a": (1.7, 0.1), "b": (45.1, 0.2), "c": (13.1, -0.3)})
        >>> shaps.add(np.array([[95]]), np.array([[96]]), {"a": (2.3, 0.15), "b": (35.7, -0.02), "c": (13.5, -0.5)})
        >>> shaps.add(np.array([[101]]), np.array([[70]]), {"a": (2.1, -0.05), "b": (30.7, 0.01), "c": (12.5, 0.5)})
        >>> shaps.relevance()  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
          variable  toshap__mean  toshap__p-value
        0        a     -0.100000         0.962910
        1        b     -0.056667         0.743855
        2        c      0.433333         0.011430

        :return:
        """
        dct = {"variable": self.values.keys(), "toshap__mean": [], "toshap__p-value": []}
        for k in self.values:
            dct["toshap__mean"].append(np.mean(self.toshaps[k]))
            dct["toshap__p-value"].append(ttest_1samp(self.toshaps[k], popmean=self.center, alternative="greater")[1])
        return DataFrame(dct)
