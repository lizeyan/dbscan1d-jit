import os
import time
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from dbscan1d.core import DBSCAN1D


def run(size: int) -> float:
    a = np.random.randn(size)
    tic = time.perf_counter()
    DBSCAN1D(eps=0.1,).fit_predict(a)
    toc = time.perf_counter()
    return toc - tic


if __name__ == '__main__':
    data = []
    for env, size, repeat in product(
        ["0", "1"],
        [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384],
        range(10),
    ):
        os.environ["NUMBA_DISABLE_JIT"] = env
        data.append({
            "JIT_enabled": env,
            "size": size,
            "elapsed_seconds": run(size),
        })

    fig = plt.figure(figsize=(4, 2), dpi=300)
    sns.barplot(data=pd.DataFrame.from_records(data), x="size", y="elapsed_seconds", hue="JIT_enabled", log_scale=True)
    plt.show()
    plt.close()
