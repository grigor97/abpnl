from __future__ import annotations

from typing import TypeAlias, Any

import pandas as pd
from numpy.typing import NDArray

from train import AbPNLTrainer
from synthproblems import PostNonlinearNToOne, generate_samples

from generate_data import *
from utils import *

T: TypeAlias = NDArray[np.floating[Any]]


def run_sample(n, d, name_noise, name_h):
    A, x = simulate_mult_pnl_erdos_renyi(n, d, name_noise, name_h)
    k = int(0.5*n)
    x_train = x[:k]
    x_test = x[k:]

    params = AbPNLTrainer.default_params
    params["logdir"] = "results/eg1"
    params["max_workers"] = 15

    abpnl = AbPNLTrainer(params)
    abpnl.doit(x_train, x_test)

    co = abpnl.causal_order

    print(co[::-1])
    print(A)
    wrong = compute_wrong_orders(co[::-1], A)
    print(wrong)

    return wrong


if __name__ == "__main__":
    n = 100
    d = 3
    num_datasets = 2
    name_noise = "gaussian"
    name_h = "cube"
    wrongs = []
    for j in range(num_datasets):
        wrong = run_sample(n, d, name_noise, name_h)
        wrongs.append(wrong)

    print(wrongs)

    df = pd.Series(wrongs)
    file_name = "../abpnl_results_" + name_noise + "_" + name_h + "_" + "abpnl" + str(n) + "_" + str(d) + str(num_datasets) + ".csv"
    df.to_csv(file_name, index=False)
