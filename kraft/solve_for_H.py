from pandas import DataFrame

from .solve_ax_equal_b import solve_ax_equal_b


def solve_for_h(V, W, method="pinv"):

    print(f"Solving for H in V{V.shape} = W{W.shape} * H{(W.shape[1], V.shape[1])} ...")

    return DataFrame(
        solve_ax_equal_b(W.values, V.values, method=method),
        index=W.columns,
        columns=V.columns,
    )
