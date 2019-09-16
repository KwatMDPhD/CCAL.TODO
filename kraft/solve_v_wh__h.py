from pandas import DataFrame

from .solve_ax_b__x import solve_ax_b__x


def solve_v_wh__h(v, w, method="pinv"):

    print(
        "Solving V{} = W{} * H{} H...".format(
            v.shape, w.shape, (w.shape[1], v.shape[1])
        )
    )

    return DataFrame(
        solve_ax_b__x(w.values, v.values, method=method),
        index=w.columns,
        columns=v.columns,
    )
