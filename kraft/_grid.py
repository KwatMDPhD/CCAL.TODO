from numpy import (
    asarray,
    diff,
    linspace,
    meshgrid,
    unique,
)

from .plot import (
    plot_heat_map,
    plot_plotly,
)


def make_g1(
    l,
    r,
    e,
    s,
):

    e *= r - l

    l -= e

    r += e

    return linspace(
        l,
        r,
        s,
    )


def reflect_g1(
    g1,
    c,
):

    h = g1.copy()

    for (
        i,
        n,
    ) in enumerate(g1):

        if n < c:

            h[i] += (c - n) * 2

        else:

            h[i] -= (n - c) * 2

    return h


def get_d(
    g1,
):

    return diff(unique(g1)).min()


def get_g1_(
    pxd,
):

    return [unique(d) for d in pxd.T]


def make_gn(
    g1_,
):

    return asarray([m.ravel() for m in meshgrid(*g1_, indexing="ij")]).T


def plot(
    gn,
    v,
    dimension_name_=None,
    value_name=None,
    file_path=None,
):

    nd = gn.shape[1]

    if dimension_name_ is None:

        dimension_name_ = ["Dimension {}".format(i) for i in range(nd)]

    g1_ = get_g1_(gn)

    v = v.reshape([g1.size for g1 in g1_])

    for (
        i,
        g1,
    ) in enumerate(g1_):

        print(
            "Grid {}: size={} min={:.2e} max={:.2e}".format(
                i,
                g1.size,
                g1.min(),
                g1.max(),
            ),
        )

    print(
        "Number: min={:.2e} max={:.2e}".format(
            v.min(),
            v.max(),
        )
    )

    if nd == 1:

        plot_plotly(
            {
                "data": [
                    {
                        "y": v,
                        "x": g1_[0],
                    }
                ],
                "layout": {
                    "yaxis": {"title": {"text": value_name}},
                    "xaxis": {"title": {"text": dimension_name_[0]}},
                },
            },
            file_path=file_path,
        )

    elif nd == 2:

        plot_heat_map(
            v,
            asarray(["{:.2e} *".format(n) for n in g1_[0]]),
            asarray(["* {:.2e}".format(n) for n in g1_[1]]),
            dimension_name_[0],
            dimension_name_[1],
            layout={"title": {"text": value_name}},
            file_path=file_path,
        )
