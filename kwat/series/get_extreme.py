from numpy import arange
from pandas import DataFrame

from ..array import check_extreme
from ..plot import plot_point


def get_extreme(se, di, pa, size=2, **ke):

    se = se.dropna().sort_values()

    nu_ = se.values

    las_ = se.index.values

    bo_ = check_extreme(nu_, di, **ke)

    la_ = las_[bo_]

    with open("{}.txt".format(pa), mode="w") as io:

        io.write("\n".join(la_))

    da = DataFrame(
        data={
            se.name: nu_,
            "Rank": arange(nu_.size),
            "Size": size,
            "Color": "#ebf6f7",
            "Opacity": 0.64,
        },
        index=las_,
    )

    da.loc[bo_, ["Size", "Color", "Opacity"]] = [
        size * 2,
        {"<": "#1f4788", ">": "#c3272b"}[di],
        0.8,
    ]

    plot_point(da, title="Extreme", pa="{}.html".format(pa))
