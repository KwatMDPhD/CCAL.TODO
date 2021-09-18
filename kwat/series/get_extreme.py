from numpy import arange
from pandas import DataFrame

from ..array import check_extreme
from ..plot import plot_point


def get_extreme(se, di, pa, size=2, **ke_ar):

    se = se.dropna().sort_values()

    sev = se.values

    la_ = se.index.values

    ex_ = check_extreme(sev, di, **ke_ar)

    lae_ = la_[ex_]

    with open("{}.txt".format(pa), mode="w") as io:

        io.write("\n".join(lae_))

    da = DataFrame(
        data={
            se.name: sev,
            "Rank": arange(sev.size),
            "Size": size,
            "Color": "#ebf6f7",
            "Opacity": 0.64,
        },
        index=la_,
    )

    da.loc[ex_, ["Size", "Color", "Opacity",]] = [
        size * 2,
        {
            "<": "#1f4788",
            ">": "#c3272b",
        }[di],
        0.8,
    ]

    plot_point(
        da,
        layout={
            "title": {
                "text": "Extreme",
            },
        },
        pa="{}.html".format(pa),
    )
