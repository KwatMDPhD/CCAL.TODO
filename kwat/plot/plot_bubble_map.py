from numpy import arange, isnan, meshgrid

from ..array import apply, normalize
from ..dictionary import merge
from .COLORBAR import COLORBAR
from .CONTINUOUS_COLORSCALE import CONTINUOUS_COLORSCALE
from .plot_plotly import plot_plotly


def plot_bubble_map(
    si_an_an, co_an_an=None, ma=24, colorscale=CONTINUOUS_COLORSCALE, layout=None, pa=""
):

    si1, si2 = si_an_an.shape

    co1_ = arange(si1)[::-1]

    co2_ = arange(si2)

    if layout is None:

        layout = {}

    layout = merge(
        {
            "height": max(480, si1 * 2 * ma),
            "width": max(480, si2 * 2 * ma),
            "yaxis": {
                "title": "{} (n={})".format(si_an_an.index.name, si1),
                "tickvals": co1_,
                "ticktext": si_an_an.index,
            },
            "xaxis": {
                "title": "{} (n={})".format(si_an_an.columns.name, si2),
                "tickvals": co2_,
                "ticktext": si_an_an.columns,
            },
        },
        layout,
    )

    si_an_an = si_an_an.values

    if co_an_an is None:

        co_an_an = si_an_an

    si2_an_an = apply(si_an_an, normalize, "0-1", up=True)

    si2_an_an[isnan(si2_an_an)] = 0.5

    co1_an_an, co2_an_an = meshgrid(co1_, co2_, indexing="ij")

    plot_plotly(
        {
            "data": [
                {
                    "y": co1_an_an.ravel(),
                    "x": co2_an_an.ravel(),
                    "text": si_an_an.ravel(),
                    "mode": "markers",
                    "marker": {
                        "size": si2_an_an.ravel() * ma,
                        "color": co_an_an.ravel(),
                        "colorscale": colorscale,
                        "colorbar": COLORBAR,
                    },
                }
            ],
            "layout": layout,
        },
        pa=pa,
    )
