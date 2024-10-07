from numpy import arange, isnan, meshgrid

from ..array import apply, normalize
from ..dictionary import merge
from .COLORBAR import COLORBAR
from .NAME_COLORSCALE import NAME_COLORSCALE
from .plot_plotly import plot_plotly


def plot_bubble_map(
    das, mac=None, si=24, colorscale=NAME_COLORSCALE["continuous"], layout=None, pr=""
):
    si1, si2 = das.shape

    if layout is None:
        layout = {}

    ti1_ = arange(si1)[::-1]

    ti2_ = arange(si2)

    layout = merge(
        {
            "height": max(480, si1 * 2 * si),
            "width": max(480, si2 * 2 * si),
            "yaxis": {
                "title": {"text", "{} (n={})".format(das.index.name, si1)},
                "tickvals": ti1_,
                "ticktext": das.index.values,
            },
            "xaxis": {
                "title": {"text", "{} (n={})".format(das.columns.name, si2)},
                "tickvals": ti2_,
                "ticktext": das.columns.values,
            },
        },
        layout,
    )

    co1_di1_di2, co2_di1_di2 = meshgrid(ti1_, ti2_, indexing="ij")

    mas = das.values

    if mac is None:
        mac = mas

    mas = apply(mas, normalize, "0-1", up=True)

    mas[isnan(mas)] = 0.5

    plot_plotly(
        [
            {
                "y": co1_di1_di2.ravel(),
                "x": co2_di1_di2.ravel(),
                "text": mas.ravel(),
                "mode": "markers",
                "marker": {
                    "size": mas.ravel() * si,
                    "color": mac.ravel(),
                    "colorscale": colorscale,
                    "colorbar": COLORBAR,
                },
            }
        ],
        layout,
        pr=pr,
    )
