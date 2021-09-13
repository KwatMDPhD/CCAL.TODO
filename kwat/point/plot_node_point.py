from numpy import absolute, integer, isnan, median, nan, where
from plotly.colors import make_colorscale

from ..array import get_not_nan_unique
from ..dictionary import merge
from ..geometry import make_convex_hull, make_delaunay_triangulation
from ..plot import (
    categorical_colorscale,
    colorbar,
    continuous_colorscale,
    get_color,
    plot_plotly,
)


def plot_node_point(
    nu_no_di,
    nu_po_di,
    non,
    no_,
    pon,
    po_,
    sh=True,
    notrace=None,
    potrace=None,
    gr_=None,
    grc=categorical_colorscale,
    co_=None,
    bap_=None,
    bag_=None,
    scn=None,
    sc_=None,
    scc=continuous_colorscale,
    scopacity=0.8,
    scnopacity=0.08,
    poh_=(),
    pa="",
):

    title = "{} {} and {} {}".format(no_.size, non, po_.size, pon)

    if scn is not None:

        title = "{}<br>{}".format(title, scn)

    axis = {
        "showgrid": False,
        "showline": False,
        "zeroline": False,
        "showticklabels": False,
    }

    layout = {
        "height": 880,
        "width": 880,
        "title": {
            "x": 0.5,
            "text": "<b>{}</b>".format(title),
            "font": {
                "size": 24,
                "color": "#4e40d8",
                "family": "Times New Roman, sans-serif",
            },
        },
        "yaxis": axis,
        "xaxis": axis,
        "annotations": [],
    }

    data = []

    ti1_, ti2_ = make_delaunay_triangulation(nu_no_di)

    hu1_, hu2_ = make_convex_hull(nu_no_di)

    data.append(
        {
            "legendgroup": "Node",
            "name": "Line",
            "y": ti1_ + hu1_,
            "x": ti2_ + hu2_,
            "mode": "lines",
            "line": {
                "color": "#171412",
            },
        }
    )

    if notrace is None:

        notrace = {}

    data.append(
        merge(
            {
                "legendgroup": "Node",
                "name": non,
                "y": nu_no_di[:, 0],
                "x": nu_no_di[:, 1],
                "text": no_,
                "mode": "markers",
                "marker": {
                    "size": 20,
                    "color": "#23191e",
                    "line": {
                        "width": 1,
                        "color": "#ebf6f7",
                    },
                },
                "hoverinfo": "text",
            },
            notrace,
        )
    )

    if sh:

        arrowwidth = 1.6

        arrowcolor = "#ebf6f7"

        layout["annotations"] += [
            {
                "y": co0,
                "x": co1,
                "text": "<b>{}</b>".format(no),
                "font": {
                    "size": 16,
                    "color": "#23191e",
                    "family": "Gravitas One, monospace",
                },
                "borderpad": 2,
                "borderwidth": arrowwidth,
                "bordercolor": arrowcolor,
                "bgcolor": "#ffffff",
                "arrowwidth": arrowwidth,
                "arrowcolor": arrowcolor,
            }
            for no, (co0, co1) in zip(no_, nu_no_di)
        ]

    if bag_ is not None:

        data.append(
            {
                "type": "contour",
                "showlegend": False,
                "z": bap_,
                "y": co_,
                "x": co_,
                "autocontour": False,
                "ncontours": 24,
                "contours": {
                    "coloring": "none",
                },
            }
        )

        n_gr = int(get_not_nan_unique(bag_).max() + 1)

        gr_co = {gr: get_color(grc, gr / max(1, n_gr - 1)) for gr in range(n_gr)}

        for gr in range(n_gr):

            data.append(
                {
                    "type": "heatmap",
                    "z": where(bag_ == gr, bap_, nan),
                    "y": co_,
                    "x": co_,
                    "colorscale": make_colorscale(["rgb(255, 255, 255)", gr_co[gr]]),
                    "showscale": False,
                    "hoverinfo": "none",
                }
            )

    if potrace is None:

        potrace = {}

    potrace = merge(
        {
            "name": pon,
            "mode": "markers",
            "marker": {
                "size": 16,
                "color": "#20d9ba",
                "line": {
                    "width": 0.8,
                    "color": "#000000",
                },
            },
            "hoverinfo": "text",
        },
        potrace,
    )

    if sc_ is not None:

        ie_ = absolute(sc_).argsort()

        sc_ = sc_[ie_]

        po_ = po_[ie_]

        nu_po_di = nu_po_di[ie_]

        scn_ = get_not_nan_unique(sc_)

        if all(isinstance(sc, integer) for sc in scn_):

            tickvals = scn_

            ticktext = tickvals

        else:

            tickvals = [
                scn_.min(),
                median(scn_),
                scn_.mean(),
                scn_.max(),
            ]

            ticktext = ["{:.2e}".format(nu) for nu in tickvals]

        data.append(
            merge(
                potrace,
                {
                    "y": nu_po_di[:, 0],
                    "x": nu_po_di[:, 1],
                    "text": po_,
                    "marker": {
                        "color": sc_,
                        "colorscale": scc,
                        "colorbar": {
                            **colorbar,
                            "tickmode": "array",
                            "tickvals": tickvals,
                            "ticktext": ticktext,
                        },
                        "opacity": where(isnan(sc_), scnopacity, scopacity),
                    },
                },
            )
        )

    elif gr_ is not None:

        for gr in range(n_gr):

            name = "Group {}".format(gr)

            is_ = gr_ == gr

            data.append(
                merge(
                    potrace,
                    {
                        "legendgroup": name,
                        "name": name,
                        "y": nu_po_di[is_, 0],
                        "x": nu_po_di[is_, 1],
                        "text": po_[is_],
                        "marker": {
                            "color": gr_co[gr],
                        },
                    },
                )
            )

    else:

        data.append(
            merge(
                potrace,
                {
                    "y": nu_po_di[:, 0],
                    "x": nu_po_di[:, 1],
                    "text": po_,
                },
            )
        )

    for po in poh_:

        co0, co1 = nu_po_di[po_ == po][0]

        layout["annotations"].append(
            {
                "y": co0,
                "x": co1,
                "text": "<b>{}</b>".format(po),
                "arrowhead": 2,
                "arrowwidth": 2,
                "arrowcolor": "#c93756",
                "standoff": None,
                "clicktoshow": "onoff",
            }
        )

    plot_plotly(
        {
            "data": data,
            "layout": layout,
        },
        pa=pa,
    )
