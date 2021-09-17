from numpy import absolute, integer, isnan, nan, where
from plotly.colors import make_colorscale

from ..array import get_not_nan_unique
from ..dictionary import merge
from ..geometry import make_convex_hull, make_delaunay_triangulation
from ..plot import COLORBAR_TEMPLATE, CONTINUOUS_COLORSCALE, GROUP_COLOR, plot_plotly


def plot(
    nu_no_di,
    nu_po_di,
    sh=True,
    ntrace=None,
    ptrace=None,
    gr_=None,
    gr_co=GROUP_COLOR,
    co_=None,
    bap_=None,
    bag_=None,
    po_sc=None,
    scolorscale=CONTINUOUS_COLORSCALE,
    sopacity=0.8,
    snopacity=0.08,
    poh_=(),
    pa="",
):

    title = "{} {} and {} {}".format(
        nu_no_di.shape[0], nu_no_di.index.name, nu_po_di.shape[0], nu_po_di.index.name
    )

    if po_sc is not None:

        title = "{}<br>{}".format(title, po_sc.name)

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

    if ntrace is None:

        ntrace = {}

    data.append(
        merge(
            {
                "legendgroup": "Node",
                "name": nu_no_di.index.name,
                "y": nu_no_di.values[:, 0],
                "x": nu_no_di.values[:, 1],
                "text": nu_no_di.index.values,
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
            ntrace,
        )
    )

    if sh:

        arrowwidth = 1.6

        arrowcolor = "#ebf6f7"

        layout["annotations"] += [
            {
                "y": co1,
                "x": co2,
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
            for no, (co1, co2) in nu_no_di.iterrows()
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

        gru_ = get_not_nan_unique(bag_)

        n_gr = gru_.size

        for gr in gru_:

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

    if ptrace is None:

        ptrace = {}

    ptrace = merge(
        {
            "name": nu_po_di.index.name,
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
        ptrace,
    )

    if gr_ is not None:

        for gr in range(gru_):

            name = "Group {}".format(gr)

            nu_po_di = nu_po_di.loc[gr_ == gr, :]

            data.append(
                merge(
                    ptrace,
                    {
                        "legendgroup": name,
                        "name": name,
                        "y": nu_po_di.values[:, 0],
                        "x": nu_po_di.values[:, 1],
                        "text": nu_po_di.index.values,
                        "marker": {
                            "color": gr_co[gr],
                        },
                    },
                )
            )

    elif po_sc is not None:

        sc_ = po_sc.values

        ie_ = absolute(sc_).argsort()

        sc_ = sc_[ie_]

        nu_po_di = nu_po_di.iloc[ie_, :]

        scn_ = get_not_nan_unique(sc_)

        if all(isinstance(sc, integer) for sc in scn_):

            tickvals = scn_

            ticktext = tickvals

        else:

            tickvals = [
                sc_.min(),
                sc_.median(),
                sc_.mean(),
                sc_.max(),
            ]

            ticktext = ["{:.2e}".format(ti) for ti in tickvals]

        data.append(
            merge(
                ptrace,
                {
                    "y": nu_po_di.values[:, 0],
                    "x": nu_po_di.values[:, 1],
                    "text": nu_po_di.index.values,
                    "marker": {
                        "color": sc_,
                        "colorscale": scolorscale,
                        "colorbar": {
                            **COLORBAR_TEMPLATE,
                            "tickmode": "array",
                            "tickvals": tickvals,
                            "ticktext": ticktext,
                        },
                        "opacity": where(isnan(sc_), snopacity, sopacity),
                    },
                },
            )
        )

    else:

        data.append(
            merge(
                ptrace,
                {
                    "y": nu_po_di.values[:, 0],
                    "x": nu_po_di.values[:, 1],
                    "text": nu_po_di.index.values,
                },
            )
        )

    for po in poh_:

        co1, co2 = nu_po_di.loc[po, :]

        layout["annotations"].append(
            {
                "y": co1,
                "x": co2,
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
