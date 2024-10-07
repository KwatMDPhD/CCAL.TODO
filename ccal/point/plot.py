from numpy import (
    absolute,
    inf,
    isnan,
    nan,
    nan_to_num,
    nanmax,
    nanmean,
    nanmedian,
    nanmin,
    where,
)
from plotly.colors import make_colorscale

from ..array import get_not_nan_unique, guess_type
from ..dictionary import merge
from ..geometry import make_convex_hull, make_delaunay_triangulation
from ..plot import COLORBAR, NAME_COLORSCALE, get_color, plot_plotly


def plot(
    nu_no_di,
    nu_po_di,
    sh=True,
    tracen=None,
    tracep=None,
    gr_=None,
    colorscaleg=NAME_COLORSCALE["categorical"],
    co_=None,
    bap_=None,
    bag_=None,
    opacityb=0.5,
    po_sc=None,
    colorscales=NAME_COLORSCALE["continuous"],
    opacityn=0.5,
    poh_=(),
    pr="",
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
            "showlegend": False,
            "y": ti1_ + hu1_,
            "x": ti2_ + hu2_,
            "mode": "lines",
            "line": {"color": "#171412"},
        }
    )

    if tracen is None:
        tracen = {}

    marker_size = 24

    data.append(
        merge(
            {
                "name": nu_no_di.index.name,
                "y": nu_no_di.values[:, 0],
                "x": nu_no_di.values[:, 1],
                "text": nu_no_di.index.values,
                "mode": "markers",
                "marker": {
                    "size": marker_size,
                    "color": "#23191e",
                    "line": {"width": marker_size / 16, "color": "#ebf6f7"},
                    "opacity": 1,
                },
                "hoverinfo": "text",
            },
            tracen,
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
                "contours": {"coloring": "none"},
            }
        )

        grf = 1

        grl = bag_.max()

        gru_ = list(range(grf, grl + 1))

        for gr in gru_:
            data.append(
                {
                    "type": "heatmap",
                    "z": where(bag_ == gr, bap_, nan),
                    "y": co_,
                    "x": co_,
                    "colorscale": make_colorscale(
                        ["rgb(255, 255, 255)", get_color(colorscaleg, gr, [grf, grl])]
                    ),
                    "opacity": opacityb,
                    "showscale": False,
                    "hoverinfo": "none",
                }
            )

    if tracep is None:
        tracep = {}

    marker_size = 16

    opacityp = 0.88

    tracep = merge(
        {
            "name": nu_po_di.index.name,
            "mode": "markers",
            "marker": {
                "size": marker_size,
                "color": "#20d9ba",
                "line": {"width": marker_size / 16, "color": "#898a74"},
                "opacity": opacityp,
            },
            "hoverinfo": "text",
        },
        tracep,
    )

    if gr_ is not None:
        for gr in gru_:
            name = "Group {}".format(gr)

            nug_po_di = nu_po_di.loc[gr_ == gr, :]

            data.append(
                merge(
                    tracep,
                    {
                        "legendgroup": name,
                        "name": name,
                        "y": nug_po_di.values[:, 0],
                        "x": nug_po_di.values[:, 1],
                        "text": nug_po_di.index.values,
                        "marker": {"color": get_color(colorscaleg, gr, [grf, grl])},
                    },
                )
            )

    elif po_sc is not None:
        sc_ = po_sc.reindex(index=nu_po_di.index).values

        ie_ = nan_to_num(absolute(sc_), nan=-inf).argsort()

        sc_ = sc_[ie_]

        nu_po_di = nu_po_di.iloc[ie_, :]

        ty = guess_type(sc_)

        if ty == "continuous":
            tickvals = [nanmin(sc_), nanmedian(sc_), nanmean(sc_), nanmax(sc_)]

            ticktext = ["{:.2e}".format(ti) for ti in tickvals]

        else:
            tickvals = get_not_nan_unique(sc_)

            ticktext = tickvals

        if ty == "binary":
            colorbar = None

        else:
            colorbar = merge(
                COLORBAR,
                {"tickmode": "array", "tickvals": tickvals, "ticktext": ticktext},
            )

        data.append(
            merge(
                tracep,
                {
                    "y": nu_po_di.values[:, 0],
                    "x": nu_po_di.values[:, 1],
                    "text": nu_po_di.index.values,
                    "marker": {
                        "color": sc_,
                        "colorscale": colorscales,
                        "colorbar": colorbar,
                        "opacity": where(isnan(sc_), opacityn, opacityp),
                    },
                },
            )
        )

    else:
        data.append(
            merge(
                tracep,
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

    plot_plotly(data, layout, pr=pr)
