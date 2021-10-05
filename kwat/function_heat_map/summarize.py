from re import sub

from numpy import nan

from ..dictionary import merge
from ..plot import NAME_COLORSCALE, plot_plotly
from ._make_data_annotation import _make_data_annotation
from ._make_target_annotation import _make_target_annotation
from ._process_data import _process_data
from ._process_target import _process_target
from .ANNOTATION import ANNOTATION
from .HEATMAP import HEATMAP
from .LAYOUT import LAYOUT


def summarize(ta, bu_, it=True, ac=True, ty="continuous", st=nan, layout=None, pr=""):

    n_ro = 1

    n_sp = 2

    for bu in bu_:

        n_ro += n_sp + bu["data"].shape[0]

    he = 1 / n_ro

    if layout is None:

        layout = {}

    layout = merge(
        merge(
            LAYOUT,
            {
                "height": max(640, 24 * n_ro),
                "title": {"text": "Function Heat Map Summary"},
                "annotations": _make_target_annotation(1 - he / 2, ta.name),
            },
        ),
        layout,
    )

    n_bu = len(bu_)

    yaxis = "yaxis{}".format(n_bu + 1)

    domain = [1 - he, 1]

    layout[yaxis] = {"domain": domain, "showticklabels": False}

    if it:

        for bu in bu_:

            ta = ta.loc[ta.index.intersection(bu["data"].columns)]

    if ac is not None:

        ta.sort_values(ascending=ac, inplace=True)

    co_ = ta.index.values

    heatmap = merge(HEATMAP, {"x": co_})

    tavp, mit, mat = _process_target(ta.values, ty, st)

    data = [
        merge(
            heatmap,
            {
                "yaxis": sub(r"axis", "", yaxis),
                "z": tavp.reshape([1, -1]),
                "zmin": mit,
                "zmax": mat,
                "colorscale": NAME_COLORSCALE[ty],
            },
        )
    ]

    for ie, bu in enumerate(bu_):

        fu = bu["statistic"].loc[da.index, :]

        fu.sort_values("Score", ascending=False, inplace=True)

        ro_ = fu.index.values

        yaxis = "yaxis{}".format(n_bu - ie)

        domain = [max(0, domain[0] - he * (n_sp + ro_.size)), domain[0] - he * n_sp]

        layout[yaxis] = {"domain": domain, "showticklabels": False}

        davp, mid, mad = _process_data(
            bu["data"].loc[ro_, :].reindex(labels=co_, axis=1).values, bu["type"], st
        )

        data.append(
            merge(
                heatmap,
                {
                    "yaxis": yaxis.replace("axis", ""),
                    "z": davp[::-1],
                    "y": ro_[::-1],
                    "zmin": mid,
                    "zmax": mad,
                    "colorscale": NAME_COLORSCALE[bu["type"]],
                },
            )
        )

        y = domain[1] + he / 2

        layout["annotations"].append(
            merge(
                ANNOTATION,
                {
                    "y": y,
                    "x": 0.5,
                    "xanchor": "center",
                    "text": "<b>{}</b>".format(bu["name"]),
                },
            )
        )

        layout["annotations"] += _make_data_annotation(y, ie == 0, he, ro_, fu.values)

    plot_plotly({"data": data, "layout": layout}, pr=pr)
