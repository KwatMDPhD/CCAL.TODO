from re import sub

from numpy import nan

from ..dictionary import merge
from ..plot import plot_plotly
from ._make_data_annotations import _make_data_annotations
from ._make_target_annotation import _make_target_annotation
from ._process_data import _process_data
from ._process_target import _process_target
from .annotation import annotation
from .heatmap import heatmap
from .layout import layout
from .type_colorscale import type_colorscale


def summarize(
    ta,
    bu_,
    it=True,
    ac=True,
    ty="continuous",
    st=nan,
    title="Function Heat Map Summary",
    pa="",
):

    #
    if it:

        for bu in bu_:

            ta = ta.loc[ta.index.intersection(bu["data"].columns)]

    #
    if ac is not None:

        ta.sort_values(ascending=ac, inplace=True)

    #
    co_ = ta.index.values

    #
    taap, mit, mat = _process_target(ta.values, ty, st)

    #
    n_ro = 1

    n_sp = 2

    for bu in bu_:

        n_ro += bu["data"].shape[0] + n_sp

    he = 1 / n_ro

    layout = merge(
        {
            "height": max(640, 24 * n_ro),
            "title": {
                "text": title,
            },
            "annotations": _make_target_annotation(1 - he / 2, ta.name),
        },
        layout,
    )

    n_bu = len(bu_)

    yaxis = "yaxis{}".format(n_bu + 1)

    domain = [1 - he, 1]

    layout[yaxis] = {
        "domain": domain,
        "showticklabels": False,
    }

    #
    data = [
        {
            "yaxis": sub(r"axis", "", yaxis),
            "z": taap.reshape([1, -1]),
            "x": co_,
            "zmin": mit,
            "zmax": mat,
            "colorscale": type_colorscale[ty],
            **heatmap,
        }
    ]

    for ie, bu in enumerate(bu_):

        da = bu["data"].reindex(labels=co_, axis=1)

        fu = bu["statistic"].loc[da.index, :]

        fu.sort_values("Score", ascending=False, inplace=True)

        ro_ = fu.index.values

        daap, mid, mad = _process_data(da.loc[ro_, :].values, bu["type"], st)

        yaxis = "yaxis{}".format(n_bu - ie)

        domain = [
            max(0, domain[0] - he * (n_sp + daap.shape[0])),
            domain[0] - he * n_sp,
        ]

        layout[yaxis] = {
            "domain": domain,
            "showticklabels": False,
        }

        data.append(
            {
                "yaxis": yaxis.replace("axis", ""),
                "z": daap[::-1],
                "y": ro_[::-1],
                "x": co_,
                "zmin": mid,
                "zmax": mad,
                "colorscale": type_colorscale[bu["type"]],
                **heatmap,
            }
        )

        y = domain[1] + he / 2

        layout["annotations"].append(
            {
                "y": y,
                "x": 0.5,
                "xanchor": "center",
                "text": "<b>{}</b>".format(bu["name"]),
                **annotation,
            }
        )

        layout["annotations"] += _make_data_annotations(y, ie == 0, he, ro_, fu.values)

    plot_plotly(
        {
            "data": data,
            "layout": layout,
        },
        pa=pa,
    )
