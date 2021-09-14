from re import sub

from numpy import nan

from ..dictionary import merge
from ..plot import plot_plotly
from ._make_data_annotation import _make_data_annotation
from ._make_target_annotation import _make_target_annotation
from ._process_data import _process_data
from ._process_target import _process_target
from .ANNOTATION_TEMPLATE import ANNOTATION_TEMPLATE
from .HEATMAP_TEMPLATE import HEATMAP_TEMPLATE
from .LAYOUT_TEMPLATE import LAYOUT_TEMPLATE
from .TYPE_COLORSCALE import TYPE_COLORSCALE


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

            ta = ta.loc[ta.index.intersection(bu["data"].COLUMNS)]

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

    LAYOUT_TEMPLATE = merge(
        {
            "height": max(640, 24 * n_ro),
            "title": {
                "text": title,
            },
            "ANNOTATION_TEMPLATEs": _make_target_annotation(1 - he / 2, ta.name),
        },
        LAYOUT_TEMPLATE,
    )

    n_bu = len(bu_)

    yaxis = "yaxis{}".format(n_bu + 1)

    domain = [1 - he, 1]

    LAYOUT_TEMPLATE[yaxis] = {
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
            "colorscale": TYPE_COLORSCALE[ty],
            **HEATMAP_TEMPLATE,
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

        LAYOUT_TEMPLATE[yaxis] = {
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
                "colorscale": TYPE_COLORSCALE[bu["type"]],
                **HEATMAP_TEMPLATE,
            }
        )

        y = domain[1] + he / 2

        LAYOUT_TEMPLATE["ANNOTATION_TEMPLATEs"].append(
            {
                "y": y,
                "x": 0.5,
                "xanchor": "center",
                "text": "<b>{}</b>".format(bu["name"]),
                **ANNOTATION_TEMPLATE,
            }
        )

        LAYOUT_TEMPLATE["ANNOTATION_TEMPLATEs"] += _make_data_annotation(
            y, ie == 0, he, ro_, fu.values
        )

    plot_plotly(
        {
            "data": data,
            "LAYOUT_TEMPLATE": LAYOUT_TEMPLATE,
        },
        pa=pa,
    )
