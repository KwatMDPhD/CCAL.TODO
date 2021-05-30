from numpy import (
    arange,
    argsort,
    isnan,
    meshgrid,
    nonzero,
    unique,
)
from plotly.colors import (
    convert_colors_to_same_type,
    find_intermediate_color,
    make_colorscale,
    qualitative,
)
from plotly.io import (
    show,
    write_html,
)

from .dictionary import (
    merge,
)
from .number___ import (
    apply_on_1,
    normalize,
)

CONTINUOUS_COLORSCALE = make_colorscale(
    (
        "#0000ff",
        "#ffffff",
        "#ff0000",
    )
)

CATEGORICAL_COLORSCALE = make_colorscale(qualitative.Plotly)

BINARY_COLORSCALE = make_colorscale(
    (
        "#006442",
        "#ffffff",
        "#ffa400",
    )
)

COLORBAR = {
    "thicknessmode": "fraction",
    "thickness": 0.024,
    "len": 0.64,
    "ticks": "outside",
    "tickfont": {"size": 10},
}


def plot_plotly(
    figure,
    pa="",
):

    figure = merge(
        {
            "layout": {
                "autosize": False,
                "template": "plotly_white",
            }
        },
        figure,
    )

    co = {"editable": True}

    show(
        figure,
        config=co,
    )

    if pa != "":

        write_html(
            figure,
            pa,
            config=co,
        )


def get_color(
    colorscale,
    fr,
):

    for ie in range(len(colorscale) - 1):

        (
            fr1,
            co1,
        ) = colorscale[ie]

        (
            fr2,
            co2,
        ) = colorscale[ie + 1]

        if fr1 <= fr <= fr2:

            co = find_intermediate_color(
                *convert_colors_to_same_type(
                    [
                        co1,
                        co2,
                    ]
                )[0],
                (fr - fr1) / (fr2 - fr1),
                colortype="rgb",
            )

            return "rgb({},{},{})".format(
                *(
                    int(float(it))
                    for it in co[4:-1].split(
                        ",",
                        2,
                    )
                ),
            )


def plot_point(
    an_po_pa,
    annotation_font_size=16,
    title="",
    pa="",
):

    data = [
        {
            "name": "Point",
            "y": an_po_pa.iloc[
                :,
                0,
            ],
            "x": an_po_pa.iloc[
                :,
                1,
            ],
            "text": an_po_pa.index,
            "mode": "markers",
            "marker": {
                "size": an_po_pa["Size"],
                "color": an_po_pa["Color"],
                "opacity": an_po_pa["Opacity"],
                "line": {"width": 0},
            },
        },
    ]

    annotations = []

    for (text, (y, x,),) in (
        an_po_pa.iloc[
            :,
            :2,
        ]
        .loc[an_po_pa["Annotate"]]
        .iterrows()
    ):

        annotations.append(
            {
                "y": y,
                "x": x,
                "text": text,
                "font": {"size": annotation_font_size},
                "arrowhead": 2,
                "arrowsize": 1,
                "clicktoshow": "onoff",
            },
        )

    plot_plotly(
        {
            "data": data,
            "layout": {
                "title": title,
                "yaxis": {"title": an_po_pa.columns[0]},
                "xaxis": {"title": an_po_pa.columns[1]},
                "annotations": annotations,
            },
        },
        pa=pa,
    )


def _get_center_index(
    gr_,
    gr,
):

    (ie1, ie2,) = nonzero(gr_ == gr)[0][
        [
            0,
            -1,
        ]
    ]

    return ie1 + (ie2 - ie1) / 2


def plot_heat_map(
    nu_an_an,
    colorscale=CONTINUOUS_COLORSCALE,
    gr1_=(),
    gr2_=(),
    colorscale1=CATEGORICAL_COLORSCALE,
    colorscale2=CATEGORICAL_COLORSCALE,
    gr1_la=None,
    gr2_la=None,
    layout=None,
    annotation1=None,
    annotation2=None,
    pa="",
):

    if 0 < len(gr1_):

        ie_ = argsort(gr1_)

        gr1_ = gr1_[ie_]

        nu_an_an = nu_an_an.iloc[
            ie_,
            :,
        ]

    if 0 < len(gr2_):

        ie_ = argsort(gr2_)

        gr2_ = gr2_[ie_]

        nu_an_an = nu_an_an.iloc[
            :,
            ie_,
        ]

    domain = (
        0,
        0.95,
    )

    if layout is None:

        layout = {}

    axis = {
        "domain": (
            0.96,
            1,
        ),
        "showgrid": False,
        "showline": False,
        "zeroline": False,
        "showticklabels": False,
    }

    layout = merge(
        {
            "yaxis": {
                "title": "{} (n={})".format(
                    nu_an_an.index.name,
                    nu_an_an.shape[0],
                ),
                "domain": domain,
            },
            "xaxis": {
                "title": "{} (n={})".format(
                    nu_an_an.columns.name,
                    nu_an_an.shape[1],
                ),
                "domain": domain,
            },
            "yaxis2": axis,
            "xaxis2": axis,
            "annotations": [],
        },
        layout,
    )

    colorbar_x = 1.04

    data = [
        {
            "type": "heatmap",
            "z": nu_an_an.to_numpy()[::-1],
            "y": nu_an_an.index.to_numpy()[::-1],
            "x": nu_an_an.columns.to_numpy(),
            "colorscale": colorscale,
            "colorbar": {
                **COLORBAR,
                "x": colorbar_x,
            },
        },
    ]

    if 0 < len(gr1_):

        gr1_ = gr1_[::-1]

        colorbar_x += 0.1

        data.append(
            {
                "xaxis": "x2",
                "type": "heatmap",
                "z": gr1_.reshape(
                    [
                        -1,
                        1,
                    ]
                ),
                "colorscale": colorscale1,
                "colorbar": {
                    **COLORBAR,
                    "x": colorbar_x,
                    "dtick": 1,
                },
                "hoverinfo": "z+y",
            },
        )

        if gr1_la is not None:

            if annotation1 is None:

                annotation1 = {}

            layout["annotations"] += [
                merge(
                    {
                        "xref": "x2",
                        "x": 0,
                        "xanchor": "left",
                        "showarrow": False,
                        "y": _get_center_index(
                            gr1_,
                            gr,
                        ),
                        "text": gr1_la[gr],
                    },
                    annotation1,
                )
                for gr in unique(gr1_)
            ]

    if 0 < len(gr2_):

        colorbar_x += 0.1

        data.append(
            {
                "yaxis": "y2",
                "type": "heatmap",
                "z": gr2_.reshape(
                    [
                        1,
                        -1,
                    ]
                ),
                "colorscale": colorscale2,
                "colorbar": {
                    **COLORBAR,
                    "x": colorbar_x,
                    "dtick": 1,
                },
                "hoverinfo": "z+x",
            },
        )

        if gr2_la is not None:

            if annotation2 is None:

                annotation2 = {}

            layout["annotations"] += [
                merge(
                    {
                        "yref": "y2",
                        "y": 0,
                        "yanchor": "bottom",
                        "textangle": -90,
                        "showarrow": False,
                        "x": _get_center_index(
                            gr2_,
                            gr,
                        ),
                        "text": gr2_la[gr],
                    },
                    annotation2,
                )
                for gr in unique(gr2_)
            ]

    plot_plotly(
        {
            "data": data,
            "layout": layout,
        },
        pa=pa,
    )


def plot_bubble_map(
    si_an_an,
    co_an_an=None,
    ma=24,
    colorscale=CONTINUOUS_COLORSCALE,
    layout=None,
    pa="",
):

    (
        si1,
        si2,
    ) = si_an_an.shape

    co1_ = arange(si1)[::-1]

    co2_ = arange(si2)

    if layout is None:

        layout = {}

    layout = merge(
        {
            "height": max(
                480,
                si1 * 2 * ma,
            ),
            "width": max(
                480,
                si2 * 2 * ma,
            ),
            "yaxis": {
                "title": "{} (n={})".format(
                    si_an_an.index.name,
                    si1,
                ),
                "tickvals": co1_,
                "ticktext": si_an_an.index,
            },
            "xaxis": {
                "title": "{} (n={})".format(
                    si_an_an.columns.name,
                    si2,
                ),
                "tickvals": co2_,
                "ticktext": si_an_an.columns,
            },
        },
        layout,
    )

    si_an_an = si_an_an.to_numpy()

    if co_an_an is None:

        co_an_an = si_an_an

    si2_an_an = apply_on_1(
        si_an_an,
        normalize,
        "0-1",
        up=True,
    )

    si2_an_an[isnan(si2_an_an)] = 0.5

    (co1_an_an, co2_an_an,) = meshgrid(
        co1_,
        co2_,
        indexing="ij",
    )

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
                },
            ],
            "layout": layout,
        },
        pa=pa,
    )


def plot_histogram(
    nu__,
    no=None,
    xbins_size=None,
    colorscale=CATEGORICAL_COLORSCALE,
    layout=None,
    pa="",
):

    ru = all(nu_.size <= 1e3 for nu_ in nu__)

    n_tr = len(nu__)

    if ru:

        he = 0.04

        ma = n_tr * he

        mi = ma + he

    else:

        ma = 0

        mi = 0

    if no is None:

        yaxis2_title_text = "N"

    else:

        yaxis2_title_text = no.title()

    if layout is None:

        layout = {}

    layout = merge(
        {
            "xaxis": {"anchor": "y"},
            "yaxis": {
                "domain": (
                    0,
                    ma,
                ),
                "zeroline": False,
                "dtick": 1,
                "showticklabels": False,
            },
            "yaxis2": {
                "domain": (
                    mi,
                    1,
                ),
                "title": {"text": yaxis2_title_text},
            },
        },
        layout,
    )

    data = []

    for (
        ie,
        nu_,
    ) in enumerate(nu__):

        co = get_color(
            colorscale,
            ie
            / max(
                1,
                (n_tr - 1),
            ),
        )

        trace = {
            "legendgroup": ie,
            "name": nu_.name,
            "x": nu_.to_numpy(),
        }

        data.append(
            {
                "yaxis": "y2",
                "type": "histogram",
                "histnorm": no,
                "xbins": {"size": xbins_size},
                "marker": {"color": co},
                **trace,
            },
        )

        if ru:

            data.append(
                {
                    "showlegend": False,
                    "y": [ie] * nu_.size,
                    "text": nu_.index,
                    "mode": "markers",
                    "marker": {
                        "symbol": "line-ns-open",
                        "color": co,
                    },
                    "hoverinfo": "x+text",
                    **trace,
                },
            )

    plot_plotly(
        {
            "data": data,
            "layout": layout,
        },
        pa=pa,
    )
