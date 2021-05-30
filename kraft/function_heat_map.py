from multiprocessing import (
    Pool,
)

from numpy import (
    asarray,
    full,
    nan,
    unique,
    where,
)
from numpy.random import (
    choice,
    seed,
    shuffle,
)
from pandas import (
    DataFrame,
)

from .clustering import (
    cluster,
)
from .CONSTANT import (
    RANDOM_SEED,
    SAMPLE_FRACTION,
)
from .dictionary import (
    merge,
)
from .number___ import (
    apply_on_1,
    apply_on_2,
    check_is_extreme,
    normalize,
)
from .plot import (
    BINARY_COLORSCALE,
    CATEGORICAL_COLORSCALE,
    CONTINUOUS_COLORSCALE,
    plot_plotly,
)
from .significance import (
    get_margin_of_error,
    get_p_value_q_value,
)

HEATMAP = {
    "type": "heatmap",
    "showscale": False,
}

LAYOUT = {
    "width": 800,
    "margin": {
        "l": 200,
        "r": 200,
    },
    "title": {"x": 0.5},
}

ANNOTATION = {
    "xref": "paper",
    "yref": "paper",
    "yanchor": "middle",
    "font": {"size": 10},
    "showarrow": False,
}

TYPE_COLORSCALE = {
    "continuous": CONTINUOUS_COLORSCALE,
    "categorical": CATEGORICAL_COLORSCALE,
    "binary": BINARY_COLORSCALE,
}


def _process_target(
    ta_,
    ty,
    st,
):

    if ty == "continuous":

        if 0 < ta_.std():

            ta_ = apply_on_1(ta_, normalize, "-0-", up=True,).clip(
                -st,
                st,
            )

        return (
            ta_,
            -st,
            st,
        )

    return (
        ta_.copy(),
        None,
        None,
    )


def _process_TODO(
    nu_an_an,
    ty,
    st,
):

    nu_an_an = nu_an_an.copy()

    if ty == "continuous":

        for ie in range(nu_an_an.shape[0]):

            nu_an_an[ie] = _process_target(
                nu_an_an[ie],
                ty,
                st,
            )[0]

        return (
            nu_an_an,
            -st,
            st,
        )

    return (
        nu_an_an,
        None,
        None,
    )


def _make_target_annotations(
    la,
    y,
):

    return [
        {
            "y": y,
            "x": 0,
            "xanchor": "right",
            "text": "<b>{}</b>".format(la),
            **ANNOTATION,
        },
    ]


def _get_statistic_x(
    ie,
):

    return 1.08 + ie / 6.4


def _make_TODO_annotations(
    la_,
    nu_an_an,
    y,
    he,
    ad,
):

    annotations = []

    if ad:

        for (ie, text,) in enumerate(
            [
                "Score (\u0394)",
                "P-Value",
                "Q-Value",
            ]
        ):

            annotations.append(
                {
                    "y": y,
                    "x": _get_statistic_x(ie),
                    "xanchor": "center",
                    "text": "<b>{}</b>".format(text),
                    **ANNOTATION,
                },
            )

    y -= he

    for ie1 in range(la_.size):

        annotations.append(
            {
                "y": y,
                "x": 0,
                "xanchor": "right",
                "text": la_[ie1],
                **ANNOTATION,
            },
        )

        (
            sc,
            ma,
            pv,
            qv,
        ) = nu_an_an[ie1]

        for (ie2, text,) in enumerate(
            (
                "{:.2f} ({:.2f})".format(
                    sc,
                    ma,
                ),
                "{:.2e}".format(pv),
                "{:.2e}".format(qv),
            ),
        ):

            annotations.append(
                {
                    "y": y,
                    "x": _get_statistic_x(ie2),
                    "xanchor": "center",
                    "text": text,
                    **ANNOTATION,
                },
            )

        y -= he

    return annotations


def make(
    ta_,
    nu_an_an,
    fu,
    ac=True,
    n_jo=1,
    ra=RANDOM_SEED,
    n_sa=10,
    n_sh=10,
    pl=True,
    n_pl=8,
    ty1="continuous",
    ty2="continuous",
    st=nan,
    title="Function Heat Map",
    pa="",
):

    ta_ = ta_.loc[ta_.index.intersection(nu_an_an.columns)]

    if ac is not None:

        ta_.sort_values(
            ascending=ac,
            inplace=True,
        )

    ta2_ = ta_.to_numpy()

    la2_ = ta_.index.to_numpy()

    nu_an_an = nu_an_an.reindex(
        labels=la2_,
        axis=1,
    )

    nu2_an_an = nu_an_an.to_numpy()

    if callable(fu):

        (
            si1,
            si2,
        ) = nu2_an_an.shape

        po = Pool(n_jo)

        seed(ra)

        print("Score ({})...".format(fu.__name__))

        sc_ = asarray(
            po.starmap(
                apply_on_2,
                (
                    [
                        ta2_,
                        ro,
                        fu,
                    ]
                    for ro in nu2_an_an
                ),
            ),
        )

        if 0 < n_sa:

            print("0.95 MoE ({} sample)...".format(n_sa))

            sc_ro_sa = full(
                [
                    si1,
                    n_sa,
                ],
                nan,
            )

            n_ch = int(si2 * SAMPLE_FRACTION)

            for ie in range(n_sa):

                ie_ = choice(
                    si2,
                    size=n_ch,
                    replace=False,
                )

                ta3_ = ta2_[ie_]

                sc_ro_sa[:, ie,] = po.starmap(
                    apply_on_2,
                    (
                        [
                            ta3_,
                            ro,
                            fu,
                        ]
                        for ro in nu2_an_an[
                            :,
                            ie_,
                        ]
                    ),
                )

            ma_ = asarray(
                [
                    apply_on_1(
                        ro,
                        get_margin_of_error,
                    )
                    for ro in sc_ro_sa
                ],
            )

        else:

            ma_ = full(
                sc_.size,
                nan,
            )

        if 0 < n_sh:

            print("P-Value and Q-Value ({} shuffle)...".format(n_sh))

            sc_ro_sh = full(
                [
                    si1,
                    n_sh,
                ],
                nan,
            )

            ta3_ = ta2_.copy()

            for ie in range(n_sh):

                shuffle(ta3_)

                sc_ro_sh[:, ie,] = po.starmap(
                    apply_on_2,
                    (
                        [
                            ta3_,
                            ro,
                            fu,
                        ]
                        for ro in nu2_an_an
                    ),
                )

            (pv_, qv_,) = get_p_value_q_value(
                sc_,
                sc_ro_sh.ravel(),
                "<>",
            )

        else:

            pv_ = qv_ = full(
                sc_.size,
                nan,
            )

        po.terminate()

        fu = DataFrame(
            asarray(
                [
                    sc_,
                    ma_,
                    pv_,
                    qv_,
                ]
            ).T,
            index=nu_an_an.index,
            columns=[
                "Score",
                "MoE",
                "P-Value",
                "Q-Value",
            ],
        )

    else:

        fu = fu.loc[
            nu_an_an.index,
            :,
        ]

    fu.sort_values(
        "Score",
        ascending=False,
        inplace=True,
    )

    if pa != "":

        fu.to_csv(
            "{}statistic.tsv".format(pa),
            sep="\t",
        )

    nu_an_an = nu_an_an.loc[
        fu.index,
        :,
    ]

    if pl:

        la1_ = nu_an_an.index.to_numpy()

        fu2 = fu.to_numpy()

        if n_pl is not None and (n_pl / 2) < fu2.shape[0]:

            bo_ = check_is_extreme(
                fu2[
                    :,
                    0,
                ],
                "<>",
                n_ex=n_pl,
            )

            fu2 = fu2[bo_]

            nu2_an_an = nu2_an_an[bo_]

            la1_ = la1_[bo_]

        (ta2_, mi1, ma1,) = _process_target(
            ta2_,
            ty1,
            st,
        )

        (nu2_an_an, mi2, ma2,) = _process_TODO(
            nu2_an_an,
            ty2,
            st,
        )

        if ty1 != "continuous":

            for (it, n_it,) in zip(
                *unique(
                    ta2_,
                    return_counts=True,
                )
            ):

                if 2 < n_it:

                    print("Clustering {}...".format(it))

                    ie_ = where(ta2_ == it)[0]

                    ie2_ = ie_[cluster(nu2_an_an.T[ie_])[0]]

                    ta2_[ie_] = ta2_[ie2_]

                    nu2_an_an[:, ie_,] = nu2_an_an[
                        :,
                        ie2_,
                    ]

                    la2_[ie_] = la2_[ie2_]

        n_ro = nu2_an_an.shape[0] + 2

        he = 1 / n_ro

        layout = merge(
            {
                "height": max(
                    480,
                    24 * n_ro,
                ),
                "title": {"text": title},
                "yaxis": {
                    "domain": (
                        0,
                        1 - he * 2,
                    ),
                    "showticklabels": False,
                },
                "yaxis2": {
                    "domain": (
                        1 - he,
                        1,
                    ),
                    "showticklabels": False,
                },
                "annotations": _make_target_annotations(
                    ta_.name,
                    1 - he / 2,
                ),
            },
            LAYOUT,
        )

        layout["annotations"] += _make_TODO_annotations(
            la1_,
            fu2,
            1 - he / 2 * 3,
            he,
            True,
        )

        if pa != "":

            pa = "{}function_heat_map.html".format(pa)

        plot_plotly(
            {
                "data": [
                    {
                        "yaxis": "y2",
                        "z": ta2_.reshape(
                            [
                                1,
                                -1,
                            ]
                        ),
                        "x": la2_,
                        "zmin": mi1,
                        "zmax": ma1,
                        "colorscale": TYPE_COLORSCALE[ty1],
                        **HEATMAP,
                    },
                    {
                        "yaxis": "y",
                        "z": nu2_an_an[::-1],
                        "y": la1_[::-1],
                        "x": la2_,
                        "zmin": mi2,
                        "zmax": ma2,
                        "colorscale": TYPE_COLORSCALE[ty2],
                        **HEATMAP,
                    },
                ],
                "layout": layout,
            },
            pa=pa,
        )

    return fu


def summarize(
    ta_,
    TODO_,
    it=True,
    ac=True,
    ty1="continuous",
    st=nan,
    title="Function Heat Map Summary",
    pa="",
):

    if it:

        for TODO in TODO_:

            ta_ = ta_.loc[ta_.index.intersection(TODO["TODO"].columns)]

    if ac is not None:

        ta_.sort_values(
            ascending=ac,
            inplace=True,
        )

    ta2_ = ta_.to_numpy()

    la2_ = ta_.index.to_numpy()

    (ta2_, mi1, ma1,) = _process_target(
        ta2_,
        ty1,
        st,
    )

    n_ro = 1

    n_sp = 2

    for TODO in TODO_:

        n_ro += TODO["TODO"].shape[0] + n_sp

    he = 1 / n_ro

    layout = merge(
        {
            "height": max(
                480,
                24 * n_ro,
            ),
            "title": {"text": title},
            "annotations": _make_target_annotations(
                ta_.name,
                1 - he / 2,
            ),
        },
        LAYOUT,
    )

    n_TO = len(TODO_)

    yaxis = "yaxis{}".format(n_TO + 1)

    domain = (
        1 - he,
        1,
    )

    layout[yaxis] = {
        "domain": domain,
        "showticklabels": False,
    }

    data = [
        {
            "yaxis": yaxis.replace(
                "axis",
                "",
            ),
            "z": ta2_.reshape(
                [
                    1,
                    -1,
                ]
            ),
            "x": la2_,
            "zmin": mi1,
            "zmax": ma1,
            "colorscale": TYPE_COLORSCALE[ty1],
            **HEATMAP,
        },
    ]

    for (
        ie,
        TODO,
    ) in enumerate(TODO_):

        nu_an_an = TODO["TODO"]

        nu_an_an = nu_an_an.reindex(
            labels=la2_,
            axis=1,
        )

        fu2 = TODO["statistic"].loc[
            nu_an_an.index,
            :,
        ]

        fu2.sort_values(
            "Score",
            ascending=False,
            inplace=True,
        )

        nu_an_an = nu_an_an.loc[
            fu2.index,
            :,
        ]

        nu2_an_an = nu_an_an.to_numpy()

        la1_ = nu_an_an.index.to_numpy()

        (nu2_an_an, mi2, ma2,) = _process_TODO(
            nu2_an_an,
            TODO["data_type"],
            st,
        )

        yaxis = "yaxis{}".format(n_TO - ie)

        domain = (
            max(
                0,
                domain[0] - he * (n_sp + nu_an_an.shape[0]),
            ),
            domain[0] - he * n_sp,
        )

        layout[yaxis] = {
            "domain": domain,
            "showticklabels": False,
        }

        data.append(
            {
                "yaxis": yaxis.replace(
                    "axis",
                    "",
                ),
                "z": nu2_an_an[::-1],
                "y": la1_[::-1],
                "x": la2_,
                "zmin": mi2,
                "zmax": ma2,
                "colorscale": TYPE_COLORSCALE[TODO["data_type"]],
                **HEATMAP,
            },
        )

        y = domain[1] + he / 2

        layout["annotations"].append(
            {
                "y": y,
                "x": 0.5,
                "xanchor": "center",
                "text": "<b>{}</b>".format(TODO["name"]),
                **ANNOTATION,
            },
        )

        layout["annotations"] += _make_TODO_annotations(
            la1_,
            fu2.to_numpy(),
            y,
            he,
            ie == 0,
        )

    plot_plotly(
        {
            "data": data,
            "layout": layout,
        },
        pa=pa,
    )
