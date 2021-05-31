from multiprocessing import Pool

from numpy import asarray, full, nan, unique, where
from numpy.random import choice, seed, shuffle
from pandas import DataFrame

from .clustering import cluster
from .CONSTANT import RANDOM_SEED, SAMPLE_FRACTION
from .dictionary import merge
from .number___ import apply_on_1, apply_on_2, check_is_extreme, normalize
from .plot import (
    BINARY_COLORSCALE,
    CATEGORICAL_COLORSCALE,
    CONTINUOUS_COLORSCALE,
    plot_plotly,
)
from .significance import get_margin_of_error, get_p_value_q_value

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
    "title": {
        "x": 0.5,
    },
}

ANNOTATION = {
    "xref": "paper",
    "yref": "paper",
    "yanchor": "middle",
    "font": {
        "size": 10,
    },
    "showarrow": False,
}

TYPE_COLORSCALE = {
    "continuous": CONTINUOUS_COLORSCALE,
    "categorical": CATEGORICAL_COLORSCALE,
    "binary": BINARY_COLORSCALE,
}


def _process_target(ta, ty, st):

    if ty == "continuous":

        if 0 < ta.std():

            ta = apply_on_1(ta, normalize, "-0-", up=True).clip(-st, st)

        return ta, -st, st

    return ta.copy(), None, None


def _process_data(da, ty, st):

    da = da.copy()

    if ty == "continuous":

        for ie in range(da.shape[0]):

            da[ie] = _process_target(da[ie], ty, st)[0]

        return da, -st, st

    return da, None, None


def _make_target_annotation(y, text):

    return [
        {
            "y": y,
            "x": 0,
            "xanchor": "right",
            "text": "<b>{}</b>".format(text),
            **ANNOTATION,
        }
    ]


def _get_statistic_x(ie):

    return 1.08 + ie / 6.4


def _make_data_annotations(y, ad, he, text_, fu):

    annotations = []

    if ad:

        for ie, text in enumerate(["Score (\u0394)", "P-Value", "Q-Value"]):

            annotations.append(
                {
                    "y": y,
                    "x": _get_statistic_x(ie),
                    "xanchor": "center",
                    "text": "<b>{}</b>".format(text),
                    **ANNOTATION,
                }
            )

    y -= he

    for ie1 in range(text_.size):

        annotations.append(
            {
                "y": y,
                "x": 0,
                "xanchor": "right",
                "text": text_[ie1],
                **ANNOTATION,
            }
        )

        sc, ma, pv, qv = fu[ie1]

        for ie2, text in enumerate(
            ["{:.2f} ({:.2f})".format(sc, ma), "{:.2e}".format(pv), "{:.2e}".format(qv)]
        ):

            annotations.append(
                {
                    "y": y,
                    "x": _get_statistic_x(ie2),
                    "xanchor": "center",
                    "text": text,
                    **ANNOTATION,
                }
            )

        y -= he

    return annotations


def make(
    ta,
    da,
    fu,
    ac=True,
    n_jo=1,
    ra=RANDOM_SEED,
    n_sa=10,
    n_sh=10,
    pl=True,
    n_pl=8,
    tyta="continuous",
    tyda="continuous",
    st=nan,
    title="Function Heat Map",
    pa="",
):

    #
    la1_ = da.index.to_numpy()

    #
    ta = ta.loc[ta.index.intersection(da.columns)]

    if ac is not None:

        ta.sort_values(ascending=ac, inplace=True)

    la2_ = ta.index.to_numpy()

    da = da.loc[:, la2_]

    #
    si1 = la1_.size

    si2 = la2_.size

    #
    taar = ta.to_numpy()

    daar = da.to_numpy()

    #
    if callable(fu):

        po = Pool(n_jo)

        seed(ra)

        #
        print("Score ({})...".format(fu.__name__))

        sc_ = asarray(po.starmap(apply_on_2, ([taar, ro, fu] for ro in daar)))

        #
        if 0 < n_sa:

            print("0.95 MoE ({} sample)...".format(n_sa))

            #
            sc_ro_sa = full([si1, n_sa], nan)

            n_ch = int(si2 * SAMPLE_FRACTION)

            for ie in range(n_sa):

                #
                ie_ = choice(si2, n_ch, False)

                taarra = taar[ie_]

                #
                sc_ro_sa[:, ie] = po.starmap(
                    apply_on_2, ([taarra, ro, fu] for ro in daar[:, ie_])
                )

            #
            ma_ = asarray([apply_on_1(ro, get_margin_of_error) for ro in sc_ro_sa])

        else:

            ma_ = full(sc_.size, nan)

        #
        if 0 < n_sh:

            print("P-Value and Q-Value ({} shuffle)...".format(n_sh))

            #
            sc_ro_sh = full([si1, n_sh], nan)

            taarra = taar.copy()

            for ie in range(n_sh):

                #
                shuffle(taarra)

                #
                sc_ro_sh[:, ie] = po.starmap(
                    apply_on_2, ([taarra, ro, fu] for ro in daar)
                )

            #
            pv_, qv_ = get_p_value_q_value(sc_, sc_ro_sh.ravel(), "<>")

        else:

            pv_ = qv_ = full(sc_.size, nan)

        #
        po.terminate()

        fu = DataFrame(
            asarray([sc_, ma_, pv_, qv_]).T,
            la1_,
            ["Score", "MoE", "P-Value", "Q-Value"],
        )

    else:

        fu = fu.loc[la1_, :]

    #
    fu.sort_values("Score", ascending=False, inplace=True)

    if pa != "":

        fu.to_csv("{}statistic.tsv".format(pa), sep="\t")

    #
    if pl:

        fuar = fu.to_numpy()

        la1_ = fu.index

        da = da.loc[la1_, :]

        daar = da.to_numpy()

        if n_pl is not None and (n_pl / 2) < si1:

            bo_ = check_is_extreme(fuar[:, 0], "<>", n_ex=n_pl)

            fuar = fuar[bo_]

            daar = daar[bo_]

            la1_ = la1_[bo_]

        tapl, mita, mata = _process_target(taar, tyta, st)

        dapl, mida, mada = _process_data(daar, tyda, st)

        if tyta != "continuous":

            for gr, si in zip(*unique(tapl, return_counts=True)):

                if 2 < si:

                    print("Clustering {}...".format(gr))

                    ie_ = where(tapl == gr)[0]

                    iecl_ = ie_[cluster(dapl.T[ie_])[0]]

                    tapl[ie_] = tapl[iecl_]

                    dapl[:, ie_] = dapl[:, iecl_]

                    la2_[ie_] = la2_[iecl_]

        n_ro = dapl.shape[0] + 2

        he = 1 / n_ro

        layout = merge(
            {
                "height": max(480, 24 * n_ro),
                "title": {
                    "text": title,
                },
                "yaxis2": {
                    "domain": [1 - he, 1],
                    "showticklabels": False,
                },
                "yaxis": {
                    "domain": [0, 1 - he * 2],
                    "showticklabels": False,
                },
                "annotations": _make_target_annotation(ta.name, 1 - he / 2),
            },
            LAYOUT,
        )

        layout["annotations"] += _make_data_annotations(
            1 - he / 2 * 3, True, he, la1_, fuar
        )

        if pa != "":

            pa = "{}function_heat_map.html".format(pa)

        plot_plotly(
            {
                "data": [
                    {
                        "yaxis": "y2",
                        "z": tapl.reshape([1, -1]),
                        "x": la2_,
                        "zmin": mita,
                        "zmax": mata,
                        "colorscale": TYPE_COLORSCALE[tyta],
                        **HEATMAP,
                    },
                    {
                        "yaxis": "y",
                        "z": dapl[::-1],
                        "y": la1_[::-1],
                        "x": la2_,
                        "zmin": mida,
                        "zmax": mada,
                        "colorscale": TYPE_COLORSCALE[tyda],
                        **HEATMAP,
                    },
                ],
                "layout": layout,
            },
            pa=pa,
        )

    return fu


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
    la2_ = ta.index.to_numpy()

    #
    tapl, mita, mata = _process_target(ta.to_numpy(), ty, st)

    #
    n_ro = 1

    n_sp = 2

    for bu in bu_:

        n_ro += bu["data"].shape[0] + n_sp

    he = 1 / n_ro

    layout = merge(
        {
            "height": max(480, 24 * n_ro),
            "title": {
                "text": title,
            },
            "annotations": _make_target_annotation(ta.name, 1 - he / 2),
        },
        LAYOUT,
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
            "yaxis": yaxis.replace("axis", ""),
            "z": tapl.reshape([1, -1]),
            "x": la2_,
            "zmin": mita,
            "zmax": mata,
            "colorscale": TYPE_COLORSCALE[ty],
            **HEATMAP,
        }
    ]

    for ie, bu in enumerate(bu_):

        da = bu["data"].reindex(la2_, axis=1)

        fu = bu["statistic"].loc[da.index, :]

        fu.sort_values("Score", ascending=False, inplace=True)

        la1_ = fu.index.to_numpy()

        dapl, mida, mada = _process_data(da.loc[la1_, :].to_numpy(), bu["type"], st)

        yaxis = "yaxis{}".format(n_bu - ie)

        domain = [
            max(0, domain[0] - he * (n_sp + dapl.shape[0])),
            domain[0] - he * n_sp,
        ]

        layout[yaxis] = {
            "domain": domain,
            "showticklabels": False,
        }

        data.append(
            {
                "yaxis": yaxis.replace("axis", ""),
                "z": dapl[::-1],
                "y": la1_[::-1],
                "x": la2_,
                "zmin": mida,
                "zmax": mada,
                "colorscale": TYPE_COLORSCALE[bu["type"]],
                **HEATMAP,
            }
        )

        y = domain[1] + he / 2

        layout["annotations"].append(
            {
                "y": y,
                "x": 0.5,
                "xanchor": "center",
                "text": "<b>{}</b>".format(bu["name"]),
                **ANNOTATION,
            }
        )

        layout["annotations"] += _make_data_annotations(
            y, ie == 0, he, la1_, fu.to_numpy()
        )

    plot_plotly(
        {
            "data": data,
            "layout": layout,
        },
        pa=pa,
    )
