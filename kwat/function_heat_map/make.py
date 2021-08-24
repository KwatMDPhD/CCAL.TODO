from multiprocessing import Pool

from numpy import array, full, nan, unique, where
from numpy.random import choice, seed, shuffle
from pandas import DataFrame

from ..array import apply as array_apply, check_is_extreme
from ..cluster import cluster
from ..constant import RANDOM_SEED, SAMPLE_FRACTION
from ..dictionary import merge
from ..function import ignore_nan_and_apply
from ..plot import plot_plotly
from ..significance import get_margin_of_error, get_p_value_q_value
from ._make_data_annotations import _make_data_annotations
from ._make_target_annotation import _make_target_annotation
from ._process_data import _process_data
from ._process_target import _process_target
from .HEATMAP import HEATMAP
from .LAYOUT import LAYOUT
from .TYPE_COLORSCALE import TYPE_COLORSCALE


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
    la1_ = da.index.values

    #
    ta = ta.loc[ta.index.intersection(da.columns)]

    if ac is not None:

        ta.sort_values(ascending=ac, inplace=True)

    la2_ = ta.index.values

    da = da.loc[:, la2_]

    #
    si1 = la1_.size

    si2 = la2_.size

    #
    taar = ta.values

    daar = da.values

    #
    if callable(fu):

        po = Pool(n_jo)

        seed(seed=ra)

        #
        print("Score ({})...".format(fu.__name__))

        sc_ = array(po.starmap(ignore_nan_and_apply, ([taar, ro, fu] for ro in daar)))

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
                    ignore_nan_and_apply, ([taarra, ro, fu] for ro in daar[:, ie_])
                )

            #
            ma_ = array([array_apply(ro, get_margin_of_error) for ro in sc_ro_sa])

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
                    ignore_nan_and_apply, ([taarra, ro, fu] for ro in daar)
                )

            #
            pv_, qv_ = get_p_value_q_value(sc_, sc_ro_sh.ravel(), "<>")

        else:

            pv_ = qv_ = full(sc_.size, nan)

        #
        po.terminate()

        fu = DataFrame(
            array([sc_, ma_, pv_, qv_]).T,
            la1_,
            ["Score", "MoE", "P-Value", "Q-Value"],
        )

    else:

        fu = fu.loc[la1_, :]

    #
    fu.sort_values("Score", ascending=False, inplace=True)

    if pa != "":

        fu.to_csv("{}statistic.tsv".format(pa), "\t")

    #
    if pl:

        fuar = fu.values

        la1_ = fu.index

        da = da.loc[la1_, :]

        daar = da.values

        if n_pl is not None and n_pl < (si1 / 2):

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
                "annotations": _make_target_annotation(1 - he / 2, ta.name),
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
                        "text": taar.reshape([1, -1]),
                        "x": la2_,
                        "zmin": mita,
                        "zmax": mata,
                        "colorscale": TYPE_COLORSCALE[tyta],
                        **HEATMAP,
                    },
                    {
                        "yaxis": "y",
                        "z": dapl[::-1],
                        "text": daar[::-1],
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
