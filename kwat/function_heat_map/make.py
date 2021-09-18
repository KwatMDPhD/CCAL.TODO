from numpy import array, full, nan, unique, where
from numpy.random import choice, seed, shuffle
from pandas import DataFrame

from ..array import apply, check_extreme
from ..cluster import cluster
from ..constant import RANDOM_SEED, SAMPLE_FRACTION
from ..dictionary import merge
from ..matrix import apply_with_vector
from ..plot import plot_plotly
from ..significance import get_margin_of_error, get_p_value_and_q_value
from ._make_data_annotation import _make_data_annotation
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
    tyt="continuous",
    tyd="continuous",
    st=nan,
    layout=None,
    pa="",
):

    ta = ta.loc[ta.index.intersection(da.columns)]

    if ac is not None:

        ta.sort_values(ascending=ac, inplace=True)

    da = da.loc[:, ta.index]

    tav = ta.values

    dav = da.values

    n_ro, n_co = dav.shape

    if callable(fu):

        def _apply_with_vector(tav, dav):

            return apply_with_vector(tav, dav, fu, n_jo=n_jo)

        print("Computing score with {}".format(fu.__name__))

        seed(seed=ra)

        sc_ = _apply_with_vector(tav, dav)

        if 0 < n_sa:

            print("Computing margin of error with {} sampling".format(n_sa))

            scs_ro_sa = full([n_ro, n_sa], nan)

            n_ch = int(n_co * SAMPLE_FRACTION)

            for ie in range(n_sa):

                ie_ = choice(n_co, size=n_ch, replace=False)

                scs_ro_sa[:, ie] = _apply_with_vector(tav[ie_], dav[:, ie_])

            ma_ = array([apply(scs_, get_margin_of_error) for scs_ in scs_ro_sa])

        else:

            ma_ = full(sc_.size, nan)

        if 0 < n_sh:

            print("Computing p-value and q-value with {} shuffling".format(n_sh))

            scs_ro_sh = full([n_ro, n_sh], nan)

            tavc = tav.copy()

            for ie in range(n_sh):

                shuffle(tavc)

                scs_ro_sh[:, ie] = _apply_with_vector(tavc, dav)

            pv_, qv_ = get_p_value_and_q_value(sc_, scs_ro_sh.ravel(), "<>")

        else:

            pv_ = full(sc_.size, nan)

            qv_ = pv_.copy()

        fu = DataFrame(
            data=array([sc_, ma_, pv_, qv_]).T,
            index=da.index,
            columns=["Score", "Margin of Error", "P-Value", "Q-Value"],
        )

    else:

        fu = fu.loc[da.index, :]

    fu.sort_values("Score", ascending=False, inplace=True)

    if pa != "":

        fu.to_csv(path_or_buf="{}statistic.tsv".format(pa), sep="\t")

    if pl:

        da = da.loc[fu.index, :]

        if 0 < n_pl < (n_ro / 2):

            ex_ = check_extreme(fu.values[:, 0], "<>", n_ex=n_pl)

            fu = fu.loc[ex_, :]

            da = da.loc[fu.index, :]

        ro_ = da.index.values

        n_ro = 2 + ro_.size

        he = 1 / n_ro

        if layout is None:

            layout = {}

        layout = merge(
            merge(
                LAYOUT,
                {
                    "height": max(640, 24 * n_ro),
                    "title": {"text": "Function Heat Map"},
                    "yaxis2": {"domain": [1 - he, 1], "showticklabels": False},
                    "yaxis": {"domain": [0, 1 - he * 2], "showticklabels": False},
                    "annotations": _make_target_annotation(1 - he / 2, ta.name)
                    + _make_data_annotation(1 - he / 2 * 3, True, he, ro_, fu.values),
                },
            ),
            layout,
        )

        tavp, mit, mat = _process_target(tav, tyt, st)

        davp, mid, mad = _process_data(da.values, tyd, st)

        co_ = da.columns.values

        if tyt != "continuous":

            for gr, n_me in array(unique(tavp, return_counts=True)).T:

                if 2 < n_me:

                    print("Clustering {}".format(gr))

                    ie_ = where(tavp == gr)[0]

                    iec_ = ie_[cluster(davp.T[ie_])[0]]

                    tavp[ie_] = tavp[iec_]

                    davp[:, ie_] = davp[:, iec_]

                    co_[ie_] = co_[iec_]

        if pa != "":

            pa = "{}function_heat_map.html".format(pa)

        heatmap = {"x": co_, **HEATMAP}

        plot_plotly(
            {
                "data": [
                    {
                        "yaxis": "y2",
                        "z": tavp.reshape([1, -1]),
                        "text": tav.reshape([1, -1]),
                        "zmin": mit,
                        "zmax": mat,
                        "colorscale": TYPE_COLORSCALE[tyt],
                        **heatmap,
                    },
                    {
                        "yaxis": "y",
                        "z": davp[::-1],
                        "text": dav[::-1],
                        "y": ro_[::-1],
                        "zmin": mid,
                        "zmax": mad,
                        "colorscale": TYPE_COLORSCALE[tyd],
                        **heatmap,
                    },
                ],
                "layout": layout,
            },
            pa=pa,
        )

    return fu
