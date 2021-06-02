from multiprocessing import Pool

from numpy import array, nan, where
from pandas import DataFrame, Series

from .information import get_jsd
from .object.array import normalize
from .plot import plot_plotly


def score_sample_and_set(
    sc_,
    el_,
    so=True,
    me="classic",
    pl=True,
    title="Score Set",
    el="Element Score",
    pa="",
):

    sc_ = sc_.dropna()

    if so:

        sc_ = sc_.sort_values()

    el_no = {el: None for el in el_}

    boh_ = array(
        [el in el_no for el in sc_.index.values],
        dtype=float,
    )

    if boh_.sum() == 0:

        return nan

    bom_ = 1 - boh_

    scab_ = sc_.abs().values

    if me == "classic":

        boh_ *= scab_

        pdh_ = boh_ / boh_.sum()

        pdm_ = bom_ / bom_.sum()

        cdh_ = pdh_[::-1].cumsum()[::-1]

        cdm_ = pdm_[::-1].cumsum()[::-1]

        enh_ = cdh_

        enm_ = cdm_

        en_ = enh_ - enm_

    elif me == "new":

        amh_ = boh_ * scab_

        amm_ = bom_ * scab_

        pdah_ = amh_ / amh_.sum()

        pdam_ = amm_ / amm_.sum()

        pda_ = scab_ / scab_.sum()

        add = 1e-8

        cdlah_ = pdah_.cumsum() + add

        cdrah_ = pdah_[::-1].cumsum()[::-1] + add

        cdlam_ = pdam_.cumsum() + add

        cdram_ = pdam_[::-1].cumsum()[::-1] + add

        cdla_ = pda_.cumsum() + add

        cdra_ = pda_[::-1].cumsum()[::-1] + add

        en_ = (
            get_jsd(cdlah_, cdlam_, nu3_=cdla_)[2]
            - get_jsd(cdrah_, cdram_, nu3_=cdra_)[2]
        )

    else:

        raise

    sesc = en_.sum() / en_.size

    if pl:

        fr = 0.16

        layout = {
            "title": {
                "text": "{}<br>Score (method={}) = {:.2f}".format(title, me, sesc),
                "x": 0.5,
            },
            "xaxis": {
                "anchor": "y",
            },
            "yaxis": {
                "domain": [0, fr],
                "title": el,
            },
            "yaxis2": {
                "domain": [fr + 0.08, 1],
            },
            "legend_orientation": "h",
            "legend": {
                "y": -0.24,
            },
        }

        ie_ = where(boh_)[0]

        data = [
            {
                "name": "Element Score ({})".format(sc_.size),
                "y": sc_.values,
                "text": sc_.index.values,
                "mode": "lines",
                "line": {
                    "width": 0,
                    "color": "#20d8ba",
                },
                "fill": "tozeroy",
            },
            {
                "name": "Element ({})".format(ie_.size),
                "yaxis": "y2",
                "x": ie_,
                "y": [0] * ie_.size,
                "text": sc_.index.values[ie_],
                "mode": "markers",
                "marker": {
                    "symbol": "line-ns-open",
                    "size": 8,
                    "color": "#2e211b",
                    "line": {
                        "width": 1.6,
                    },
                },
                "hoverinfo": "x+text",
            },
        ]

        for name, ies_, color in [
            ["- Enrichment", en_ < 0, "#0088ff"],
            ["+ Enrichment", 0 < en_, "#ff1968"],
        ]:

            data.append(
                {
                    "name": name,
                    "yaxis": "y2",
                    "y": where(ies_, en_, 0),
                    "mode": "lines",
                    "line": {
                        "width": 0,
                        "color": color,
                    },
                    "fill": "tozeroy",
                }
            )

        plot_plotly(
            {
                "data": data,
                "layout": layout,
            },
            pa=pa,
        )

    return sesc


def _score_sample_and_sets(sc_, se_el_, me):

    print(sc_.name)

    sc_ = Series(
        normalize(sc_.values, "-0-"),
        sc_.index,
        name=sc_.name,
    ).sort_values()

    return [
        score_sample_and_set(
            sc_,
            el_,
            so=False,
            me=me,
            pl=False,
        )
        for el_ in se_el_.values()
    ]


def score_samples_and_sets(nu_el_sa, se_el_, me="ks", n_jo=1, pa=""):

    po = Pool(n_jo)

    en_se_sa = DataFrame(
        array(
            po.starmap(
                _score_sample_and_sets,
                ([sc_, se_el_, me] for _, sc_ in nu_el_sa.iteritems()),
            )
        ).T,
        se_el_.keys(),
        nu_el_sa.columns,
    )

    en_se_sa.index.name = "Set"

    if pa != "":

        en_se_sa.to_csv(pa, "\t")

    return en_se_sa
