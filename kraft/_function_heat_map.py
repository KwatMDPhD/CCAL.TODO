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

from .array import (
    check_is_extreme,
    function_on_1_number_array_not_nan,
    function_on_2_number_array_not_nan,
    normalize,
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
from .plot import (
    BINARY_COLORSCALE,
    CATEGORICAL_COLORSCALE,
    CONTINUOUS_COLORSCALE,
    plot_plotly,
)
from .significance import (
    get_moe,
    get_p__q_,
)

HEATMAP_BASE = {
    "type": "heatmap",
    "showscale": False,
}

LAYOUT_BASE = {
    "width": 800,
    "margin": {
        "l": 200,
        "r": 200,
    },
    "title": {"x": 0.5},
}

ANNOTATION_BASE = {
    "xref": "paper",
    "yref": "paper",
    "yanchor": "middle",
    "font": {"size": 10},
    "showarrow": False,
}

DATA_TYPE_TO_COLORSCALE = {
    "continuous": CONTINUOUS_COLORSCALE,
    "categorical": CATEGORICAL_COLORSCALE,
    "binary": BINARY_COLORSCALE,
}


def _process_vector_for_plot(
    v,
    t,
    s,
):

    if t == "continuous":

        if 0 < v.std():

            v = function_on_1_number_array_not_nan(
                v,
                normalize,
                "-0-",
                update=True,
            ).clip(
                -s,
                s,
            )

        return (
            v,
            -s,
            s,
        )

    return (
        v.copy(),
        None,
        None,
    )


def _process_matrix_for_plot(
    m,
    t,
    s,
):

    m = m.copy()

    if t == "continuous":

        for i in range(m.shape[0]):

            m[i] = _process_vector_for_plot(
                m[i],
                t,
                s,
            )[0]

        return (
            m,
            -s,
            s,
        )

    return (
        m,
        None,
        None,
    )


def _make_vector_annotation(
    n,
    y,
):

    return [
        {
            "y": y,
            "x": 0,
            "xanchor": "right",
            "text": "<b>{}</b>".format(n),
            **ANNOTATION_BASE,
        },
    ]


def _get_statistic_x(
    i,
):

    return 1.08 + i / 6.4


def _make_matrix_annotation(
    l_,
    m,
    y,
    h,
    a,
):

    a_ = []

    if a:

        for (i, s,) in enumerate(
            [
                "Score (\u0394)",
                "P-Value",
                "Q-Value",
            ]
        ):

            a_.append(
                {
                    "y": y,
                    "x": _get_statistic_x(i),
                    "xanchor": "center",
                    "text": "<b>{}</b>".format(s),
                    **ANNOTATION_BASE,
                },
            )

    y -= h

    for i in range(l_.size):

        a_.append(
            {
                "y": y,
                "x": 0,
                "xanchor": "right",
                "text": l_[i],
                **ANNOTATION_BASE,
            },
        )

        (
            score,
            moe,
            p,
            q,
        ) = m[i]

        for (i, s,) in enumerate(
            (
                "{:.2f} ({:.2f})".format(
                    score,
                    moe,
                ),
                "{:.2e}".format(p),
                "{:.2e}".format(q),
            ),
        ):

            a_.append(
                {
                    "y": y,
                    "x": _get_statistic_x(i),
                    "xanchor": "center",
                    "text": s,
                    **ANNOTATION_BASE,
                },
            )

        y -= h

    return a_


def make(
    v,
    m,
    fs,
    v_ascending=True,
    n_job=1,
    random_seed=RANDOM_SEED,
    n_sample=10,
    n_shuffle=10,
    plot=True,
    n_plot=8,
    v_data_type="continuous",
    m_data_type="continuous",
    plot_std=nan,
    title="Function Heat Map",
    directory_path=None,
):

    v = v.loc[v.index.intersection(m.columns)]

    if v_ascending is not None:

        v.sort_values(
            ascending=v_ascending,
            inplace=True,
        )

    v2 = v.to_numpy()

    a1l_ = v.index.to_numpy()

    m = m.reindex(
        labels=a1l_,
        axis=1,
    )

    if callable(fs):

        m2 = m.to_numpy()

        (
            ma0s,
            a1s,
        ) = m.shape

        pool = Pool(n_job)

        seed(random_seed)

        print("Score ({})...".format(fs.__name__))

        score_ = asarray(
            pool.starmap(
                function_on_2_number_array_not_nan,
                (
                    [
                        v2,
                        r,
                        fs,
                    ]
                    for r in m2
                ),
            ),
        )

        if 0 < n_sample:

            print("0.95 MoE ({} sample)...".format(n_sample))

            rxs = full(
                [
                    ma0s,
                    n_sample,
                ],
                nan,
            )

            n = int(a1s * SAMPLE_FRACTION)

            for i in range(n_sample):

                i_ = choice(
                    a1s,
                    size=n,
                    replace=False,
                )

                vc = v2[i_]

                rxs[:, i,] = pool.starmap(
                    function_on_2_number_array_not_nan,
                    (
                        [
                            vc,
                            r,
                            fs,
                        ]
                        for r in m2[
                            :,
                            i_,
                        ]
                    ),
                )

            moe_ = asarray(
                [
                    function_on_1_number_array_not_nan(
                        r,
                        get_moe,
                    )
                    for r in rxs
                ],
            )

        else:

            moe_ = full(
                score_.size,
                nan,
            )

        if 0 < n_shuffle:

            print("P-Value and Q-Value ({} shuffle)...".format(n_shuffle))

            rxs = full(
                [
                    ma0s,
                    n_shuffle,
                ],
                nan,
            )

            vc = v2.copy()

            for i in range(n_shuffle):

                shuffle(vc)

                rxs[:, i,] = pool.starmap(
                    function_on_2_number_array_not_nan,
                    (
                        [
                            vc,
                            r,
                            fs,
                        ]
                        for r in m2
                    ),
                )

            (p_, q_,) = get_p__q_(
                score_,
                rxs.ravel(),
                "<>",
            )

        else:

            p_ = q_ = full(
                score_.size,
                nan,
            )

        pool.terminate()

        fs = DataFrame(
            asarray(
                [
                    score_,
                    moe_,
                    p_,
                    q_,
                ]
            ).T,
            index=m.index,
            columns=[
                "Score",
                "MoE",
                "P-Value",
                "Q-Value",
            ],
        )

    else:

        fs = fs.loc[
            m.index,
            :,
        ]

    fs.sort_values(
        "Score",
        ascending=False,
        inplace=True,
    )

    if directory_path is not None:

        fs.to_csv(
            "{}statistic.tsv".format(directory_path),
            sep="\t",
        )

    m = m.loc[
        fs.index,
        :,
    ]

    if plot:

        m2 = m.to_numpy()

        a0l_ = m.index.to_numpy()

        fs2 = fs.to_numpy()

        if n_plot is not None and (n_plot / 2) < fs2.shape[0]:

            is_ = check_is_extreme(
                fs2[
                    :,
                    0,
                ],
                "<>",
                n=n_plot,
            )

            fs2 = fs2[is_]

            m2 = m2[is_]

            a0l_ = a0l_[is_]

        (v2, vmi, vma,) = _process_vector_for_plot(
            v2,
            v_data_type,
            plot_std,
        )

        (m2, mmi, mma,) = _process_matrix_for_plot(
            m2,
            m_data_type,
            plot_std,
        )

        if v_data_type != "continuous":

            for (n, c,) in zip(
                *unique(
                    v2,
                    return_counts=True,
                )
            ):

                if 2 < c:

                    print("Clustering v={}...".format(n))

                    i_ = where(v2 == n)[0]

                    ci_ = i_[cluster(m2.T[i_])[0]]

                    v2[i_] = v2[ci_]

                    m2[:, i_,] = m2[
                        :,
                        ci_,
                    ]

                    a1l_[i_] = a1l_[ci_]

        nr = m2.shape[0] + 2

        h = 1 / nr

        layout = merge(
            {
                "height": max(
                    480,
                    24 * nr,
                ),
                "title": {"text": title},
                "yaxis2": {
                    "domain": (
                        1 - h,
                        1,
                    ),
                    "showticklabels": False,
                },
                "yaxis": {
                    "domain": (
                        0,
                        1 - h * 2,
                    ),
                    "showticklabels": False,
                },
                "annotations": _make_vector_annotation(
                    v.name,
                    1 - h / 2,
                ),
            },
            LAYOUT_BASE,
        )

        layout["annotations"] += _make_matrix_annotation(
            a0l_,
            fs2,
            1 - h / 2 * 3,
            h,
            True,
        )

        if directory_path is None:

            file_path = None

        else:

            file_path = "{}function_heat_map.html".format(directory_path)

        plot_plotly(
            {
                "data": [
                    {
                        "yaxis": "y2",
                        "z": v2.reshape(
                            [
                                1,
                                -1,
                            ]
                        ),
                        "x": a1l_,
                        "zmin": vmi,
                        "zmax": vma,
                        "colorscale": DATA_TYPE_TO_COLORSCALE[v_data_type],
                        **HEATMAP_BASE,
                    },
                    {
                        "yaxis": "y",
                        "z": m2[::-1],
                        "y": a0l_[::-1],
                        "x": a1l_,
                        "zmin": mmi,
                        "zmax": mma,
                        "colorscale": DATA_TYPE_TO_COLORSCALE[m_data_type],
                        **HEATMAP_BASE,
                    },
                ],
                "layout": layout,
            },
            file_path=file_path,
        )

    return fs


def summarize(
    v,
    d_,
    intersect=True,
    v_ascending=True,
    v_data_type="continuous",
    plot_std=nan,
    title="Function Heat Map Summary",
    file_path=None,
):

    if intersect:

        for d in d_:

            v = v.loc[v.index.intersection(d["m"].columns)]

    if v_ascending is not None:

        v.sort_values(
            ascending=v_ascending,
            inplace=True,
        )

    v2 = v.to_numpy()

    a1l_ = v.index.to_numpy()

    (v2, vmi, vma,) = _process_vector_for_plot(
        v2,
        v_data_type,
        plot_std,
    )

    nr = 1

    ns = 2

    for d in d_:

        nr += d["m"].shape[0] + ns

    h = 1 / nr

    layout = merge(
        {
            "height": max(
                480,
                24 * nr,
            ),
            "title": {"text": title},
            "annotations": _make_vector_annotation(
                v.name,
                1 - h / 2,
            ),
        },
        LAYOUT_BASE,
    )

    nd = len(d_)

    yaxis = "yaxis{}".format(nd + 1)

    domain = (
        1 - h,
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
            "z": v2.reshape(
                (
                    1,
                    -1,
                )
            ),
            "x": a1l_,
            "zmin": vmi,
            "zmax": vma,
            "colorscale": DATA_TYPE_TO_COLORSCALE[v_data_type],
            **HEATMAP_BASE,
        },
    ]

    for (
        i,
        d,
    ) in enumerate(d_):

        m = d["m"]

        m = m.reindex(
            labels=a1l_,
            axis=1,
        )

        statistic_m = d["statistic"].loc[
            m.index,
            :,
        ]

        statistic_m.sort_values(
            "Score",
            ascending=False,
            inplace=True,
        )

        m = m.loc[
            statistic_m.index,
            :,
        ]

        m2 = m.to_numpy()

        a0l_ = m.index.to_numpy()

        statistic_matrix = statistic_m.to_numpy()

        (m2, mmi, mma,) = _process_matrix_for_plot(
            m2,
            d["data_type"],
            plot_std,
        )

        yaxis = "yaxis{}".format(nd - i)

        domain = (
            max(
                0,
                domain[0] - h * (ns + m.shape[0]),
            ),
            domain[0] - h * ns,
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
                "z": m2[::-1],
                "y": a0l_[::-1],
                "x": a1l_,
                "zmin": mmi,
                "zmax": mma,
                "colorscale": DATA_TYPE_TO_COLORSCALE[d["data_type"]],
                **HEATMAP_BASE,
            },
        )

        y = domain[1] + h / 2

        layout["annotations"].append(
            {
                "y": y,
                "x": 0.5,
                "xanchor": "center",
                "text": "<b>{}</b>".format(d["name"]),
                **ANNOTATION_BASE,
            },
        )

        layout["annotations"] += _make_matrix_annotation(
            a0l_,
            statistic_matrix,
            y,
            h,
            i == 0,
        )

    plot_plotly(
        {
            "data": data,
            "layout": layout,
        },
        file_path=file_path,
    )
