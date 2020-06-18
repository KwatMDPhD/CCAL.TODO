from numpy import full, isnan, nan
from numpy.random import seed, shuffle
from sklearn.linear_model import LinearRegression

from .compute_empirical_p_value import compute_empirical_p_value
from .merge_2_dicts_recursively import merge_2_dicts_recursively
from .plot_plotly_figure import plot_plotly_figure
from .RANDOM_SEED import RANDOM_SEED


def correlate_2_vectors(
    vector_0,
    vector_1,
    n_permutation=10,
    random_seed=RANDOM_SEED,
    plot=True,
    marker_size=None,
    layout=None,
    html_file_path=None,
):

    model = LinearRegression()

    xs = tuple((x,) for x in vector_0)

    model.fit(xs, vector_1)

    r2 = model.score(xs, vector_1)

    if n_permutation == 0:

        p_value = nan

    else:

        r2s_shuffled = full(n_permutation, nan)

        model_ = LinearRegression()

        vector_1_copy = vector_1.copy()

        seed(seed=random_seed)

        for i in range(n_permutation):

            shuffle(vector_1_copy)

            model_.fit(xs, vector_1_copy)

            r2s_shuffled[i] = model_.score(xs, vector_1_copy)

        p_value = min(
            compute_empirical_p_value(r2, r2s_shuffled, "<"),
            compute_empirical_p_value(r2, r2s_shuffled, ">"),
        )

    if plot:

        if isnan(p_value):

            statistic_str = "R^2 = {:.2e}".format(r2)

        else:

            statistic_str = "R^2 = {:.2e}<br>P-Value = {:.2e}".format(r2, p_value)

        layout_template = {
            "annotations": [
                {
                    "xref": "paper",
                    "yref": "paper",
                    "x": 0.5,
                    "y": 1.05,
                    "xanchor": "center",
                    "yanchor": "middle",
                    "showarrow": False,
                    "text": statistic_str,
                }
            ]
        }

        if layout is None:

            layout = layout_template

        else:

            layout = merge_2_dicts_recursively(layout_template, layout)

        plot_plotly_figure(
            {
                "layout": layout,
                "data": [
                    {
                        "type": "scatter",
                        "name": "Data",
                        "x": vector_0,
                        "y": vector_1,
                        "mode": "markers",
                        "marker": {"size": marker_size, "opacity": 0.5},
                    },
                    {
                        "type": "scatter",
                        "name": "Fit",
                        "x": vector_0,
                        "y": model.coef_ * vector_0 + model.intercept_,
                    },
                ],
            },
            html_file_path,
        )

    return r2, p_value
