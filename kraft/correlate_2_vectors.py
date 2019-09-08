from numpy import full, isnan, nan
from numpy.random import seed, shuffle
from sklearn.linear_model import LinearRegression

from .compute_empirical_p_value import compute_empirical_p_value
from .plot_plotly_figure import plot_plotly_figure
from .RANDOM_SEED import RANDOM_SEED


def correlate_2_vectors(
    vector_0,
    vector_1,
    n_permutation=10,
    random_seed=RANDOM_SEED,
    plot=True,
    marker_size=16,
    title=None,
    xaxis=None,
    yaxis=None,
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

        r2_p_value_str = "R^2={:.3f}".format(r2)

        if not isnan(p_value):

            r2_p_value_str = "{} & P-Value={:.3e}".format(r2_p_value_str, p_value)

        if title["text"] is not None:

            title["text"] = "{}<br>{}".format(title["text"], r2_p_value_str)

        else:

            title["text"] = r2_p_value_str

        plot_plotly_figure(
            {
                "layout": {"title": title, "xaxis": xaxis, "yaxis": yaxis},
                "data": [
                    {
                        "type": "scatter",
                        "name": "Data",
                        "x": vector_0,
                        "y": vector_1,
                        "mode": "markers",
                        "marker": {"size": marker_size, "opacity": 0.64},
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
