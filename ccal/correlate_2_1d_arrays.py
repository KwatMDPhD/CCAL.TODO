from numpy import full, isnan, nan
from numpy.random import seed, shuffle
from sklearn.linear_model import LinearRegression

from .compute_empirical_p_value import compute_empirical_p_value
from .plot_and_save import plot_and_save
from .RANDOM_SEED import RANDOM_SEED


def correlate_2_1d_arrays(
    _1d_array_0,
    _1d_array_1,
    n_permutation=10,
    random_seed=RANDOM_SEED,
    plot=True,
    marker_size=16,
    title=None,
    xaxis_title=None,
    yaxis_title=None,
    html_file_path=None,
):

    model = LinearRegression()

    xs = tuple((x_,) for x_ in _1d_array_0)

    model.fit(xs, _1d_array_1)

    r2 = model.score(xs, _1d_array_1)

    if n_permutation == 0:

        p_value = nan

    else:

        r2s_shuffled = full(n_permutation, nan)

        m_ = LinearRegression()

        y_ = _1d_array_1.copy()

        seed(seed=random_seed)

        for i in range(n_permutation):

            shuffle(y_)

            m_.fit(xs, y_)

            r2s_shuffled[i] = m_.score(xs, y_)

        p_value = min(
            compute_empirical_p_value(r2, r2s_shuffled, "<"),
            compute_empirical_p_value(r2, r2s_shuffled, ">"),
        )

    if plot:

        r2_p_value_str = "R^2={:.3f}".format(r2)

        if not isnan(p_value):

            r2_p_value_str = "{} & P-Value={:.3e}".format(r2_p_value_str, p_value)

        if title:

            title = "{}\n{}".format(title, r2_p_value_str)

        else:

            title = r2_p_value_str

        plot_and_save(
            {
                "layout": {
                    "title": {"text": title},
                    "xaxis": {"title": xaxis_title},
                    "yaxis": {"title": yaxis_title},
                },
                "data": [
                    {
                        "type": "scatter",
                        "x": _1d_array_0,
                        "y": _1d_array_1,
                        "name": "Data",
                        "mode": "markers",
                        "marker": {
                            "size": marker_size,
                            "color": "#9017e6",
                            "opacity": 0.64,
                        },
                    },
                    {
                        "type": "scatter",
                        "x": _1d_array_0,
                        "y": model.coef_ * _1d_array_0 + model.intercept_,
                        "name": "Fit",
                        "marker": {"color": "#20d9ba"},
                    },
                ],
            },
            html_file_path,
        )

    return r2, p_value
