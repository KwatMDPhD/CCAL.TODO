from numpy import full, isnan, nan
from numpy.random import seed, shuffle
from sklearn.linear_model import LinearRegression

from .compute_empirical_p_value import compute_empirical_p_value
from .plot_plotly_figure import plot_plotly_figure
from .RANDOM_SEED import RANDOM_SEED


def correlate_2_vectors(
    _vector_0,
    _vector_1,
    n_permutation=10,
    random_seed=RANDOM_SEED,
    plot=True,
    marker_size=16,
    title_text=None,
    xaxis_title_text=None,
    yaxis_title_text=None,
    html_file_path=None,
):

    model = LinearRegression()

    xs = tuple((x_,) for x_ in _vector_0)

    model.fit(xs, _vector_1)

    r2 = model.score(xs, _vector_1)

    if n_permutation == 0:

        p_value = nan

    else:

        r2s_shuffled = full(n_permutation, nan)

        m_ = LinearRegression()

        y_ = _vector_1.copy()

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

        r2_p_value_str = f"R^2={r2:.3f}"

        if not isnan(p_value):

            r2_p_value_str = f"{r2_p_value_str} & P-Value={p_value:.3e}"

        if title_text:

            title_text = f"{title_text}\n{r2_p_value_str}"

        else:

            title_text = r2_p_value_str

        plot_plotly_figure(
            {
                "layout": {
                    "title": {"text": title_text},
                    "xaxis": {"title": {"text": xaxis_title_text}},
                    "yaxis": {"title": {"text": yaxis_title_text}},
                },
                "data": [
                    {
                        "type": "scatter",
                        "x": _vector_0,
                        "y": _vector_1,
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
                        "x": _vector_0,
                        "y": model.coef_ * _vector_0 + model.intercept_,
                        "name": "Fit",
                        "marker": {"color": "#20d9ba"},
                    },
                ],
            },
            html_file_path,
        )

    return r2, p_value
