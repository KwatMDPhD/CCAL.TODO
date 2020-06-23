from numpy import absolute, apply_along_axis, diff, log2, product, unique

from .kernel_density import get_density
from .plot import plot_plotly
from .point_x_dimension import get_grid_1ds, plot_grid_nd, shape


def get_probability(
    point_x_dimension, plot=True, dimension_names=None, **get_density_keyword_arguments,
):

    grid_nd, densities = get_density(
        point_x_dimension,
        plot=plot,
        dimension_names=dimension_names,
        **get_density_keyword_arguments,
    )

    probabilities = densities / (
        densities.sum()
        * product(tuple(diff(unique(dimension)).min() for dimension in grid_nd.T))
    )

    if plot:

        plot_grid_nd(
            grid_nd,
            probabilities,
            dimension_names=dimension_names,
            number_name="Probability",
        )

    return grid_nd, probabilities


def get_posterior_probability(
    point_x_dimension,
    target_dimension_number=None,
    plot=True,
    names=None,
    **get_density_keyword_arguments,
):

    grid_point_x_dimension, grid_point_x_dimension_joint_probability = get_probability(
        point_x_dimension,
        plot=plot,
        dimension_names=names,
        **get_density_keyword_arguments,
    )

    d_target_dimension = diff(unique(grid_point_x_dimension[:, -1])).min()

    def get_probability_(array):

        return array / array.sum()

    grid_point_x_dimension_posterior_probability = (
        apply_along_axis(
            get_probability_,
            -1,
            shape(
                grid_point_x_dimension_joint_probability,
                get_grid_1ds(grid_point_x_dimension),
            ),
        )
        * d_target_dimension
    ).reshape(grid_point_x_dimension_joint_probability.shape)

    if plot:

        plot_grid_nd(
            grid_point_x_dimension,
            grid_point_x_dimension_posterior_probability,
            dimension_names=names,
            number_name="Posterior Probability",
        )

    if target_dimension_number is None:

        return grid_point_x_dimension, grid_point_x_dimension_posterior_probability

    else:

        target_dimension_grid = unique(grid_point_x_dimension[:, -1])

        target_dimension_index = absolute(
            target_dimension_grid - target_dimension_number
        ).argmin()

        grid_point_x_dimension_ = grid_point_x_dimension[
            target_dimension_index :: target_dimension_grid.size, :-1
        ]

        grid_point_x_dimension_posterior_probability_ = grid_point_x_dimension_posterior_probability[
            target_dimension_index :: target_dimension_grid.size
        ]

        if plot:

            n_dimension = grid_point_x_dimension.shape[1]

            if names is None:

                names = tuple("Dimension {}".format(i) for i in range(n_dimension))

            plot_grid_nd(
                grid_point_x_dimension_,
                grid_point_x_dimension_posterior_probability_,
                dimension_names=names,
                number_name="P({} = {:.2e} (~{}) | {})".format(
                    names[-1],
                    target_dimension_grid[target_dimension_index],
                    target_dimension_number,
                    *names[:-1],
                ),
            )

        return grid_point_x_dimension_, grid_point_x_dimension_posterior_probability_


def plot_nomogram(p_t1, p_t0, names, p_t10__, plot_=False):

    layout = {
        "title": {"text": "Nomogram"},
        "xaxis": {"title": {"text": "Log Odd Ratio"}},
        "yaxis": {
            "title": {"text": "Evidence"},
            "tickvals": tuple(range(1 + len(p_t10__))),
            "ticktext": ("Prior", *names),
        },
    }

    trace_ = {"mode": "lines"}

    monogram_trace_ = {"showlegend": False}

    data = [
        {
            "x": (0, log2(p_t1 / p_t0)),
            "y": (0,) * 2,
            "marker": {"color": "#080808"},
            **monogram_trace_,
        }
    ]

    for i, (name, (p_t1__, p_t0__)) in enumerate(zip(names, p_t10__)):

        log_odd_ratios = log2((p_t1__ / p_t0__) / (p_t1 / p_t0))

        if plot_:

            plot_plotly(
                {
                    "layout": {"title": {"text": name}},
                    "data": [
                        {"y": p_t1__, **trace_},
                        {"y": p_t0__, **trace_},
                        {"name": "Log Odd Ratio", "y": log_odd_ratios, **trace_},
                    ],
                },
            )

        data.append(
            {
                "x": (log_odd_ratios.min(), log_odd_ratios.max()),
                "y": (1 + i,) * 2,
                **monogram_trace_,
            }
        )

    plot_plotly({"layout": layout, "data": data})
