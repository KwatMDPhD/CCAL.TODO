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
    point_x_dimension, plot=True, dimension_names=None, **get_density_keyword_arguments,
):

    grid_nd, joint_probabilities = get_probability(
        point_x_dimension,
        plot=plot,
        dimension_names=dimension_names,
        **get_density_keyword_arguments,
    )

    d_target_dimension = diff(unique(grid_nd[:, -1])).min()

    def get_probability_(array):

        return array / array.sum()

    posterior_probabilities = (
        apply_along_axis(
            get_probability_, -1, shape(joint_probabilities, get_grid_1ds(grid_nd))
        )
        * d_target_dimension
    ).reshape(joint_probabilities.shape)

    if plot:

        plot_grid_nd(
            grid_nd,
            posterior_probabilities,
            dimension_names=dimension_names,
            number_name="Posterior Probability",
        )

    return grid_nd, posterior_probabilities


def target_posterior_probability(
    grid_nd,
    posterior_probabilities,
    target_dimension_number,
    plot=True,
    dimension_names=None,
):

    if dimension_names is None:

        dimension_names = tuple(
            "Dimension {}".format(i) for i in range(grid_nd.shape[1])
        )

    target_dimension_grid = unique(grid_nd[:, -1])

    target_dimension_i = absolute(
        target_dimension_grid - target_dimension_number
    ).argmin()

    grid_nd_ = grid_nd[target_dimension_i :: target_dimension_grid.size, :-1]

    posterior_probabilities_ = posterior_probabilities[
        target_dimension_i :: target_dimension_grid.size
    ]

    if plot:

        plot_grid_nd(
            grid_nd_,
            posterior_probabilities_,
            dimension_names=dimension_names,
            number_name="P({} = {:.2e} (~{}) | {})".format(
                dimension_names[-1],
                target_dimension_grid[target_dimension_i],
                target_dimension_number,
                *dimension_names[:-1],
            ),
        )

    return grid_nd_, posterior_probabilities_


def plot_nomogram(p_t1, p_t0, dimension_names, p_t10__):

    layout = {
        "title": {"text": "Nomogram"},
        "xaxis": {"title": {"text": "Log Odd Ratio"}},
        "yaxis": {
            "title": {"text": "Evidence"},
            "tickvals": tuple(range(1 + len(p_t10__))),
            "ticktext": ("Prior", *dimension_names),
        },
    }

    nomogram_trace_template = {"showlegend": False}

    data = [
        {
            "x": (0, log2(p_t1 / p_t0)),
            "y": (0,) * 2,
            "marker": {"color": "#080808"},
            **nomogram_trace_template,
        }
    ]

    for i, (dimension_name, (p_t1__, p_t0__)) in enumerate(
        zip(dimension_names, p_t10__)
    ):

        log_odd_ratios = log2((p_t1__ / p_t0__) / (p_t1 / p_t0))

        plot_plotly(
            {
                "layout": {"title": {"text": dimension_name}},
                "data": [
                    {"name": "P(Target = 1)", "y": p_t1__},
                    {"name": "P(Target = 0)", "y": p_t0__},
                    {"name": "Log Odd Ratio", "y": log_odd_ratios},
                ],
            },
        )

        data.append(
            {
                "x": (log_odd_ratios.min(), log_odd_ratios.max()),
                "y": (1 + i,) * 2,
                **nomogram_trace_template,
            }
        )

    plot_plotly({"layout": layout, "data": data})
