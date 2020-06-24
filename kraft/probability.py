from numpy import absolute, apply_along_axis, diff, log2, product, unique

from .grid import get_grid_1ds, plot_grid_nd, shape
from .kernel_density import get_density
from .plot import plot_plotly


def get_probability(
    point_x_dimension, plot=True, dimension_names=None, **get_density_keyword_arguments,
):

    grid_nd, grid_nd_densities = get_density(
        point_x_dimension,
        plot=plot,
        dimension_names=dimension_names,
        **get_density_keyword_arguments,
    )

    grid_nd_probabilities = grid_nd_densities / (
        grid_nd_densities.sum()
        * product(tuple(diff(unique(dimension)).min() for dimension in grid_nd.T))
    )

    if plot:

        plot_grid_nd(
            grid_nd,
            grid_nd_probabilities,
            dimension_names=dimension_names,
            number_name="Probability",
        )

    return grid_nd, grid_nd_probabilities


def get_posterior_probability(
    point_x_dimension, plot=True, dimension_names=None, **get_density_keyword_arguments,
):

    grid_nd, grid_nd_joint_probabilities = get_probability(
        point_x_dimension,
        plot=plot,
        dimension_names=dimension_names,
        **get_density_keyword_arguments,
    )

    d_target_dimension = diff(unique(grid_nd[:, -1])).min()

    def get_probability_(array):

        return array / array.sum()

    grid_nd_posterior_probabilities = (
        apply_along_axis(
            get_probability_,
            -1,
            shape(grid_nd_joint_probabilities, get_grid_1ds(grid_nd)),
        )
        * d_target_dimension
    ).reshape(grid_nd_joint_probabilities.shape)

    if plot:

        plot_grid_nd(
            grid_nd,
            grid_nd_posterior_probabilities,
            dimension_names=dimension_names,
            number_name="Posterior Probability",
        )

    return grid_nd, grid_nd_posterior_probabilities


def target_posterior_probability(
    grid_nd,
    grid_nd_posterior_probabilities,
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

    grid_nd_posterior_probabilities_ = grid_nd_posterior_probabilities[
        target_dimension_i :: target_dimension_grid.size
    ]

    if plot:

        plot_grid_nd(
            grid_nd_,
            grid_nd_posterior_probabilities_,
            dimension_names=dimension_names,
            number_name="P({} = {:.2e} (~{}) | {})".format(
                dimension_names[-1],
                target_dimension_grid[target_dimension_i],
                target_dimension_number,
                *dimension_names[:-1],
            ),
        )

    return grid_nd_, grid_nd_posterior_probabilities_


def plot_nomogram(p_t0, p_t1, dimension_names, p_t0__s, p_t1__s, html_file_path=None):

    layout = {
        "title": {"text": "Nomogram"},
        "xaxis": {"title": {"text": "Log Odd Ratio"}},
        "yaxis": {
            "title": {"text": "Evidence"},
            "tickvals": tuple(range(1 + len(dimension_names))),
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

    for i, (dimension_name, p_t0__, p_t1__) in enumerate(
        zip(dimension_names, p_t0__s, p_t1__s)
    ):

        log_odd_ratios = log2((p_t1__ / p_t0__) / (p_t1 / p_t0))

        plot_plotly(
            {
                "layout": {"title": {"text": dimension_name}},
                "data": [
                    {"name": "P(Target = 0)", "y": p_t0__},
                    {"name": "P(Target = 1)", "y": p_t1__},
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

    plot_plotly({"layout": layout, "data": data}, html_file_path=html_file_path)
