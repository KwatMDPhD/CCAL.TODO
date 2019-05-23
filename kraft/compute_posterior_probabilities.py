from numpy import apply_along_axis, rot90
from pandas import DataFrame

from .plot_heat_map import plot_heat_map


def compute_posterior_probabilities(probabilities, plot=True, names=None):

    n_dimension = probabilities.ndim

    p_tv__ntvs = apply_along_axis(
        lambda nd_array: nd_array / nd_array.sum(), n_dimension - 1, probabilities
    )

    if plot and n_dimension == 2:

        if names is None:

            names = tuple(f"Axis-{i} Variable" for i in range(n_dimension))

        plot_heat_map(
            DataFrame(rot90(p_tv__ntvs)),
            title=f"P({names[1]} | {names[0]})",
            xaxis_title=names[0],
            yaxis_title=names[1],
        )

    return p_tv__ntvs
