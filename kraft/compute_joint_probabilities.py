from numpy import rot90
from pandas import DataFrame

from .estimate_kernel_density import estimate_kernel_density
from .plot_heat_map import plot_heat_map


def compute_joint_probabilities(variables, n_grid=64, plot=True, names=None):

    kernel_density = estimate_kernel_density(variables, n_grid=n_grid)

    probabilities = kernel_density / kernel_density.sum()

    n_dimension = len(variables)

    if plot and n_dimension == 2:

        if names is None:

            names = tuple("Variable {}".format(i) for i in range(n_dimension))

        plot_heat_map(
            DataFrame(rot90(probabilities)),
            title="P({}, {})".format(names[0], names[1]),
            xaxis_title=names[0],
            yaxis_title=names[1],
        )

    return probabilities
