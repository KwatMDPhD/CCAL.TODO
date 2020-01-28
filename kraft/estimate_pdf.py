from numpy import diff, product, unique

from .estimate_density import estimate_density
from .plot_mesh import plot_mesh


def estimate_pdf(
    point_x_dimension,
    plot=True,
    names=None,
    **estimate_kernel_density_keyword_arguments,
):

    grid_point_x_dimension, kernel_density = estimate_density(
        point_x_dimension,
        plot=plot,
        names=names,
        **estimate_kernel_density_keyword_arguments,
    )

    pdf = kernel_density / (
        kernel_density.sum()
        * product(
            tuple(diff(unique(vector)).min() for vector in grid_point_x_dimension.T)
        )
    )

    if plot:

        plot_mesh(
            grid_point_x_dimension, pdf, names=names, value_name="PDF",
        )

    return grid_point_x_dimension, pdf
