from numpy import absolute, asarray, full, linspace, minimum, nan
from statsmodels.sandbox.distributions.extras import ACSkewT_gen

from .compute_pdf_and_pdf_reference_context import compute_pdf_and_pdf_reference_context
from .fit_vector_to_skew_t_pdf import fit_vector_to_skew_t_pdf
from .is_array_bad import is_array_bad
from .make_reflecting_grid import make_reflecting_grid


def compute_vector_context(
    vector,
    n_data=None,
    location=None,
    scale=None,
    degree_of_freedom=None,
    shape=None,
    fit_initial_location=None,
    fit_initial_scale=None,
    n_grid=int(1e3),
    degree_of_freedom_for_tail_reduction=1e8,
    multiply_distance_from_reference_argmax=False,
    global_location=None,
    global_scale=None,
    global_degree_of_freedom=None,
    global_shape=None,
):

    is_good = ~is_array_bad(vector, raise_if_bad=False)

    vector_good = vector[is_good]

    if any(
        parameter is None
        for parameter in (n_data, location, scale, degree_of_freedom, shape)
    ):

        n_data, location, scale, degree_of_freedom, shape = fit_vector_to_skew_t_pdf(
            vector_good,
            fit_initial_location=fit_initial_location,
            fit_initial_scale=fit_initial_scale,
        )

    grid = linspace(vector_good.min(), vector_good.max(), num=n_grid)

    skew_t_model = ACSkewT_gen()

    pdf = skew_t_model.pdf(grid, degree_of_freedom, shape, loc=location, scale=scale)

    shape_pdf_reference = minimum(
        pdf,
        skew_t_model.pdf(
            make_reflecting_grid(grid, grid[pdf.argmax()]),
            degree_of_freedom_for_tail_reduction,
            shape,
            loc=location,
            scale=scale,
        ),
    )

    shape_context = compute_pdf_and_pdf_reference_context(
        grid, pdf, shape_pdf_reference, multiply_distance_from_reference_argmax
    )

    if any(
        parameter is None
        for parameter in (
            global_location,
            global_scale,
            global_degree_of_freedom,
            global_shape,
        )
    ):

        location_pdf_reference = None

        location_context = None

        context = shape_context

    else:

        location_pdf_reference = minimum(
            pdf,
            skew_t_model.pdf(
                grid,
                global_degree_of_freedom,
                global_shape,
                loc=global_location,
                scale=global_scale,
            ),
        )

        location_context = compute_pdf_and_pdf_reference_context(
            grid, pdf, location_pdf_reference, multiply_distance_from_reference_argmax
        )

        context = shape_context + location_context

    context_like_array = full(vector.size, nan)

    context_like_array[is_good] = context[
        [absolute(grid - value).argmin() for value in vector_good]
    ]

    return {
        "fit": asarray((n_data, location, scale, degree_of_freedom, shape)),
        "grid": grid,
        "pdf": pdf,
        "shape_pdf_reference": shape_pdf_reference,
        "shape_context": shape_context,
        "location_pdf_reference": location_pdf_reference,
        "location_context": location_context,
        "context": context,
        "context_like_array": context_like_array,
    }
