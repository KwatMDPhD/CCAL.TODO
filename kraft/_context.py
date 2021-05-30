from numpy import (
    absolute,
    asarray,
    concatenate,
    cumsum,
    full,
    inf,
    linspace,
    minimum,
    nan,
    nanmax,
)
from pandas import DataFrame, concat
from statsmodels.sandbox.distributions.extras import ACSkewT_gen

from .array import check_array_for_bad
from .dict import merge
from .information import compute_kld
from .plot import plot_plotly


def compute_pdf_and_pdf_reference_context(
    grid, pdf, pdf_reference, multiply_distance_from_reference_argmax
):

    center = pdf_reference.argmax()

    left_kl = compute_kld(pdf[:center], pdf_reference[:center])

    right_kl = compute_kld(pdf[center:], pdf_reference[center:])

    left_kl[left_kl == inf] = 0

    right_kl[right_kl == inf] = 0

    left_context = -cumsum((left_kl / left_kl.sum())[::-1])[::-1]

    right_context = cumsum(right_kl / right_kl.sum())

    left_context *= left_kl.sum() / left_kl.size

    right_context *= right_kl.sum() / right_kl.size

    context = concatenate((left_context, right_context))

    if multiply_distance_from_reference_argmax:

        context *= absolute(grid - grid[center])

    return context


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

    is_good = ~check_array_for_bad(vector, raise_for_bad=False)

    vector_good = vector[is_good]

    if any(
        parameter is None
        for parameter in (n_data, location, scale, degree_of_freedom, shape)
    ):

        (n_data, location, scale, degree_of_freedom, shape) = fit_vector_to_skew_t_pdf(
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


def make_context_matrix(
    dataframe,
    n_job=1,
    skew_t_pdf_fit_parameter=None,
    n_grid=int(1e3),
    degree_of_freedom_for_tail_reduction=1e8,
    multiply_distance_from_reference_argmax=False,
    global_location=None,
    global_scale=None,
    global_degree_of_freedom=None,
    global_shape=None,
    tsv_file_path=None,
):

    context_matrix = concat(
        call_function_with_multiprocess(
            make_context_matrix_,
            (
                (
                    dataframe_,
                    skew_t_pdf_fit_parameter,
                    n_grid,
                    degree_of_freedom_for_tail_reduction,
                    multiply_distance_from_reference_argmax,
                    global_location,
                    global_scale,
                    global_degree_of_freedom,
                    global_shape,
                )
                for dataframe_ in split_dataframe(
                    dataframe, 0, min(dataframe.shape[0], n_job)
                )
            ),
            n_job,
        )
    )

    if tsv_file_path is not None:

        context_matrix.to_csv(tsv_file_path, sep="\t")

    return context_matrix


def make_context_matrix_(
    dataframe,
    skew_t_pdf_fit_parameter,
    n_grid,
    degree_of_freedom_for_tail_reduction,
    multiply_distance_from_reference_argmax,
    global_location,
    global_scale,
    global_degree_of_freedom,
    global_shape,
):

    context_matrix = full(dataframe.shape, nan)

    n = dataframe.shape[0]

    for (i, (index, series)) in enumerate(dataframe.iterrows()):

        if skew_t_pdf_fit_parameter is None:

            n_data = location = scale = degree_of_freedom = shape = None

        else:

            (
                n_data,
                location,
                scale,
                degree_of_freedom,
                shape,
            ) = skew_t_pdf_fit_parameter.loc[
                index, ["N Data", "Location", "Scale", "Degree of Freedom", "Shape"]
            ]

        context_matrix[i] = compute_vector_context(
            series.to_numpy(),
            n_data=n_data,
            location=location,
            scale=scale,
            degree_of_freedom=degree_of_freedom,
            shape=shape,
            n_grid=n_grid,
            degree_of_freedom_for_tail_reduction=degree_of_freedom_for_tail_reduction,
            multiply_distance_from_reference_argmax=multiply_distance_from_reference_argmax,
            global_location=global_location,
            global_scale=global_scale,
            global_degree_of_freedom=global_degree_of_freedom,
            global_shape=global_shape,
        )["context_like_array"]

    return DataFrame(
        data=context_matrix, index=dataframe.index, columns=dataframe.columns
    )


def plot_context(
    series,
    y_max_is_pdf_max=False,
    n_bin=None,
    plot_rug=True,
    layout=None,
    html_file_path=None,
    **compute_vector_context_keyword_arguments,
):

    if plot_rug:

        yaxis_domain = (0, 0.1)

        yaxis2_domain = (0.15, 1)

    else:

        yaxis_domain = (0, 0)

        yaxis2_domain = (0, 1)

    base = {
        "title": {"x": 0.5, "text": series.name},
        "yaxis": {"domain": yaxis_domain, "dtick": 1, "showticklabels": False},
        "yaxis2": {"domain": yaxis2_domain},
        "legend": {"orientation": "h", "x": 0.5, "y": -0.2, "xanchor": "center"},
        "annotations": [],
    }

    if layout is None:

        layout = base

    else:

        layout = merge(base, layout)

    context_dict = compute_vector_context(
        series.to_numpy(), **compute_vector_context_keyword_arguments
    )

    for (i, (format_, fit_parameter)) in enumerate(
        zip(
            (
                "N = {:.0f}",
                "Location = {:.2f}",
                "Scale = {:.2f}",
                "DF = {:.2f}",
                "Shape = {:.2f}",
            ),
            context_dict["fit"],
        )
    ):

        layout["annotations"].append(
            {
                "xref": "paper",
                "yref": "paper",
                "x": (i + 1) / (5 + 1),
                "y": 1.1,
                "xanchor": "center",
                "text": format_.format(fit_parameter),
                "showarrow": False,
            }
        )

    base = {
        "yaxis": "y2",
        "type": "scatter",
        "x": context_dict["grid"],
        "line": {"width": 2},
    }

    data = [
        {
            "yaxis": "y2",
            "type": "histogram",
            "legendgroup": "Data",
            "name": "Data",
            "x": series,
            "histnorm": "probability density",
            "marker": {"color": "#20d9ba"},
            "hoverinfo": "x+y",
        },
        merge(
            base,
            {"name": "PDF", "y": context_dict["pdf"], "line": {"color": "#24e7c0"}},
        ),
    ]

    if plot_rug:

        data.append(
            {
                "type": "scatter",
                "legendgroup": "Data",
                "showlegend": False,
                "x": series,
                "y": (0,) * series.size,
                "text": series.index,
                "mode": "markers",
                "marker": {"symbol": "line-ns-open", "color": "#20d9ba"},
                "hoverinfo": "x+text",
            }
        )

    if n_bin is not None:

        series_min = series.min()

        series_max = series.max()

        data[0]["xbins"] = {
            "start": series_min,
            "end": series_max,
            "size": (series_max - series_min) / n_bin,
        }

    shape_pdf_reference = context_dict["shape_pdf_reference"]

    shape_pdf_reference[context_dict["pdf"] <= shape_pdf_reference] = None

    data.append(
        merge(
            base,
            {
                "name": "Shape Reference",
                "y": shape_pdf_reference,
                "line": {"color": "#9017e6"},
            },
        )
    )

    location_pdf_reference = context_dict["location_pdf_reference"]

    if location_pdf_reference is not None:

        location_pdf_reference[context_dict["pdf"] <= location_pdf_reference] = None

        data.append(
            merge(
                base,
                {
                    "name": "Location Reference",
                    "y": location_pdf_reference,
                    "line": {"color": "#4e40d8"},
                },
            )
        )

    context_abs = absolute(context_dict["context"])

    context_abs_max = nanmax(context_abs)

    if y_max_is_pdf_max:

        pdf_max = context_dict["pdf"].max()

        if pdf_max < context_abs_max:

            context_abs *= pdf_max / context_abs_max

    for (name, indices, color) in (
        ("- Context", context_dict["context"] < 0, "#0088ff"),
        ("+ Context", 0 < context_dict["context"], "#ff1968"),
    ):

        data.append(
            merge(
                base,
                {
                    "name": name,
                    "x": context_dict["grid"][indices],
                    "y": context_abs[indices],
                    "line": {"color": color},
                    "fill": "tozeroy",
                },
            )
        )

    plot_plotly({"layout": layout, "data": data}, html_file_path)
