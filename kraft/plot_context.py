from numpy import absolute, nanmax

from .compute_vector_context import compute_vector_context
from .plot_plotly_figure import plot_plotly_figure


def plot_context(
    series,
    n_data=None,
    location=None,
    scale=None,
    degree_of_freedom=None,
    shape=None,
    fit_initial_location=None,
    fit_initial_scale=None,
    n_grid=1e3,
    degree_of_freedom_for_tail_reduction=1e8,
    multiply_distance_from_reference_argmax=False,
    global_location=None,
    global_scale=None,
    global_degree_of_freedom=None,
    global_shape=None,
    y_max_is_pdf_max=False,
    n_bin=None,
    layout=None,
    xaxis=None,
    html_file_path=None,
):

    context_dict = compute_vector_context(
        series.values,
        n_data=n_data,
        location=location,
        scale=scale,
        degree_of_freedom=degree_of_freedom,
        shape=shape,
        fit_initial_location=fit_initial_location,
        fit_initial_scale=fit_initial_scale,
        n_grid=n_grid,
        degree_of_freedom_for_tail_reduction=degree_of_freedom_for_tail_reduction,
        multiply_distance_from_reference_argmax=multiply_distance_from_reference_argmax,
        global_location=global_location,
        global_scale=global_scale,
        global_degree_of_freedom=global_degree_of_freedom,
        global_shape=global_shape,
    )

    pdf_max = context_dict["pdf"].max()

    context = context_dict["context"]

    context_abs = absolute(context)

    context_abs_max = nanmax(context_abs)

    if y_max_is_pdf_max:

        y_max = pdf_max

        if y_max < context_abs_max:

            context_abs /= context_abs_max * y_max

    else:

        y_max = max(pdf_max, context_abs_max)

    layout_template = {
        "title": series.name,
        "xaxis": xaxis,
        "yaxis": {
            "domain": (0, 0.2),
            "dtick": 1,
            "zeroline": False,
            "showticklabels": False,
        },
        "yaxis2": {"domain": (0.22, 1)},
        "legend": {"orientation": "h", "xanchor": "center", "x": 0.5, "y": -0.2},
        "annotations": [],
    }

    if layout is None:

        layout = layout_template

    else:

        layout = {**layout_template, **layout}

    for i, (format_, fit_parameter) in enumerate(
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
                "y": 1.05,
                "xanchor": "center",
                "text": format_.format(fit_parameter),
                "showarrow": False,
            }
        )

    data = []

    histogram_trace = {
        "yaxis": "y2",
        "type": "histogram",
        "legendgroup": "Data",
        "name": "Data",
        "x": series.values,
        "marker": {"color": "#20d9ba"},
        "histnorm": "probability density",
        "hoverinfo": "x+y",
    }

    if n_bin is not None:

        series_values_min = series.values.min()

        series_values_max = series.values.max()

        histogram_trace["xbins"] = {
            "start": series_values_min,
            "end": series_values_max,
            "size": (series_values_max - series_values_min) / n_bin,
        }

    data.append(histogram_trace)

    data.append(
        {
            "type": "scatter",
            "legendgroup": "Data",
            "showlegend": False,
            "x": series.values,
            "y": (0,) * series.size,
            "text": series.index,
            "mode": "markers",
            "marker": {"symbol": "line-ns-open", "color": "#20d9ba"},
            "hoverinfo": "x+text",
        }
    )

    grid = context_dict["grid"]

    line_width = 3

    pdf = context_dict["pdf"]

    data.append(
        {
            "yaxis": "y2",
            "type": "scatter",
            "name": "PDF",
            "x": grid,
            "y": pdf,
            "line": {"width": line_width, "color": "#24e7c0"},
        }
    )

    shape_pdf_reference = context_dict["shape_pdf_reference"]

    shape_pdf_reference[pdf <= shape_pdf_reference] = None

    data.append(
        {
            "yaxis": "y2",
            "type": "scatter",
            "name": "Shape Reference",
            "x": grid,
            "y": shape_pdf_reference,
            "line": {"width": line_width, "color": "#9017e6"},
        }
    )

    location_pdf_reference = context_dict["location_pdf_reference"]

    if location_pdf_reference is not None:

        location_pdf_reference[pdf <= location_pdf_reference] = None

        data.append(
            {
                "yaxis": "y2",
                "type": "scatter",
                "name": "Location Reference",
                "x": grid,
                "y": location_pdf_reference,
                "line": {"width": line_width, "color": "#4e40d8"},
            }
        )

    is_negative = context_dict["context"] < 0

    is_positive = 0 < context_dict["context"]

    for name, indices, color in (
        ("- Context", is_negative, "#0088ff"),
        ("+ Context", is_positive, "#ff1968"),
    ):

        data.append(
            {
                "yaxis": "y2",
                "type": "scatter",
                "name": name,
                "x": grid[indices],
                "y": context_abs[indices],
                "line": {"width": line_width, "color": color},
                "fill": "tozeroy",
            }
        )

    plot_plotly_figure({"layout": layout, "data": data}, html_file_path)
