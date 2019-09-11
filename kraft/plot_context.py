from numpy import absolute, nanmax

from .compute_vector_context import compute_vector_context
from .merge_2_dicts_recursively import merge_2_dicts_recursively
from .plot_plotly_figure import plot_plotly_figure


def plot_context(
    series,
    y_max_is_pdf_max=False,
    n_bin=None,
    layout=None,
    html_file_path=None,
    **compute_vector_context_keyword_arguments,
):

    layout_template = {
        "title": series.name,
        "yaxis": {"domain": (0, 0.2), "dtick": 1, "showticklabels": False},
        "yaxis2": {"domain": (0.22, 1)},
        "legend": {"orientation": "h", "xanchor": "center", "x": 0.5, "y": -0.2},
        "annotations": [],
    }

    if layout is None:

        layout = layout_template

    else:

        layout = merge_2_dicts_recursively(layout_template, layout)

    context_dict = compute_vector_context(
        series.values, **compute_vector_context_keyword_arguments
    )

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

    data_template = {
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
            "x": series.values,
            "histnorm": "probability density",
            "marker": {"color": "#20d9ba"},
            "hoverinfo": "x+y",
        },
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
        },
        merge_2_dicts_recursively(
            data_template,
            {"name": "PDF", "y": context_dict["pdf"], "line": {"color": "#24e7c0"}},
        ),
    ]

    if n_bin is not None:

        series_values_min = series.values.min()

        series_values_max = series.values.max()

        data[0]["xbins"] = {
            "start": series_values_min,
            "end": series_values_max,
            "size": (series_values_max - series_values_min) / n_bin,
        }

    shape_pdf_reference = context_dict["shape_pdf_reference"]

    shape_pdf_reference[context_dict["pdf"] <= shape_pdf_reference] = None

    data.append(
        merge_2_dicts_recursively(
            data_template,
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
            merge_2_dicts_recursively(
                data_template,
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

    for name, indices, color in (
        ("- Context", context_dict["context"] < 0, "#0088ff"),
        ("+ Context", 0 < context_dict["context"], "#ff1968"),
    ):

        data.append(
            merge_2_dicts_recursively(
                data_template,
                {
                    "name": name,
                    "x": context_dict["grid"][indices],
                    "y": context_abs[indices],
                    "line": {"color": color},
                    "fill": "tozeroy",
                },
            )
        )

    plot_plotly_figure({"layout": layout, "data": data}, html_file_path)
