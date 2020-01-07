from numpy import absolute, nanmax

from .compute_vector_context import compute_vector_context
from .merge_2_dicts_recursively import merge_2_dicts_recursively
from .plot_plotly import plot_plotly


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

    layout_template = {
        "title": {"x": 0.5, "text": series.name},
        "yaxis": {"domain": yaxis_domain, "dtick": 1, "showticklabels": False},
        "yaxis2": {"domain": yaxis2_domain},
        "legend": {"orientation": "h", "x": 0.5, "y": -0.2, "xanchor": "center"},
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
                "y": 1.1,
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
            "x": series,
            "histnorm": "probability density",
            "marker": {"color": "#20d9ba"},
            "hoverinfo": "x+y",
        },
        merge_2_dicts_recursively(
            data_template,
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

    plot_plotly({"layout": layout, "data": data}, html_file_path)
