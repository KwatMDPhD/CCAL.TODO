from numpy import asarray, nonzero, unique

from .cast_object_to_builtin import cast_object_to_builtin
from .make_colorscale_from_colors import make_colorscale_from_colors
from .pick_colors import pick_colors
from .plot_plotly_figure import plot_plotly_figure


def plot_heat_map(
    dataframe,
    showscale=None,
    colorbar_x=None,
    layout_width=880,
    layout_height=880,
    heat_map_axis_domain=(0, 0.9),
    annotation_axis_domain=(0.92, 1),
    row_annotation=None,
    row_annotation_colors=None,
    row_annotation_str=None,
    row_annotation_keyword_arguments=None,
    column_annotation=None,
    column_annotation_colors=None,
    column_annotation_str=None,
    column_annotation_keyword_arguments=None,
    title_text=None,
    xaxis_title_text=None,
    yaxis_title_text=None,
    show_xaxis_ticks=True,
    show_yaxis_ticks=True,
    html_file_path=None,
):

    heat_map_axis_template = {
        "domain": heat_map_axis_domain,
        "showgrid": False,
        "zeroline": False,
    }

    annotation_axis_template = {"zeroline": False, "ticks": "", "showticklabels": False}

    layout = {
        "width": layout_width,
        "height": layout_height,
        "title": {"text": title_text},
        "xaxis": {
            "title": {"text": f"{xaxis_title_text} ({dataframe.shape[1]})"},
            "tickangle": 90,
            **heat_map_axis_template,
        },
        "xaxis2": {
            "domain": annotation_axis_domain,
            **annotation_axis_template,
        },
        "yaxis": {
            "title": {"text": f"{yaxis_title_text} ({dataframe.shape[0]})"},
            "tickangle": 0,
            **heat_map_axis_template,
        },
        "yaxis2": {
            "domain": annotation_axis_domain,
            **annotation_axis_template,
        },
    }

    if not show_xaxis_ticks:

        layout["xaxis"].update({"ticks": "", "showticklabels": False})

    if not show_yaxis_ticks:

        layout["yaxis"].update({"ticks": "", "showticklabels": False})

    colorbar_template = {"len": 0.64, "thickness": 16}

    if row_annotation is not None or column_annotation is not None:

        colorbar_template["y"] = (heat_map_axis_domain[1] - heat_map_axis_domain[0]) / 2

    if any(isinstance(cast_object_to_builtin(i), str) for i in dataframe.columns):

        x = dataframe.columns

    else:

        x = None

    if any(isinstance(cast_object_to_builtin(i), str) for i in dataframe.index):

        y = dataframe.index[::-1]

    else:

        y = None

    data = [
        {
            "type": "heatmap",
            "z": dataframe.values[::-1],
            "x": x,
            "y": y,
            "colorscale": make_colorscale_from_colors(pick_colors(dataframe)),
            "showscale": showscale,
            "colorbar": {"x": colorbar_x, **colorbar_template},
        }
    ]

    if row_annotation is not None or column_annotation is not None:

        layout["annotations"] = []

        annotation_keyword_arguments = {"showarrow": False, "borderpad": 0}

        if row_annotation is not None:

            row_annotation = asarray(row_annotation)

            if row_annotation_colors is None:

                row_annotation_colors = pick_colors(row_annotation)

            data.append(
                {
                    "xaxis": "x2",
                    "type": "heatmap",
                    "z": tuple((i,) for i in row_annotation[::-1]),
                    "colorscale": make_colorscale_from_colors(row_annotation_colors),
                    "showscale": False,
                    "hoverinfo": "y+z",
                }
            )

            if row_annotation_str is not None:

                if row_annotation_keyword_arguments is None:

                    row_annotation_keyword_arguments = {}

                for a in unique(row_annotation):

                    indices = nonzero(row_annotation == a)[0]

                    index_0 = indices[0]

                    layout["annotations"].append(
                        {
                            "xref": "x2",
                            "x": 0,
                            "y": index_0 + (indices[-1] - index_0) / 2,
                            "text": f"<b>{row_annotation_str[a]}</b>",
                            **annotation_keyword_arguments,
                            **row_annotation_keyword_arguments,
                        }
                    )

        if column_annotation is not None:

            row_annotation = asarray(row_annotation)

            if column_annotation_colors is None:

                column_annotation_colors = pick_colors(column_annotation)

            data.append(
                {
                    "yaxis": "y2",
                    "type": "heatmap",
                    "z": tuple((i,) for i in column_annotation),
                    "transpose": True,
                    "colorscale": make_colorscale_from_colors(column_annotation_colors),
                    "showscale": False,
                    "hoverinfo": "x+z",
                }
            )

            if column_annotation_str is not None:

                if column_annotation_keyword_arguments is None:

                    column_annotation_keyword_arguments = {"textangle": -90}

                for a in unique(column_annotation):

                    indices = nonzero(column_annotation == a)[0]

                    index_0 = indices[0]

                    layout["annotations"].append(
                        {
                            "yref": "y2",
                            "x": index_0 + (indices[-1] - index_0) / 2,
                            "y": 0,
                            "text": f"<b>{column_annotation_str[a]}</b>",
                            **annotation_keyword_arguments,
                            **column_annotation_keyword_arguments,
                        }
                    )

    plot_plotly_figure({"layout": layout, "data": data}, html_file_path)
