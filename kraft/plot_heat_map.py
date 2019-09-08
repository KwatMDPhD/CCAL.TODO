from numpy import asarray, nonzero, unique

from .cast_object_to_builtin import cast_object_to_builtin
from .make_colorscale_from_colors import make_colorscale_from_colors
from .pick_colors import pick_colors
from .plot_plotly_figure import plot_plotly_figure


def plot_heat_map(
    dataframe,
    row_annotations=None,
    row_annotation_colors=None,
    row_annotation_str=None,
    row_annotation=None,
    column_annotations=None,
    column_annotation_colors=None,
    column_annotation_str=None,
    column_annotation=None,
    colorbar=None,
    layout=None,
    heat_map_xaxis=None,
    heat_map_yaxis=None,
    annotation_axis=None,
    html_file_path=None,
):

    heat_map_axis_template = {"domain": (0, 0.9), "zeroline": False, "showgrid": False}

    if heat_map_xaxis is None:

        heat_map_xaxis = heat_map_axis_template

    else:

        heat_map_xaxis = {**heat_map_axis_template, **heat_map_xaxis}

    if heat_map_yaxis is None:

        heat_map_yaxis = heat_map_axis_template

    else:

        heat_map_yaxis = {**heat_map_axis_template, **heat_map_yaxis}

    annotation_axis_template = {
        "domain": (0.92, 1),
        "zeroline": False,
        "showgrid": False,
        "ticks": "",
        "showticklabels": False,
    }

    if annotation_axis is None:

        annotation_axis = annotation_axis_template

    else:

        annotation_axis = {**annotation_axis_template, **annotation_axis}

    layout_template = {
        "height": 880,
        "width": 880,
        "xaxis": heat_map_xaxis,
        "yaxis": heat_map_yaxis,
        "xaxis2": annotation_axis,
        "yaxis2": annotation_axis,
        "annotations": [],
    }

    if layout is None:

        layout = layout_template

    else:

        layout = {**layout_template, **layout}

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
            "colorbar": colorbar,
        }
    ]

    annotation_template = {"showarrow": False, "borderpad": 0}

    if row_annotations is not None:

        row_annotations = asarray(row_annotations)

        if row_annotation_colors is None:

            row_annotation_colors = pick_colors(row_annotations)

        data.append(
            {
                "xaxis": "x2",
                "type": "heatmap",
                "z": tuple((i,) for i in row_annotations[::-1]),
                "colorscale": make_colorscale_from_colors(row_annotation_colors),
                "showscale": False,
                "hoverinfo": "z+y",
            }
        )

        if row_annotation_str is not None:

            row_annotation_template = annotation_template

            if row_annotation is None:

                row_annotation = row_annotation_template

            else:

                row_annotation = {**row_annotation_template, **row_annotation}

            for i in unique(row_annotations):

                indices = nonzero(row_annotations == i)[0]

                index_0 = indices[0]

                layout["annotations"].append(
                    {
                        "xref": "x2",
                        "x": 0,
                        "y": index_0 + (indices[-1] - index_0) / 2,
                        "text": "<b>{}</b>".format(row_annotation_str[i]),
                        **row_annotation,
                    }
                )

    if column_annotations is not None:

        column_annotations = asarray(column_annotations)

        if column_annotation_colors is None:

            column_annotation_colors = pick_colors(column_annotations)

        data.append(
            {
                "yaxis": "y2",
                "type": "heatmap",
                "z": tuple((i,) for i in column_annotations),
                "transpose": True,
                "colorscale": make_colorscale_from_colors(column_annotation_colors),
                "showscale": False,
                "hoverinfo": "z+x",
            }
        )

        if column_annotation_str is not None:

            column_annotation_template = {"textangle": -90, **annotation_template}

            if column_annotation is None:

                column_annotation = column_annotation_template

            else:

                column_annotation = {**column_annotation_template, **column_annotation}

            for i in unique(column_annotations):

                indices = nonzero(column_annotations == i)[0]

                index_0 = indices[0]

                layout["annotations"].append(
                    {
                        "yref": "y2",
                        "x": index_0 + (indices[-1] - index_0) / 2,
                        "y": 0,
                        "text": "<b>{}</b>".format(column_annotation_str[i]),
                        **column_annotation,
                    }
                )

    plot_plotly_figure({"layout": layout, "data": data}, html_file_path)
