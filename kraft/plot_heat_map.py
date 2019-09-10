from numpy import asarray, nonzero, unique

from .cast_object_to_builtin import cast_object_to_builtin
from .make_colorscale_from_colors import make_colorscale_from_colors
from .merge_2_dicts_recursively import merge_2_dicts_recursively
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
    layout=None,
    html_file_path=None,
):

    axis_template = {"zeroline": False, "showgrid": False}

    heat_map_axis_template = {"domain": (0, 0.95), **axis_template}

    annotation_axis_template = {
        "domain": (0.96, 1),
        "ticks": "",
        "showticklabels": False,
        **axis_template,
    }

    layout_template = {
        "height": 640,
        "width": 640,
        "xaxis": heat_map_axis_template,
        "xaxis2": annotation_axis_template,
        "yaxis": heat_map_axis_template,
        "yaxis2": annotation_axis_template,
        "annotations": [],
    }

    if layout is None:

        layout = layout_template

    else:

        layout = merge_2_dicts_recursively(layout_template, layout)

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
            "colorbar": {"thicknessmode": "fraction", "thickness": 0.02, "len": 0.5},
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

                index_first, index_last = nonzero(row_annotations == i)[0][[0, -1]]

                layout["annotations"].append(
                    {
                        "xref": "x2",
                        "x": 0,
                        "y": index_first + (index_last - index_first) / 2,
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

                index_first, index_last = nonzero(column_annotations == i)[0][[0, -1]]

                layout["annotations"].append(
                    {
                        "yref": "y2",
                        "x": index_first + (index_last - index_first) / 2,
                        "y": 0,
                        "text": "<b>{}</b>".format(column_annotation_str[i]),
                        **column_annotation,
                    }
                )

    plot_plotly_figure({"layout": layout, "data": data}, html_file_path)
