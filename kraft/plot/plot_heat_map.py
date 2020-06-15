from numpy import argsort, asarray, nonzero, unique
from pandas import DataFrame

from ..support.cast_builtin import cast_builtin
from ..support.merge_2_dicts import merge_2_dicts
from .COLORBAR import COLORBAR
from .DATA_TYPE_COLORSCALE import DATA_TYPE_COLORSCALE
from .plot_plotly import plot_plotly


def plot_heat_map(
    matrix,
    colorscale=None,
    ordered_annotation=False,
    row_annotations=None,
    row_annotation_colorscale=None,
    row_annotation_str=None,
    column_annotations=None,
    column_annotation_colorscale=None,
    column_annotation_str=None,
    layout=None,
    layout_annotation_row=None,
    layout_annotation_column=None,
    html_file_path=None,
):

    if not isinstance(matrix, DataFrame):

        matrix = DataFrame(matrix)

    if row_annotations is not None:

        row_annotations = asarray(row_annotations)

        if not ordered_annotation:

            index = argsort(row_annotations)

            row_annotations = row_annotations[index]

            matrix = matrix.iloc[index]

    if column_annotations is not None:

        column_annotations = asarray(column_annotations)

        if not ordered_annotation:

            index = argsort(column_annotations)

            column_annotations = column_annotations[index]

            matrix = matrix.iloc[:, index]

    heat_map_axis_ = {"domain": (0, 0.95)}

    annotation_axis_ = {"domain": (0.96, 1), "showticklabels": False}

    layout_ = {
        "xaxis": {
            "title": "{} (n={})".format(matrix.columns.name, matrix.columns.size),
            **heat_map_axis_,
        },
        "yaxis": {
            "title": "{} (n={})".format(matrix.index.name, matrix.index.size),
            **heat_map_axis_,
        },
        "xaxis2": annotation_axis_,
        "yaxis2": annotation_axis_,
        "annotations": [],
    }

    if layout is None:

        layout = layout_

    else:

        layout = merge_2_dicts(layout_, layout)

    if any(isinstance(cast_builtin(i), str) for i in matrix.columns):

        x = matrix.columns

    else:

        x = None

    if any(isinstance(cast_builtin(i), str) for i in matrix.index):

        y = matrix.index[::-1]

    else:

        y = None

    if colorscale is None:

        colorscale = DATA_TYPE_COLORSCALE["continuous"]

    colorbar_x = 1.05

    data = [
        {
            "type": "heatmap",
            "x": x,
            "y": y,
            "z": matrix.values[::-1],
            "colorscale": colorscale,
            "colorbar": {**COLORBAR, "x": colorbar_x},
        }
    ]

    annotation_ = {"showarrow": False}

    if row_annotations is not None:

        row_annotations = row_annotations[::-1]

        colorbar_x += 0.1

        data.append(
            {
                "xaxis": "x2",
                "type": "heatmap",
                "z": row_annotations.reshape(row_annotations.size, 1),
                "colorscale": row_annotation_colorscale,
                "colorbar": {**COLORBAR, "x": colorbar_x, "dtick": 1},
                "hoverinfo": "z+y",
            }
        )

        if row_annotation_str is not None:

            layout_annotation_row_ = {
                "xref": "x2",
                "x": 0,
                "xanchor": "left",
                **annotation_,
            }

            if layout_annotation_row is None:

                layout_annotation_row = layout_annotation_row_

            else:

                layout_annotation_row = merge_2_dicts(
                    layout_annotation_row_, layout_annotation_row
                )

            for i in unique(row_annotations):

                index_0, index__1 = nonzero(row_annotations == i)[0][[0, -1]]

                layout["annotations"].append(
                    {
                        "y": index_0 + (index__1 - index_0) / 2,
                        "text": row_annotation_str[i],
                        **layout_annotation_row,
                    }
                )

    if column_annotations is not None:

        colorbar_x += 0.1

        data.append(
            {
                "yaxis": "y2",
                "type": "heatmap",
                "z": column_annotations.reshape(column_annotations.size, 1),
                "transpose": True,
                "colorscale": column_annotation_colorscale,
                "colorbar": {**COLORBAR, "x": colorbar_x, "dtick": 1},
                "hoverinfo": "z+x",
            }
        )

        if column_annotation_str is not None:

            layout_column_annotation_ = {
                "yref": "y2",
                "y": 0,
                "yanchor": "bottom",
                "textangle": -90,
                **annotation_,
            }

            if layout_annotation_column is None:

                layout_annotation_column = layout_column_annotation_

            else:

                layout_annotation_column = merge_2_dicts(
                    layout_column_annotation_, layout_annotation_column
                )

            for i in unique(column_annotations):

                index_0, index__1 = nonzero(column_annotations == i)[0][[0, -1]]

                layout["annotations"].append(
                    {
                        "x": index_0 + (index__1 - index_0) / 2,
                        "text": column_annotation_str[i],
                        **layout_annotation_column,
                    }
                )

    plot_plotly({"layout": layout, "data": data}, html_file_path=html_file_path)
