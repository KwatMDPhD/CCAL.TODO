from numpy import argsort, asarray, nonzero, unique
from pandas import DataFrame

from .cast_builtin import cast_builtin
from .COLORBAR import COLORBAR
from .DATA_TYPE_COLORSCALE import DATA_TYPE_COLORSCALE
from .merge_2_dicts import merge_2_dicts
from .plot_plotly import plot_plotly


def plot_heat_map(
    matrix,
    colorscale=None,
    row_annotations=None,
    row_annotation_colorscale=None,
    row_annotation_str=None,
    row_annotation=None,
    column_annotations=None,
    column_annotation_colorscale=None,
    column_annotation_str=None,
    column_annotation=None,
    layout=None,
):

    if not isinstance(matrix, DataFrame):

        matrix = DataFrame(matrix)

    heat_map_axis_template = {"domain": (0, 0.95)}

    annotation_axis_template = {"domain": (0.96, 1), "showticklabels": False}

    if row_annotations is not None:

        sorting_indices = argsort(row_annotations)

        row_annotations = [row_annotations[i] for i in sorting_indices]

        matrix = matrix.iloc[sorting_indices]

    if column_annotations is not None:

        sorting_indices = argsort(column_annotations)

        column_annotations = [column_annotations[i] for i in sorting_indices]

        matrix = matrix.iloc[:, sorting_indices]

    layout_template = {
        "xaxis": {
            "title": "{} (n={})".format(matrix.columns.name, matrix.columns.size),
            **heat_map_axis_template,
        },
        "yaxis": {
            "title": "{} (n={})".format(matrix.index.name, matrix.index.size),
            **heat_map_axis_template,
        },
        "xaxis2": annotation_axis_template,
        "yaxis2": annotation_axis_template,
        "annotations": [],
    }

    if layout is None:

        layout = layout_template

    else:

        layout = merge_2_dicts(layout_template, layout)

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

    annotation_template = {"showarrow": False}

    if row_annotations is not None:

        row_annotations = row_annotations[::-1]

        colorbar_x += 0.1

        data.append(
            {
                "xaxis": "x2",
                "type": "heatmap",
                "z": tuple((i,) for i in row_annotations),
                "colorscale": row_annotation_colorscale,
                "colorbar": {**COLORBAR, "x": colorbar_x, "dtick": 1},
                "hoverinfo": "z+y",
            }
        )

        if row_annotation_str is not None:

            row_annotation_template = {
                "xref": "x2",
                "x": 0,
                "xanchor": "left",
                **annotation_template,
            }

            if row_annotation is None:

                row_annotation = row_annotation_template

            else:

                row_annotation = merge_2_dicts(row_annotation_template, row_annotation)

            for i in unique(row_annotations):

                index_first, index_last = nonzero(asarray(row_annotations) == i)[0][
                    [0, -1]
                ]

                layout["annotations"].append(
                    {
                        "y": index_first + (index_last - index_first) / 2,
                        "text": row_annotation_str[i],
                        **row_annotation,
                    }
                )

    if column_annotations is not None:

        colorbar_x += 0.1

        data.append(
            {
                "yaxis": "y2",
                "type": "heatmap",
                "z": tuple((i,) for i in column_annotations),
                "transpose": True,
                "colorscale": column_annotation_colorscale,
                "colorbar": {**COLORBAR, "x": colorbar_x, "dtick": 1},
                "hoverinfo": "z+x",
            }
        )

        if column_annotation_str is not None:

            column_annotation_template = {
                "yref": "y2",
                "y": 0,
                "yanchor": "bottom",
                "textangle": -90,
                **annotation_template,
            }

            if column_annotation is None:

                column_annotation = column_annotation_template

            else:

                column_annotation = merge_2_dicts(
                    column_annotation_template, column_annotation
                )

            for i in unique(column_annotations):

                index_first, index_last = nonzero(asarray(column_annotations) == i)[0][
                    [0, -1]
                ]

                layout["annotations"].append(
                    {
                        "x": index_first + (index_last - index_first) / 2,
                        "text": column_annotation_str[i],
                        **column_annotation,
                    }
                )

    plot_plotly({"layout": layout, "data": data})
