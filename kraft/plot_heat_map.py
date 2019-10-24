from numpy import argsort, asarray, nonzero, unique

from .cast_object_to_builtin import cast_object_to_builtin
from .COLORBAR import COLORBAR
from .DATA_TYPE_COLORSCALE import DATA_TYPE_COLORSCALE
from .merge_2_dicts_recursively import merge_2_dicts_recursively
from .plot_plotly_figure import plot_plotly_figure


def plot_heat_map(
    dataframe,
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
    html_file_path=None,
):

    heat_map_axis_template = {"domain": (0, 0.95)}

    annotation_axis_template = {"domain": (0.96, 1), "showticklabels": False}

    if row_annotations is not None:

        sorting_indices = argsort(row_annotations)

        row_annotations = [row_annotations[i] for i in sorting_indices]

        dataframe = dataframe.iloc[sorting_indices]

    if column_annotations is not None:

        sorting_indices = argsort(column_annotations)

        column_annotations = [column_annotations[i] for i in sorting_indices]

        dataframe = dataframe.iloc[:, sorting_indices]

    layout_template = {
        "xaxis": {
            "title": "{} (n={})".format(dataframe.columns.name, dataframe.columns.size),
            **heat_map_axis_template,
        },
        "yaxis": {
            "title": "{} (n={})".format(dataframe.index.name, dataframe.index.size),
            **heat_map_axis_template,
        },
        "xaxis2": annotation_axis_template,
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
        print('wtf')

        y = None

    if colorscale is None:

        colorscale = DATA_TYPE_COLORSCALE["continuous"]

    colorbar_x = 1.05

    data = [
        {
            "type": "heatmap",
            "x": x,
            "y": y,
            "z": dataframe.values[::-1],
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

                row_annotation = merge_2_dicts_recursively(
                    row_annotation_template, row_annotation
                )

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

                column_annotation = merge_2_dicts_recursively(
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

    plot_plotly_figure({"layout": layout, "data": data}, html_file_path)
