from gzip import open as gzip_open
from pickle import dump, load

from numpy import apply_along_axis, full, isnan, nan, unique
from pandas import DataFrame
from plotly.colors import make_colorscale
from scipy.spatial import Delaunay

from .array import normalize
from .CONSTANT import RANDOM_SEED
from .geometry import get_convex_hull, get_triangulation
from .kernel_density import get_bandwidth
from .plot import COLORBAR, get_color, plot_heat_map, plot_plotly
from .point import map_point, pull_point
from .point_x_dimension import get_grids, grid, reshape
from .probability import get_pdf
from .support import merge_2_dicts


def plot_node_point(
    node_x_dimension,
    point_x_dimension,
    node_marker_size=16,
    opacity=0.8,
    # TODO: consider using vector
    point_label=None,
    # TODO: consider renaming
    dimension_grid=None,
    # TODO: consider renaming
    grid_probability=None,
    # TODO: consider renaming
    grid_label=None,
    point_label_colorscale=None,
    point_value=None,
    point_value_na_opacity=None,
    point_value_colorscale=None,
    ticktext_function=None,
    layout=None,
    show_node_text=True,
    point_trace=None,
    points_to_highlight=(),
    html_file_path=None,
):

    node_x_dimension = DataFrame(
        {
            "x": node_x_dimension.iloc[:, 1].values,
            "y": 1 - node_x_dimension.iloc[:, 0].values,
        },
        index=node_x_dimension.index,
    )

    point_x_dimension = DataFrame(
        {
            "x": point_x_dimension.iloc[:, 1].values,
            "y": 1 - point_x_dimension.iloc[:, 0].values,
        },
        index=point_x_dimension.index,
    )

    title_text = "{} {} and {} {}".format(
        node_x_dimension.index.size,
        node_x_dimension.index.name,
        point_x_dimension.index.size,
        point_x_dimension.index.name,
    )

    if point_value is not None:

        title_text = "{}<br>{}".format(title_text, point_value.name)

    axis = {"showgrid": False, "zeroline": False, "showticklabels": False}

    layout_template = {
        "height": 880,
        "width": 880,
        "title": {
            "x": 0.5,
            "text": "<b>{}</b>".format(title_text),
            "font": {
                "size": 24,
                "color": "#4e40d8",
                "family": "Times New Roman, sans-serif",
            },
        },
        "xaxis": axis,
        "yaxis": axis,
        "annotations": [],
    }

    if layout is None:

        layout = layout_template

    else:

        layout = merge_2_dicts(layout_template, layout)

    triangulation_xs, triangulation_ys = get_triangulation(node_x_dimension)

    convex_hull_xs, convex_hull_ys = get_convex_hull(node_x_dimension)

    data = [
        {
            "name": "Line",
            "x": triangulation_xs + convex_hull_xs,
            "y": triangulation_ys + convex_hull_ys,
            "mode": "lines",
            "line": {"color": "#171412"},
        }
    ]

    data.append(
        {
            "name": node_x_dimension.index.name,
            "x": node_x_dimension["x"],
            "y": node_x_dimension["y"],
            "text": node_x_dimension.index,
            "mode": "markers",
            "marker": {
                "size": node_marker_size,
                "color": "#23191e",
                "line": {"width": 1, "color": "#ebf6f7"},
            },
            "hoverinfo": "text",
        }
    )

    if show_node_text:

        border_arrow_width = 1.6

        border_arrow_color = "#ebf6f7"

        layout["annotations"] += [
            {
                "x": x,
                "y": y,
                "text": "<b>{}</b>".format(node_name),
                "font": {
                    "size": 16,
                    "color": "#23191e",
                    "family": "Gravitas One, monospace",
                },
                "bgcolor": "#ffffff",
                "borderpad": 2,
                "borderwidth": border_arrow_width,
                "bordercolor": border_arrow_color,
                "arrowwidth": border_arrow_width,
                "arrowcolor": border_arrow_color,
                "opacity": opacity,
            }
            for node_name, (x, y) in node_x_dimension.iterrows()
        ]

    point_trace_template = {
        "name": point_x_dimension.index.name,
        "mode": "markers",
        "marker": {
            "size": 16,
            "color": "#20d9ba",
            "line": {"width": 0.8, "color": "#ebf6f7"},
            "opacity": opacity,
        },
        "hoverinfo": "text",
    }

    if point_trace is None:

        point_trace = point_trace_template

    else:

        point_trace = merge_2_dicts(point_trace_template, point_trace)

    if grid_label is not None:

        grid_label_not_nan_unique = unique(grid_label[~isnan(grid_label)])

        # TODO: consider renaming label to reflect int
        label_color = {
            label: get_color(
                point_label_colorscale, label, grid_label_not_nan_unique.size
            )
            for label in grid_label_not_nan_unique
        }

        data.append(
            {
                "type": "contour",
                "showlegend": False,
                # TODO: rename
                "x": dimension_grid,
                "y": 1 - dimension_grid,
                # TODO: rename
                "z": grid_probability,
                "autocontour": False,
                "ncontours": 24,
                "contours": {"coloring": "none"},
            }
        )

        for label in grid_label_not_nan_unique:

            z = grid_probability.copy()

            z[grid_label != label] = nan

            data.append(
                {
                    "type": "heatmap",
                    "x": dimension_grid,
                    "y": 1 - dimension_grid,
                    "z": z,
                    "colorscale": make_colorscale(
                        ("rgb(255, 255, 255)", label_color[label])
                    ),
                    "showscale": False,
                    "opacity": opacity,
                    "hoverinfo": "none",
                }
            )

    if point_value is not None:

        point_value = point_value.reindex(index=point_x_dimension.index)

        point_value_opacity = opacity

        if point_value_na_opacity is None:

            point_value.dropna(inplace=True)

        else:

            point_value_opacity = point_value.where(
                isnan, other=point_value_opacity
            ).fillna(value=point_value_na_opacity)

        point_value = point_value[
            point_value.abs().sort_values(na_position="first").index
        ]

        point_x_dimension = point_x_dimension.loc[point_value.index]

        if point_value.astype(float).map(float.is_integer).all():

            tickvals = point_value.unique()

            if ticktext_function is None:

                ticktext_function = "{:.0f}".format

        else:

            tickvals = point_value.describe()[
                ["min", "25%", "50%", "mean", "75%", "max"]
            ].values

            ticktext_function = "{:.2e}".format

        data.append(
            merge_2_dicts(
                point_trace,
                {
                    "x": point_x_dimension["x"],
                    "y": point_x_dimension["y"],
                    "text": tuple(
                        "{}<br>{:.2e}".format(point_name, value)
                        for point_name, value in point_value.items()
                    ),
                    "marker": {
                        "color": point_value,
                        "colorscale": point_value_colorscale,
                        "colorbar": merge_2_dicts(
                            COLORBAR,
                            {
                                "tickmode": "array",
                                "tickvals": tickvals,
                                "ticktext": tuple(
                                    ticktext_function(tickval) for tickval in tickvals
                                ),
                            },
                        ),
                        "opacity": point_value_opacity,
                    },
                },
            )
        )

    elif point_label is not None:

        for label in grid_label_not_nan_unique:

            is_label = point_label == label

            name = "Label {:.0f}".format(label)

            data.append(
                merge_2_dicts(
                    point_trace,
                    {
                        "legendgroup": name,
                        "name": name,
                        "x": point_x_dimension["x"][is_label],
                        "y": point_x_dimension["y"][is_label],
                        "text": point_x_dimension.index[is_label],
                        "marker": {"color": label_color[label]},
                    },
                )
            )

    else:

        data.append(
            merge_2_dicts(
                point_trace,
                {
                    "x": point_x_dimension["x"],
                    "y": point_x_dimension["y"],
                    "text": point_x_dimension.index,
                },
            )
        )

    layout["annotations"] += [
        {
            "x": point_x_dimension.loc[point, "x"],
            "y": point_x_dimension.loc[point, "y"],
            "text": "<b>{}</b>".format(point),
            "arrowhead": 2,
            "arrowwidth": 2,
            "arrowcolor": "#c93756",
            "standoff": None,
            "clicktoshow": "onoff",
        }
        for point in points_to_highlight
    ]

    plot_plotly({"layout": layout, "data": data}, html_file_path=html_file_path)


class GPSMap:
    def __init__(self, node_x_node_distance, point_x_node, random_seed=RANDOM_SEED):

        self.point_x_node = point_x_node

        self.node_x_dimension = DataFrame(
            map_point(node_x_node_distance, 2, random_seed=random_seed),
            index=self.point_x_node.columns,
        )

        self.point_x_dimension = DataFrame(
            pull_point(self.node_x_dimension.values, self.point_x_node.values),
            index=self.point_x_node.index,
        )

        self.point_label = None

        self.dimension_grid = None

        self.grid_probability = None

        self.grid_label = None

        self.point_label_colorscale = None

    def plot(self, **plot_gps_map_keyword_arguments):

        plot_node_point(
            self.node_x_dimension,
            self.point_x_dimension,
            point_label=self.point_label,
            dimension_grid=self.dimension_grid,
            grid_probability=self.grid_probability,
            grid_label=self.grid_label,
            point_label_colorscale=self.point_label_colorscale,
            **plot_gps_map_keyword_arguments,
        )

    def set_point_label(
        self, point_label, point_label_colorscale=None, n_grid=64,
    ):

        assert 3 <= point_label.value_counts().min()

        assert not point_label.isna().any()

        self.point_label = point_label

        mask_grid = full((n_grid,) * 2, nan)

        triangulation = Delaunay(self.node_x_dimension)

        self.dimension_grid = grid(0, 1, 1e-3, n_grid)

        for i in range(n_grid):

            for j in range(n_grid):

                mask_grid[i, j] = triangulation.find_simplex(
                    (self.dimension_grid[i], self.dimension_grid[j])
                )

        label_grid_probability = {}

        bandwidths = tuple(self.point_x_dimension.apply(get_bandwidth))

        grids = (self.dimension_grid,) * 2

        for label in self.point_label.unique():

            grid_point_x_dimension, point_pdf = get_pdf(
                self.point_x_dimension[self.point_label == label].values,
                plot=False,
                bandwidths=bandwidths,
                grids=grids,
            )

            label_grid_probability[label] = reshape(
                point_pdf, get_grids(grid_point_x_dimension)
            )

        self.grid_probability = full((n_grid,) * 2, nan)

        self.grid_label = full((n_grid,) * 2, nan)

        for i in range(n_grid):

            for j in range(n_grid):

                if mask_grid[i, j] != -1:

                    max_probability = 0

                    max_label = nan

                    for label, grid_probability in label_grid_probability.items():

                        probability = grid_probability[i, j]

                        if max_probability < probability:

                            max_probability = probability

                            max_label = label

                    self.grid_probability[i, j] = max_probability

                    self.grid_label[i, j] = max_label

        self.point_label_colorscale = point_label_colorscale

        plot_heat_map(
            DataFrame(
                apply_along_axis(normalize, 1, self.point_x_node.values, "-0-"),
                index=self.point_x_node.index,
                columns=self.point_x_node.columns,
            ).T,
            column_annotations=self.point_label,
            column_annotation_colorscale=self.point_label_colorscale,
            layout={"yaxis": {"dtick": 1}},
        )

    def predict(self, new_point_x_node, **plot_gps_map_keyword_arguments):

        plot_node_point(
            self.node_x_dimension,
            DataFrame(
                pull_point(self.node_x_dimension.values, new_point_x_node.values),
                index=new_point_x_node.index,
                columns=self.node_x_dimension.columns,
            ),
            point_label=None,
            dimension_grid=self.dimension_grid,
            grid_probability=self.grid_probability,
            grid_label=self.grid_label,
            point_label_colorscale=self.point_label_colorscale,
            **plot_gps_map_keyword_arguments,
        )


def read_gps_map(pickle_gz_file_path):

    with gzip_open(pickle_gz_file_path) as io:

        return load(io)


def write_gps_map(pickle_gz_file_path, gps_map):

    with gzip_open(pickle_gz_file_path, mode="wb") as io:

        dump(gps_map, io)
