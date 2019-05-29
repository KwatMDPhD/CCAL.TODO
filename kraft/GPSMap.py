from matplotlib.colors import LinearSegmentedColormap, to_hex
from numpy import (
    arange,
    asarray,
    cos,
    diag,
    exp,
    full,
    linspace,
    mean,
    nan,
    pi,
    rot90,
    sin,
    sort,
    unique,
)
from numpy.random import choice, normal, random_sample, seed
from pandas import DataFrame, Series
from scipy.spatial import Delaunay
from scipy.spatial.distance import euclidean, pdist, squareform
from scipy.stats import pearsonr

from .apply_function_on_2_2d_arrays_slices import apply_function_on_2_2d_arrays_slices
from .check_nd_array_for_bad import check_nd_array_for_bad
from .clip_nd_array_by_standard_deviation import clip_nd_array_by_standard_deviation
from .cluster_2d_array import cluster_2d_array
from .compute_1d_array_bandwidth import compute_1d_array_bandwidth
from .compute_information_distance_between_2_1d_arrays import (
    compute_information_distance_between_2_1d_arrays,
)
from .estimate_density import estimate_density
from .get_colormap_colors import get_colormap_colors
from .get_data_type import get_data_type
from .get_triangulation_edges_from_point_x_dimension import (
    get_triangulation_edges_from_point_x_dimension,
)
from .make_colorscale_from_colors import make_colorscale_from_colors
from .normalize_nd_array import normalize_nd_array
from .pick_colors import pick_colors
from .plot_and_save import plot_and_save
from .plot_heat_map import plot_heat_map
from .RANDOM_SEED import RANDOM_SEED
from .scale_point_x_dimension_dimension import scale_point_x_dimension_dimension
from .train_and_classify import train_and_classify
from .unmesh import unmesh

element_marker_size = 16

grid_label_opacity_without_annotation = 0.64

grid_label_opacity_with_annotation = 0.24

grid_extension = 1 / 1e3


def _check_node_x_element(node_x_element):

    if node_x_element.index.has_duplicates:

        raise ValueError("node_x_element should not have duplicated index.")

    if node_x_element.columns.has_duplicates:

        raise ValueError("node_x_element should not have duplicated column.")

    if not node_x_element.applymap(
        lambda value: isinstance(value, (int, float))
    ).values.all():

        raise ValueError("node_x_element should be only int or float.")

    check_nd_array_for_bad(node_x_element.values)


def _make_element_x_dimension(node_x_element, node_x_dimension, n_pull, pull_power):

    element_x_dimension = full(
        (node_x_element.shape[1], node_x_dimension.shape[1]), nan
    )

    node_x_element = normalize_nd_array(node_x_element, None, "0-1")

    for i in range(node_x_element.shape[1]):

        pulls = node_x_element[:, i]

        if 3 < pulls.size:

            pulls = normalize_nd_array(pulls, None, "0-1")

        if n_pull is not None:

            pulls[pulls < sort(pulls)[-n_pull]] = 0

        if pull_power is not None:

            pulls = pulls ** pull_power

        for j in range(node_x_dimension.shape[1]):

            element_x_dimension[i, j] = (
                pulls * node_x_dimension[:, j]
            ).sum() / pulls.sum()

    return element_x_dimension


def _plot(
    nodes,
    node_name,
    node_x_dimension,
    elements,
    element_name,
    element_x_dimension,
    element_marker_size,
    element_label,
    grid_values,
    grid_labels,
    label_colors,
    grid_label_opacity,
    annotation_x_element,
    annotation_types,
    annotation_std_maxs,
    layout_size,
    title,
    html_file_path,
):

    axis_template = {
        "showgrid": False,
        "zeroline": False,
        "ticks": "",
        "showticklabels": False,
    }

    layout = {
        "width": layout_size,
        "height": layout_size,
        "title": {"text": title, "font": {"size": 32, "color": "#4c221b"}},
        "xaxis": axis_template,
        "yaxis": axis_template,
    }

    data = []

    node_opacity = 0.88

    edge_xs, edge_ys = get_triangulation_edges_from_point_x_dimension(node_x_dimension)

    data.append(
        {
            "type": "scatter",
            "legendgroup": node_name,
            "showlegend": False,
            "x": edge_xs,
            "y": edge_ys,
            "mode": "lines",
            "line": {"color": "#23191e"},
            "opacity": node_opacity,
            "hoverinfo": None,
        }
    )

    node_color = "#2e211b"

    node_color_accent = "#ebf6f7"

    data.append(
        {
            "type": "scatter",
            "name": node_name,
            "legendgroup": node_name,
            "x": node_x_dimension[:, 0],
            "y": node_x_dimension[:, 1],
            "text": arange(len(nodes)),
            "mode": "markers+text",
            "marker": {
                "size": 32,
                "color": node_color,
                "line": {"width": 2.4, "color": node_color_accent},
            },
            "textfont": {"size": 16, "color": node_color_accent},
            "opacity": node_opacity,
        }
    )

    if grid_values is not None:

        x = linspace(0 - grid_extension, 1 + grid_extension, num=grid_values.shape[1])

        y = linspace(0 - grid_extension, 1 + grid_extension, num=grid_values.shape[0])

        data.append(
            {
                "type": "contour",
                "showlegend": False,
                "z": grid_values[::-1],
                "x": x,
                "y": y,
                "autocontour": False,
                "ncontours": 24,
                "contours": {"coloring": "none"},
                "showscale": False,
                "opacity": node_opacity,
            }
        )

        grid_labels_unique = unique(
            grid_labels[~check_nd_array_for_bad(grid_labels, raise_for_bad=False)]
        ).astype(int)

        for label in grid_labels_unique:

            z = grid_values.copy()

            z[grid_labels != label] = nan

            data.append(
                {
                    "type": "heatmap",
                    "z": z[::-1],
                    "x": x,
                    "y": y,
                    "colorscale": make_colorscale_from_colors(
                        get_colormap_colors(
                            LinearSegmentedColormap.from_list(
                                None, ("#ffffff", label_colors[label])
                            )
                        )
                    ),
                    "showscale": False,
                    "opacity": grid_label_opacity,
                }
            )

    element_marker_line = {"width": 1, "color": node_color}

    element_opacity = 0.8

    if annotation_x_element is not None:

        if 1 < annotation_x_element.shape[0]:

            shapes = []

            marker_size_factor = element_marker_size / layout_size

        for i, (_, element_value) in enumerate(annotation_x_element.iterrows()):

            element_value.dropna(inplace=True)

            element_x_dimension_ = element_x_dimension[
                [elements.index(element) for element in element_value.index]
            ]

            if annotation_types is None:

                data_type = get_data_type(element_value)

            else:

                data_type = annotation_types[i]

            if data_type in ("binary", "categorical"):

                element_value = element_value.rank(method="dense")

            elif data_type == "continuous":

                if annotation_std_maxs is not None:

                    element_value = Series(
                        clip_nd_array_by_standard_deviation(
                            element_value.values,
                            annotation_std_maxs[i],
                            raise_for_bad=False,
                        ),
                        name=element_value.name,
                        index=element_value.index,
                    )

            if annotation_x_element.shape[0] == 1:

                sorted_indices = element_value.abs().argsort()

                element_value = element_value[sorted_indices]

                element_x_dimension_ = element_x_dimension_[sorted_indices]

            colorscale = make_colorscale_from_colors(pick_colors(element_value))

            if 1 == annotation_x_element.shape[0]:

                data.append(
                    {
                        "type": "scatter",
                        "name": element_name,
                        "showlegend": False,
                        "x": element_x_dimension_[:, 0],
                        "y": element_x_dimension_[:, 1],
                        "text": element_value.index,
                        "mode": "markers",
                        "marker": {
                            "size": element_marker_size,
                            "color": element_value,
                            "colorscale": colorscale,
                            "showscale": data_type == "continuous",
                            "colorbar": {"len": 0.64, "thickness": layout_size // 64},
                            "line": element_marker_line,
                        },
                        "opacity": element_opacity,
                        "hoverinfo": "text",
                    }
                )

            else:

                for value, (x, y) in zip(element_value.values, element_x_dimension_):

                    color = to_hex(
                        colorscale[
                            int(
                                (len(colorscale) - 1)
                                * (value - element_value.min())
                                / (element_value.max() - element_value.min())
                            )
                        ][1]
                    )

                    sector_radian = pi * 2 / annotation_x_element.shape[0]

                    sector_radians = linspace(
                        sector_radian * i, sector_radian * (i + 1), num=16
                    )

                    path = f"M {x} {y}"

                    for x_, y_ in zip(cos(sector_radians), sin(sector_radians)):

                        path += f" L {x + x_ * marker_size_factor} {y + y_ * marker_size_factor}"

                    path += " Z"

                    shapes.append(
                        {
                            "type": "path",
                            "x0": x,
                            "y0": y,
                            "path": path,
                            "fillcolor": color,
                            "line": element_marker_line,
                            "opacity": element_opacity,
                        }
                    )

        if 1 < annotation_x_element.shape[0]:

            layout["shapes"] = shapes

    elif element_label is not None:

        for label in grid_labels_unique:

            element_indices = element_label == label

            label_str = str(label)

            data.append(
                {
                    "type": "scatter",
                    "name": label_str,
                    "legendgroup": label_str,
                    "x": element_x_dimension[element_indices, 0],
                    "y": element_x_dimension[element_indices, 1],
                    "text": asarray(elements)[element_indices],
                    "mode": "markers",
                    "marker": {
                        "size": element_marker_size,
                        "color": label_colors[label],
                        "line": element_marker_line,
                    },
                    "opacity": element_opacity,
                    "hoverinfo": "text",
                }
            )

    else:

        data.append(
            {
                "type": "scatter",
                "name": element_name,
                "x": element_x_dimension[:, 0],
                "y": element_x_dimension[:, 1],
                "text": elements,
                "mode": "markers",
                "marker": {
                    "size": element_marker_size,
                    "color": "#20d9ba",
                    "line": element_marker_line,
                },
                "opacity": element_opacity,
                "hoverinfo": "text",
            }
        )

    plot_and_save({"layout": layout, "data": data}, html_file_path)


class GPSMap:
    def __init__(
        self,
        w=None,
        h=None,
        function_to_blend_node_node_distance=mean,
        node_x_dimension=None,
        mds_random_seed=RANDOM_SEED,
        w_n_pull=None,
        w_pull_power=None,
        h_n_pull=None,
        h_pull_power=None,
        plot=True,
    ):

        self.w = None

        self.h = None

        self.nodes = None

        self.node_name = None

        self.w_elements = None

        self.w_element_name = None

        self.w_distance__node_x_node = None

        self.h_elements = None

        self.h_element_name = None

        self.h_distance__node_x_node = None

        self.distance__node_x_node = None

        self.node_x_dimension = node_x_dimension

        self.triangulation = None

        self.w_n_pull = w_n_pull

        self.w_pull_power = w_pull_power

        self.w_element_x_dimension = None

        self.h_n_pull = h_n_pull

        self.h_pull_power = h_pull_power

        self.h_element_x_dimension = None

        self.n_grid = None

        self.mask_grid = None

        self.w_element_label = None

        self.w_bandwidth_factor = None

        self.w_grid_values = None

        self.w_grid_labels = None

        self.w_label_colors = None

        self.h_element_label = None

        self.h_bandwidth_factor = None

        self.h_grid_values = None

        self.h_grid_labels = None

        self.h_label_colors = None

        self.w_distance__element_x_element = None

        self.w_distance__node_x_element = None

        self.h_distance__element_x_element = None

        self.h_distance__node_x_element = None

        if w is not None:

            _check_node_x_element(w)

        if h is not None:

            _check_node_x_element(h)

        if w is not None and h is not None:

            if (w.index == h.index).all() and w.index.name == h.index.name:

                self.nodes = w.index.tolist()

                self.node_name = w.index.name

            else:

                raise ValueError("w and h indices mismatch.")

        elif w is not None:

            self.nodes = w.index.tolist()

            self.node_name = w.index.name

        elif h is not None:

            self.nodes = h.index.tolist()

            self.node_name = h.index.name

        if w is not None:

            self.w = w.values

            self.w_elements = w.columns.tolist()

            self.w_element_name = w.columns.name

            self.w_distance__node_x_node = squareform(
                pdist(self.w, metric=compute_information_distance_between_2_1d_arrays)
            )

            if plot:

                plot_heat_map(
                    DataFrame(self.w, index=self.nodes, columns=self.w_elements).iloc[
                        :, cluster_2d_array(self.w, 1)
                    ],
                    title="W",
                    xaxis_title=self.w_element_name,
                    yaxis_title=self.node_name,
                )

                plot_heat_map(
                    DataFrame(
                        self.w_distance__node_x_node,
                        index=self.nodes,
                        columns=self.nodes,
                    ).iloc[
                        cluster_2d_array(self.w_distance__node_x_node, 0),
                        cluster_2d_array(self.w_distance__node_x_node, 1),
                    ],
                    title=f"{self.node_name}-{self.node_name} Distance in W",
                    xaxis_title=self.node_name,
                    yaxis_title=self.node_name,
                )

        if h is not None:

            self.h = h.values

            self.h_elements = h.columns.tolist()

            self.h_element_name = h.columns.name

            self.h_distance__node_x_node = squareform(
                pdist(self.h, metric=compute_information_distance_between_2_1d_arrays)
            )

            if plot:

                plot_heat_map(
                    DataFrame(self.h, index=self.nodes, columns=self.h_elements).iloc[
                        :, cluster_2d_array(self.h, 1)
                    ],
                    title="H",
                    xaxis_title=self.h_element_name,
                    yaxis_title=self.node_name,
                )

                plot_heat_map(
                    DataFrame(
                        self.h_distance__node_x_node,
                        index=self.nodes,
                        columns=self.nodes,
                    ).iloc[
                        cluster_2d_array(self.h_distance__node_x_node, 0),
                        cluster_2d_array(self.h_distance__node_x_node, 1),
                    ],
                    title=f"{self.node_name}-{self.node_name} Distance in H",
                    xaxis_title=self.node_name,
                    yaxis_title=self.node_name,
                )

        if w is not None and h is not None:

            self.distance__node_x_node = full((len(self.nodes),) * 2, nan)

            for i in range(len(self.nodes)):

                for j in range(len(self.nodes)):

                    self.distance__node_x_node[
                        i, j
                    ] = function_to_blend_node_node_distance(
                        (
                            self.w_distance__node_x_node[i, j],
                            self.h_distance__node_x_node[i, j],
                        )
                    )

            if plot:

                plot_heat_map(
                    DataFrame(
                        self.distance__node_x_node, index=self.nodes, columns=self.nodes
                    ).iloc[
                        cluster_2d_array(self.distance__node_x_node, 0),
                        cluster_2d_array(self.distance__node_x_node, 1),
                    ],
                    title=f"{self.node_name}-{self.node_name} Distance in W and H",
                    xaxis_title=self.node_name,
                    yaxis_title=self.node_name,
                )

        elif w is not None:

            self.distance__node_x_node = self.w_distance__node_x_node

        elif h is not None:

            self.distance__node_x_node = self.h_distance__node_x_node

        if self.node_x_dimension is None:

            self.node_x_dimension = normalize_nd_array(
                scale_point_x_dimension_dimension(
                    2,
                    distance__point_x_point=self.distance__node_x_node,
                    random_seed=mds_random_seed,
                ),
                0,
                "0-1",
            )

        self.triangulation = Delaunay(self.node_x_dimension)

        if w is not None:

            self.w_element_x_dimension = _make_element_x_dimension(
                self.w, self.node_x_dimension, self.w_n_pull, self.w_pull_power
            )

        if h is not None:

            self.h_element_x_dimension = _make_element_x_dimension(
                self.h, self.node_x_dimension, self.h_n_pull, self.h_pull_power
            )

    def plot(
        self,
        w_or_h,
        grid_label_opacity=None,
        annotation_x_element=None,
        annotation_types=None,
        annotation_std_maxs=None,
        elements_to_be_emphasized=None,
        element_marker_size=element_marker_size,
        layout_size=880,
        title=None,
        html_file_path=None,
    ):

        if w_or_h == "w":

            elements = self.w_elements

            element_name = self.w_element_name

            element_x_dimension = self.w_element_x_dimension

            element_label = self.w_element_label

            grid_values = self.w_grid_values

            grid_labels = self.w_grid_labels

            label_colors = self.w_label_colors

        elif w_or_h == "h":

            elements = self.h_elements

            element_name = self.h_element_name

            element_x_dimension = self.h_element_x_dimension

            element_label = self.h_element_label

            grid_values = self.h_grid_values

            grid_labels = self.h_grid_labels

            label_colors = self.h_label_colors

        if annotation_x_element is not None:

            annotation_x_element = annotation_x_element.reindex(columns=elements)

        if grid_label_opacity is None:

            if annotation_x_element is None:

                grid_label_opacity = grid_label_opacity_without_annotation

            else:

                grid_label_opacity = grid_label_opacity_with_annotation

        if title is None:

            title = w_or_h.title()

        _plot(
            self.nodes,
            self.node_name,
            self.node_x_dimension,
            elements,
            element_name,
            element_x_dimension,
            element_marker_size,
            element_label,
            grid_values,
            grid_labels,
            label_colors,
            grid_label_opacity,
            annotation_x_element,
            annotation_types,
            annotation_std_maxs,
            layout_size,
            title,
            html_file_path,
        )

    def set_element_label(
        self,
        w_or_h,
        element_label,
        n_grid=128,
        bandwidth_factor=1,
        label_colors=None,
        plot=True,
    ):

        if not element_label.map(lambda label: isinstance(label, int)).all():

            raise ValueError("element_label should be int.")

        if (element_label.value_counts() < 3).any():

            raise ValueError("Each label should have at least 3 elements.")

        if w_or_h == "w":

            element_x_dimension = self.w_element_x_dimension

        elif w_or_h == "h":

            element_x_dimension = self.h_element_x_dimension

        self.n_grid = n_grid

        self.mask_grid = full((self.n_grid,) * 2, nan)

        x_grid_for_j = linspace(0 - grid_extension, 1 + grid_extension, num=self.n_grid)

        y_grid_for_i = linspace(1 + grid_extension, 0 - grid_extension, num=self.n_grid)

        for i in range(self.n_grid):

            for j in range(self.n_grid):

                self.mask_grid[i, j] = (
                    self.triangulation.find_simplex((x_grid_for_j[j], y_grid_for_i[i]))
                    == -1
                )

        label_grid_probabilities = {}

        global_bandwidths = (
            asarray(
                (
                    compute_1d_array_bandwidth(element_x_dimension[:, 0]),
                    compute_1d_array_bandwidth(element_x_dimension[:, 1]),
                )
            )
            * bandwidth_factor
        )

        n_dimension = element_x_dimension.shape[1]

        grid_mins = (0 - grid_extension,) * n_dimension

        grid_maxs = (1 + grid_extension,) * n_dimension

        fraction_grid_extensions = (0,) * n_dimension

        n_grids = (self.n_grid,) * n_dimension

        for label in element_label.sort_values().unique():

            density = rot90(
                unmesh(
                    *estimate_density(
                        element_x_dimension[element_label == label],
                        bandwidths=global_bandwidths,
                        grid_mins=grid_mins,
                        grid_maxs=grid_maxs,
                        fraction_grid_extensions=fraction_grid_extensions,
                        n_grids=n_grids,
                        plot=False,
                    )
                )[1]
            )

            label_grid_probabilities[label] = density / density.sum()

        shape = (self.n_grid,) * n_dimension

        grid_values = full(shape, nan)

        grid_labels = full(shape, nan)

        for i in range(self.n_grid):

            for j in range(self.n_grid):

                if not self.mask_grid[i, j]:

                    max_probability = 0

                    max_label = nan

                    for label, grid_probabilities in label_grid_probabilities.items():

                        probability = grid_probabilities[i, j]

                        if max_probability < probability:

                            max_probability = probability

                            max_label = label

                    grid_values[i, j] = max_probability

                    grid_labels[i, j] = max_label

        if label_colors is None:

            label_colors = pick_colors(element_label)

        if w_or_h == "w":

            self.w_element_label = element_label

            self.w_bandwidth_factor = bandwidth_factor

            self.w_grid_values = grid_values

            self.w_grid_labels = grid_labels

            self.w_label_colors = label_colors

        elif w_or_h == "h":

            self.h_element_label = element_label

            self.h_bandwidth_factor = bandwidth_factor

            self.h_grid_values = grid_values

            self.h_grid_labels = grid_labels

            self.h_label_colors = label_colors

        if plot:

            if w_or_h == "w":

                column_annotation = self.w_element_label.sort_values()

                z = DataFrame(self.w, index=self.nodes, columns=self.w_elements)[
                    column_annotation.index
                ]

                element_name = self.w_element_name

            elif w_or_h == "h":

                column_annotation = self.h_element_label.sort_values()

                z = DataFrame(self.h, index=self.nodes, columns=self.h_elements)[
                    column_annotation.index
                ]

                element_name = self.h_element_name

            plot_heat_map(
                z,
                column_annotation=column_annotation,
                column_annotation_colors=label_colors,
                title=w_or_h.title(),
                xaxis_title=element_name,
                yaxis_title=self.node_name,
            )

    def predict(
        self,
        w_or_h,
        node_x_predicting_element,
        support_vector_parameter_c=1e3,
        n_pull=None,
        pull_power=None,
        grid_label_opacity=None,
        annotation_x_element=None,
        annotation_types=None,
        annotation_std_maxs=None,
        element_marker_size=element_marker_size,
        layout_size=880,
        title=None,
        html_file_path=None,
    ):

        _check_node_x_element(node_x_predicting_element)

        predicting_elements = node_x_predicting_element.columns.tolist()

        if w_or_h == "w":

            node_x_element = self.w

            element_name = self.w_element_name

            if n_pull is None:

                n_pull = self.w_n_pull

            if pull_power is None:

                pull_power = self.w_pull_power

            element_label = self.w_element_label

            grid_values = self.w_grid_values

            grid_labels = self.w_grid_labels

            label_colors = self.w_label_colors

        elif w_or_h == "h":

            node_x_element = self.h

            element_name = self.h_element_name

            if n_pull is None:

                n_pull = self.h_n_pull

            if pull_power is None:

                pull_power = self.h_pull_power

            element_label = self.h_element_label

            grid_values = self.h_grid_values

            grid_labels = self.h_grid_labels

            label_colors = self.h_label_colors

        predicting_element_x_dimension = _make_element_x_dimension(
            node_x_predicting_element.values, self.node_x_dimension, n_pull, pull_power
        )

        if element_label is not None:

            element_predicted_label = Series(
                train_and_classify(
                    node_x_element.T,
                    element_label,
                    node_x_predicting_element.T,
                    c=support_vector_parameter_c,
                    tol=1e-8,
                ),
                name=f"Predicted {element_name} Label",
                index=predicting_elements,
            )

        else:

            element_predicted_label = None

        element_predicted_label = None

        if annotation_x_element is not None:

            annotation_x_element = annotation_x_element.reindex(
                columns=predicting_elements
            )

        if grid_label_opacity is None:

            if annotation_x_element is None:

                grid_label_opacity = grid_label_opacity_without_annotation

            else:

                grid_label_opacity = grid_label_opacity_with_annotation

        if title is None:

            title = f"{w_or_h.title()} (predicted)"

        _plot(
            self.nodes,
            self.node_name,
            self.node_x_dimension,
            predicting_elements,
            element_name,
            predicting_element_x_dimension,
            element_marker_size,
            element_predicted_label,
            grid_values,
            grid_labels,
            label_colors,
            grid_label_opacity,
            annotation_x_element,
            annotation_types,
            annotation_std_maxs,
            layout_size,
            title,
            html_file_path,
        )

        return element_predicted_label

    def anneal(
        self,
        w_or_h,
        node_node_score_weight=0,
        element_element_score_weight=0.5,
        node_element_score_weight=0.5,
        n_fraction_node_to_move=1,
        n_fraction_element_to_move=1,
        random_seed=RANDOM_SEED,
        n_iteration=int(1e3),
        initial_temperature=1e-4,
        scale=1e-3,
        triangulate=True,
        print_acceptance=True,
    ):

        if w_or_h == "w":

            if self.w_distance__element_x_element is None:

                self.w_distance__element_x_element = squareform(
                    pdist(
                        self.w.T,
                        metric=compute_information_distance_between_2_1d_arrays,
                    )
                )

            distance__element_x_element = self.w_distance__element_x_element

            if self.w_distance__node_x_element is None:

                distance__node_x_w_element_ = apply_function_on_2_2d_arrays_slices(
                    self.w,
                    diag((1,) * len(self.w_elements)),
                    0,
                    compute_information_distance_between_2_1d_arrays,
                )

                distance__w_element_x_node_ = apply_function_on_2_2d_arrays_slices(
                    self.w,
                    diag((1,) * len(self.nodes)),
                    1,
                    compute_information_distance_between_2_1d_arrays,
                )

                self.w_distance__node_x_element = (
                    distance__node_x_w_element_ + distance__w_element_x_node_.T
                ) / 2

            distance__node_x_element = self.w_distance__node_x_element

            element_x_dimension = self.w_element_x_dimension

        elif w_or_h == "h":

            if self.h_distance__element_x_element is None:

                self.h_distance__element_x_element = squareform(
                    pdist(
                        self.h.T,
                        metric=compute_information_distance_between_2_1d_arrays,
                    )
                )

            distance__element_x_element = self.h_distance__element_x_element

            if self.h_distance__node_x_element is None:

                distance__node_x_h_element_ = apply_function_on_2_2d_arrays_slices(
                    self.h,
                    diag((1,) * len(self.h_elements)),
                    0,
                    compute_information_distance_between_2_1d_arrays,
                )

                distance__h_element_x_node_ = apply_function_on_2_2d_arrays_slices(
                    self.h,
                    diag((1,) * len(self.nodes)),
                    1,
                    compute_information_distance_between_2_1d_arrays,
                )

                self.h_distance__node_x_element = (
                    distance__node_x_h_element_ + distance__h_element_x_node_.T
                ) / 2

            distance__node_x_element = self.h_distance__node_x_element

            element_x_dimension = self.h_element_x_dimension

        target_distance__node_x_node = squareform(self.distance__node_x_node)

        target_distance__element_x_element = squareform(distance__element_x_element)

        target_distance__node_x_element = distance__node_x_element.ravel()

        scores = full((n_iteration, 5), nan)

        node_x_node_score = pearsonr(
            pdist(self.node_x_dimension), target_distance__node_x_node
        )[0]

        element_x_element_score = pearsonr(
            pdist(element_x_dimension), target_distance__element_x_element
        )[0]

        node_x_element_score = pearsonr(
            apply_function_on_2_2d_arrays_slices(
                self.node_x_dimension, element_x_dimension, 0, euclidean
            ).ravel(),
            target_distance__node_x_element,
        )[0]

        fitness = (
            node_x_node_score * node_node_score_weight
            + element_x_element_score * element_element_score_weight
            + node_x_element_score * node_element_score_weight
        )

        n_node = self.distance__node_x_node.shape[0]

        n_node_to_move = int(n_node * n_fraction_node_to_move)

        n_element = distance__element_x_element.shape[0]

        n_element_to_move = int(n_element * n_fraction_element_to_move)

        n_per_print = max(1, n_iteration // 10)

        seed(seed=random_seed)

        for i in range(n_iteration):

            if i % n_per_print == 0:

                print(f"\t{i + 1}/{n_iteration} ...")

            r__node_x_dimension = self.node_x_dimension.copy()

            indices = choice(range(n_node), size=n_node_to_move, replace=True)

            r__node_x_dimension[indices] = normal(
                r__node_x_dimension[indices], scale=scale
            )

            if triangulate:

                n_triangulation = Delaunay(r__node_x_dimension)

            r__element_x_dimension = element_x_dimension.copy()

            for index in choice(range(n_element), size=n_element_to_move, replace=True):

                element_x_y = r__element_x_dimension[index]

                r__element_x_y = normal(element_x_y, scale=scale)

                if triangulate:

                    while n_triangulation.find_simplex(r__element_x_y) == -1:

                        r__element_x_y = normal(element_x_y, scale=scale)

                r__element_x_dimension[index] = r__element_x_y

            r__node_x_node_score = pearsonr(
                pdist(r__node_x_dimension), target_distance__node_x_node
            )[0]

            r__element_x_element_score = pearsonr(
                pdist(r__element_x_dimension), target_distance__element_x_element
            )[0]

            r__node_x_element_score = pearsonr(
                apply_function_on_2_2d_arrays_slices(
                    r__node_x_dimension, r__element_x_dimension, 0, euclidean
                ).ravel(),
                target_distance__node_x_element,
            )[0]

            r__fitness = (
                r__node_x_node_score * node_node_score_weight
                + r__element_x_element_score * element_element_score_weight
                + r__node_x_element_score * node_element_score_weight
            )

            temperature = initial_temperature * (1 - i / (n_iteration + 1))

            if random_sample() < exp((r__fitness - fitness) / temperature):

                if print_acceptance:

                    print(f"\t\t{fitness:.3e} =(accept)=> {r__fitness:.3e} ...")

                self.node_x_dimension = r__node_x_dimension

                element_x_dimension = r__element_x_dimension

                node_x_node_score = r__node_x_node_score

                element_x_element_score = r__element_x_element_score

                node_x_element_score = r__node_x_element_score

                fitness = r__fitness

            scores[i, :] = (
                temperature,
                node_x_node_score,
                element_x_element_score,
                node_x_element_score,
                fitness,
            )

        x = arange(n_iteration)

        if n_iteration < 1e3:

            mode = "markers"

        else:

            mode = "lines"

        plot_and_save(
            {
                "layout": {
                    "title": {"text": "Annealing Summary"},
                    "xaxis": {"title": "Iteration"},
                },
                "data": [
                    {
                        "type": "scatter",
                        "name": "Temperature",
                        "x": x,
                        "y": scores[:, 0],
                        "mode": mode,
                    },
                    {
                        "type": "scatter",
                        "name": "Node-Node",
                        "x": x,
                        "y": scores[:, 1],
                        "mode": mode,
                    },
                    {
                        "type": "scatter",
                        "name": "Element-Element",
                        "x": x,
                        "y": scores[:, 2],
                        "mode": mode,
                    },
                    {
                        "type": "scatter",
                        "name": "Node-Element",
                        "x": x,
                        "y": scores[:, 3],
                        "mode": mode,
                    },
                    {
                        "type": "scatter",
                        "name": "Fitness",
                        "x": x,
                        "y": scores[:, 4],
                        "mode": mode,
                    },
                ],
            },
            None,
        )

        if w_or_h == "w":

            self.w_element_x_dimension = element_x_dimension

            element_label = self.w_element_label

            bandwidth_factor = self.w_bandwidth_factor

        elif w_or_h == "h":

            self.h_element_x_dimension = element_x_dimension

            element_label = self.h_element_label

            bandwidth_factor = self.h_bandwidth_factor

        if element_label is not None:

            self.set_element_label(
                w_or_h,
                element_label,
                n_grid=self.n_grid,
                bandwidth_factor=bandwidth_factor,
                plot=False,
            )
