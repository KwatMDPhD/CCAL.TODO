from numpy import arange, diag, exp, full, linspace, mean, nan, rot90, asarray, integer
from .normalize_array_on_axis import normalize_array_on_axis
from .get_colorscale_color import get_colorscale_color
from pandas import value_counts
from .DATA_TYPE_COLORSCALE import DATA_TYPE_COLORSCALE

from numpy.random import choice, normal, random_sample, seed
from pandas import DataFrame, Series
from scipy.spatial import Delaunay
from scipy.spatial.distance import euclidean, pdist, squareform
from scipy.stats import pearsonr

from .apply_function_on_slices_from_2_matrices import (
    apply_function_on_slices_from_2_matrices,
)
from .check_dataframe_number import check_dataframe_number
from .cluster_matrix import cluster_matrix
from .compute_element_x_dimension_joint_probability import (
    compute_element_x_dimension_joint_probability,
)
from .compute_information_distance_between_2_vectors import (
    compute_information_distance_between_2_vectors,
)
from .compute_vector_bandwidth import compute_vector_bandwidth
from .make_element_x_dimension_from_node_x_element_and_node_dimension import (
    make_element_x_dimension_from_node_x_element_and_node_dimension,
)

from .normalize_dataframe import normalize_dataframe
from .plot_gps_map import plot_gps_map
from .plot_heat_map import plot_heat_map
from .plot_plotly_figure import plot_plotly_figure
from .RANDOM_SEED import RANDOM_SEED
from .scale_element_x_dimension_dimension import scale_element_x_dimension_dimension
from .train_and_classify import train_and_classify
from .unmesh import unmesh


class GPSMap:
    def __init__(
        self,
        w=None,
        h=None,
        function_to_blend_node_node_distance=mean,
        mds_random_seed=RANDOM_SEED,
        w_n_pull=None,
        w_pull_power=None,
        h_n_pull=None,
        h_pull_power=None,
        plot=True,
    ):
        self.nodes = None
        self.node_name = None

        self.w = None
        self.w_elements = None
        self.w_element_name = None
        self.w_distance__node_x_node = None
        self.h = None
        self.h_elements = None
        self.h_element_name = None
        self.h_distance__node_x_node = None

        self.distance__node_x_node = None

        self.triangulation = None

        self.w_n_pull = w_n_pull
        self.w_pull_power = w_pull_power
        self.w_element_x_dimension = None
        self.h_n_pull = h_n_pull
        self.h_pull_power = h_pull_power
        self.h_element_x_dimension = None

        self.n_grid = None
        self.dimension_grid = None
        self.mask_grid = None

        self.w_element_label = None
        self.w_bandwidth_factor = None
        self.w_grid_values = None
        self.w_grid_labels = None
        self.w_label_colorscale = None
        self.h_element_label = None
        self.h_bandwidth_factor = None
        self.h_grid_values = None
        self.h_grid_labels = None
        self.h_label_colorscale = None

        if w is not None:
            check_dataframe_number(w)
        if h is not None:
            check_dataframe_number(h)

        if w is not None and h is not None:
            assert (w.index == h.index).all()
            assert w.index.name == h.index.name
            self.nodes = w.index.tolist()
            self.node_name = w.index.name
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
                pdist(self.w, metric=compute_information_distance_between_2_vectors)
            )
            if plot:
                plot_heat_map(
                    DataFrame(self.w, index=self.nodes, columns=self.w_elements).iloc[
                        :, cluster_matrix(self.w, 1)
                    ],
                    layout={
                        "title": {"text": "W"},
                        "xaxis": {"title": {"text": self.w_element_name}},
                        "yaxis": {"title": {"text": self.node_name}},
                    },
                )
                plot_heat_map(
                    DataFrame(
                        self.w_distance__node_x_node,
                        index=self.nodes,
                        columns=self.nodes,
                    ).iloc[
                        cluster_matrix(self.w_distance__node_x_node, 0),
                        cluster_matrix(self.w_distance__node_x_node, 1),
                    ],
                    layout={
                        "title": {
                            "text": "{0}-{0} Distance in W".format(self.node_name)
                        },
                        "xaxis": {"title": {"text": self.node_name}},
                        "yaxis": {"title": {"text": self.node_name}},
                    },
                )

        if h is not None:
            self.h = h.values
            self.h_elements = h.columns.tolist()
            self.h_element_name = h.columns.name
            self.h_distance__node_x_node = squareform(
                pdist(self.h, metric=compute_information_distance_between_2_vectors)
            )
            if plot:
                plot_heat_map(
                    DataFrame(self.h, index=self.nodes, columns=self.h_elements).iloc[
                        :, cluster_matrix(self.h, 1)
                    ],
                    layout={
                        "title": {"text": "H"},
                        "xaxis": {"title": {"text": self.h_element_name}},
                        "yaxis": {"title": {"text": self.node_name}},
                    },
                )
                plot_heat_map(
                    DataFrame(
                        self.h_distance__node_x_node,
                        index=self.nodes,
                        columns=self.nodes,
                    ).iloc[
                        cluster_matrix(self.h_distance__node_x_node, 0),
                        cluster_matrix(self.h_distance__node_x_node, 1),
                    ],
                    layout={
                        "title": {
                            "text": "{0}-{0} Distance in H".format(self.node_name)
                        },
                        "xaxis": {"title": {"text": self.node_name}},
                        "yaxis": {"title": {"text": self.node_name}},
                    },
                )

        if w is not None and h is not None:
            n_node = len(self.nodes)
            self.distance__node_x_node = full((n_node,) * 2, nan)
            for i in range(n_node):
                for j in range(n_node):
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
                        cluster_matrix(self.distance__node_x_node, 0),
                        cluster_matrix(self.distance__node_x_node, 1),
                    ],
                    layout={
                        "title": {
                            "text": "{0}-{0} Distance in W and H".format(self.node_name)
                        },
                        "xaxis": {"title": {"text": self.node_name}},
                        "yaxis": {"title": {"text": self.node_name}},
                    },
                )
        elif w is not None:
            self.distance__node_x_node = self.w_distance__node_x_node
        elif h is not None:
            self.distance__node_x_node = self.h_distance__node_x_node

        self.node_x_dimension = normalize_array_on_axis(
            scale_element_x_dimension_dimension(
                2,
                distance__point_x_point=self.distance__node_x_node,
                random_seed=mds_random_seed,
            ),
            0,
            "0-1",
        )

        self.triangulation = Delaunay(self.node_x_dimension)

        if w is not None:
            self.w_element_x_dimension = make_element_x_dimension_from_node_x_element_and_node_dimension(
                self.w, self.node_x_dimension, self.w_n_pull, self.w_pull_power
            )
        if h is not None:
            self.h_element_x_dimension = make_element_x_dimension_from_node_x_element_and_node_dimension(
                self.h, self.node_x_dimension, self.h_n_pull, self.h_pull_power
            )

    def plot(self, w_or_h, **plot_gps_map_keyword_arguments):

        if w_or_h == "w":
            elements = self.w_elements
            element_name = self.w_element_name
            element_x_dimension = self.w_element_x_dimension
            element_label = self.w_element_label
            grid_values = self.w_grid_values
            grid_labels = self.w_grid_labels
            label_colorscale = self.w_label_colorscale

        elif w_or_h == "h":
            elements = self.h_elements
            element_name = self.h_element_name
            element_x_dimension = self.h_element_x_dimension
            element_label = self.h_element_label
            grid_values = self.h_grid_values
            grid_labels = self.h_grid_labels
            label_colorscale = self.h_label_colorscale

        plot_gps_map(
            self.nodes,
            self.node_name,
            self.node_x_dimension,
            elements,
            element_name,
            element_x_dimension,
            dimension_grid=self.dimension_grid,
            element_label=element_label,
            grid_values=grid_values,
            grid_labels=grid_labels,
            label_colorscale=label_colorscale,
            **plot_gps_map_keyword_arguments,
        )

    def set_element_label(
        self,
        w_or_h,
        element_label,
        n_grid=128,
        bandwidth_factor=1,
        label_colorscale=None,
        plot=True,
    ):
        assert all(isinstance(label, (int, integer)) for label in element_label)
        assert all(3 <= value_counts(element_label))
        element_label = tuple(int(label) for label in element_label)

        if w_or_h == "w":
            element_x_dimension = self.w_element_x_dimension
        elif w_or_h == "h":
            element_x_dimension = self.h_element_x_dimension

        self.n_grid = n_grid
        grid_shape = (self.n_grid,) * 2
        self.mask_grid = full(grid_shape, nan)
        dimension_grid_min = 0 - 1e-3
        dimension_grid_max = 1 + 1e-3
        self.dimension_grid = linspace(
            dimension_grid_min, dimension_grid_max, num=self.n_grid
        )
        for i in range(self.n_grid):
            for j in range(self.n_grid):
                self.mask_grid[i, j] = (
                    self.triangulation.find_simplex(
                        (self.dimension_grid[j], self.dimension_grid[-i])
                    )
                    == -1
                )

        label_grid_probabilities = {}
        dimension_bandwidths = tuple(
            compute_vector_bandwidth(element_x_dimension[:, i]) for i in range(2)
        )
        element_label_set_sorted = sorted(set(element_label))
        for label in element_label_set_sorted:
            label_grid_probabilities[label] = rot90(
                unmesh(
                    *compute_element_x_dimension_joint_probability(
                        element_x_dimension[asarray(element_label) == label],
                        plot=False,
                        dimension_bandwidths=dimension_bandwidths,
                        dimension_bandwidth_factors=(bandwidth_factor,) * 2,
                        dimension_grid_mins=(dimension_grid_min,) * 2,
                        dimension_grid_maxs=(dimension_grid_max,) * 2,
                        dimension_fraction_grid_extensions=(0,) * 2,
                        dimension_n_grids=grid_shape,
                    )
                )[1]
            )
        grid_values = full(grid_shape, nan)
        grid_labels = full(grid_shape, nan)
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

        if label_colorscale is None:
            label_colorscale = DATA_TYPE_COLORSCALE["categorical"]

        if w_or_h == "w":
            self.w_element_label = element_label
            self.w_bandwidth_factor = bandwidth_factor
            self.w_grid_values = grid_values
            self.w_grid_labels = grid_labels
            self.w_label_colorscale = label_colorscale

        elif w_or_h == "h":
            self.h_element_label = element_label
            self.h_bandwidth_factor = bandwidth_factor
            self.h_grid_values = grid_values
            self.h_grid_labels = grid_labels
            self.h_label_colorscale = label_colorscale

        if plot:
            if w_or_h == "w":
                dataframe = DataFrame(self.w, index=self.nodes, columns=self.w_elements)
                column_annotation = self.w_element_label
                element_name = self.w_element_name

            elif w_or_h == "h":
                dataframe = DataFrame(self.h, index=self.nodes, columns=self.h_elements)
                column_annotation = self.h_element_label
                element_name = self.h_element_name

            plot_heat_map(
                normalize_dataframe(dataframe, 0, "-0-"),
                column_annotations=column_annotation,
                column_annotation_colorscale=label_colorscale,
                layout={
                    "title": {"text": w_or_h.title()},
                    "xaxis": {"title": {"text": element_name}},
                    "yaxis": {"title": {"text": self.node_name}},
                },
            )

    # def predict(
    #     self,
    #     w_or_h,
    #     node_x_predicting_element,
    #     support_vector_parameter_c=1e3,
    #     n_pull=None,
    #     pull_power=None,
    #     annotation_x_element=None,
    #     annotation_types=None,
    #     annotation_std_maxs=None,
    #     annotation_colorscales=None,
    #     element_marker_size=element_marker_size,
    #     layout_size=880,
    #     highlight_binary=False,
    #     title=None,
    #     html_file_path=None,
    # ):

    #     check_dataframe_number(node_x_predicting_element)

    #     predicting_elements = node_x_predicting_element.columns.tolist()

    #     if w_or_h == "w":

    #         node_x_element = self.w

    #         element_name = self.w_element_name

    #         if n_pull is None:

    #             n_pull = self.w_n_pull

    #         if pull_power is None:

    #             pull_power = self.w_pull_power

    #         element_label = self.w_element_label

    #         grid_values = self.w_grid_values

    #         grid_labels = self.w_grid_labels

    #         label_colorscale = self.w_label_colorscale

    #     elif w_or_h == "h":

    #         node_x_element = self.h

    #         element_name = self.h_element_name

    #         if n_pull is None:

    #             n_pull = self.h_n_pull

    #         if pull_power is None:

    #             pull_power = self.h_pull_power

    #         element_label = self.h_element_label

    #         grid_values = self.h_grid_values

    #         grid_labels = self.h_grid_labels

    #         label_colorscale = self.h_label_colorscale

    #     predicting_element_x_dimension = make_element_x_dimension_from_node_x_element_and_node_dimension(
    #         node_x_predicting_element.values, self.node_x_dimension, n_pull, pull_power
    #     )

    #     if element_label is not None:

    #         element_predicted_label = Series(
    #             train_and_classify(
    #                 node_x_element.T,
    #                 element_label,
    #                 node_x_predicting_element.T,
    #                 c=support_vector_parameter_c,
    #                 tol=1e-8,
    #             ),
    #             name="Predicted {} Label".format(element_name),
    #             index=predicting_elements,
    #         )

    #     else:

    #         element_predicted_label = None

    #     element_predicted_label = None

    #     if annotation_x_element is not None:

    #         annotation_x_element = annotation_x_element.reindex(
    #             columns=predicting_elements
    #         )

    #     plot_gps_map(
    #         self.nodes,
    #         self.node_name,
    #         self.node_x_dimension,
    #         predicting_elements,
    #         element_name,
    #         predicting_element_x_dimension,
    #         element_marker_size,
    #         element_predicted_label,
    # self.dimension_grid,
    #         grid_values,
    #         grid_labels,
    #         label_colorscale,
    # grid_label_opacity,
    #         annotation_x_element,
    #         annotation_types,
    #         annotation_std_maxs,
    #         annotation_colorscales,
    #         highlight_binary,
    #         layout,
    #         html_file_path,
    #     )

    #     return element_predicted_label
