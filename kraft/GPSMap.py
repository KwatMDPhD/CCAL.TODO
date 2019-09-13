from numpy import asarray, full, linspace, mean, nan, rot90
from pandas import DataFrame
from scipy.spatial import Delaunay
from scipy.spatial.distance import pdist, squareform

from .check_dataframe_number import check_dataframe_number
from .cluster_matrix import cluster_matrix
from .compute_element_x_dimension_joint_probability import (
    compute_element_x_dimension_joint_probability,
)
from .compute_information_distance_between_2_vectors import (
    compute_information_distance_between_2_vectors,
)
from .compute_vector_bandwidth import compute_vector_bandwidth
from .DATA_TYPE_COLORSCALE import DATA_TYPE_COLORSCALE
from .make_element_x_dimension_from_node_x_element_and_node_dimension import (
    make_element_x_dimension_from_node_x_element_and_node_dimension,
)
from .normalize_array_on_axis import normalize_array_on_axis
from .normalize_dataframe import normalize_dataframe
from .plot_gps_map import plot_gps_map
from .plot_heat_map import plot_heat_map
from .RANDOM_SEED import RANDOM_SEED
from .scale_element_x_dimension_dimension import scale_element_x_dimension_dimension
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

        self.w_element_label = None

        self.w_bandwidth_factor = None

        self.w_grid_probability = None

        self.w_grid_label = None

        self.w_label_colorscale = None

        self.h_element_label = None

        self.h_bandwidth_factor = None

        self.h_grid_probability = None

        self.h_grid_label = None

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

        if plot:

            heat_map_axis = {"title": {"text": self.node_name}}

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
                        "yaxis": heat_map_axis,
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
                        "xaxis": heat_map_axis,
                        "yaxis": heat_map_axis,
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
                        "yaxis": heat_map_axis,
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
                        "xaxis": heat_map_axis,
                        "yaxis": heat_map_axis,
                    },
                )

        if w is not None and h is not None:

            self.distance__node_x_node = full((len(self.nodes),) * 2, nan)

            for i in range(self.distance__node_x_node.shape[0]):

                for j in range(self.distance__node_x_node.shape[1]):

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
                        "xaxis": heat_map_axis,
                        "yaxis": heat_map_axis,
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

            grid_probability = self.w_grid_probability

            grid_label = self.w_grid_label

            label_colorscale = self.w_label_colorscale

        elif w_or_h == "h":

            elements = self.h_elements

            element_name = self.h_element_name

            element_x_dimension = self.h_element_x_dimension

            element_label = self.h_element_label

            grid_probability = self.h_grid_probability

            grid_label = self.h_grid_label

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
            grid_probability=grid_probability,
            grid_label=grid_label,
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

        assert 3 <= element_label.value_counts().min()

        if w_or_h == "w":

            element_x_dimension = self.w_element_x_dimension

        elif w_or_h == "h":

            element_x_dimension = self.h_element_x_dimension

        self.n_grid = n_grid

        grid_shape = (self.n_grid,) * 2

        mask_grid = full(grid_shape, nan)

        self.dimension_grid = linspace(-1e-3, 1 + 1e-3, num=self.n_grid)

        for i in mask_grid.shape[0]:

            for j in mask_grid.shape[1]:

                mask_grid[i, j] = self.triangulation.find_simplex(
                    (self.dimension_grid[j], self.dimension_grid[-i])
                )

        label_grid_probability = {}

        dimension_bandwidths = tuple(
            compute_vector_bandwidth(element_x_dimension[:, i]) for i in range(2)
        )

        for label in set(element_label):

            label_grid_probability[label] = rot90(
                unmesh(
                    *compute_element_x_dimension_joint_probability(
                        element_x_dimension[asarray(element_label) == label],
                        plot=False,
                        dimension_bandwidths=dimension_bandwidths,
                        dimension_bandwidth_factors=(bandwidth_factor,) * 2,
                        dimension_grid_mins=(self.dimension_grid.min(),) * 2,
                        dimension_grid_maxs=(self.dimension_grid.max(),) * 2,
                        dimension_fraction_grid_extensions=(0,) * 2,
                        dimension_n_grids=grid_shape,
                    )
                )[1]
            )

        grid_probability = full(grid_shape, nan)

        grid_label = full(grid_shape, nan)

        for i in mask_grid.shape[0]:

            for j in mask_grid.shape[1]:

                if mask_grid[i, j] == -1:

                    continue

                max_probability = 0

                max_label = nan

                for label, grid_probability in label_grid_probability.items():

                    probability = grid_probability[i, j]

                    if max_probability < probability:

                        max_probability = probability

                        max_label = label

                grid_probability[i, j] = max_probability

                grid_label[i, j] = max_label

        if label_colorscale is None:

            label_colorscale = DATA_TYPE_COLORSCALE["categorical"]

        if w_or_h == "w":

            self.w_element_label = element_label

            self.w_bandwidth_factor = bandwidth_factor

            self.w_grid_probability = grid_probability

            self.w_grid_label = grid_label

            self.w_label_colorscale = label_colorscale

        elif w_or_h == "h":

            self.h_element_label = element_label

            self.h_bandwidth_factor = bandwidth_factor

            self.h_grid_probability = grid_probability

            self.h_grid_label = grid_label

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

    def predict(
        self,
        w_or_h,
        node_x_predicting_element,
        n_pull=None,
        pull_power=None,
        **plot_gps_map_keyword_arguments,
    ):

        check_dataframe_number(node_x_predicting_element)

        if w_or_h == "w":

            element_name = self.w_element_name

            if n_pull is None:

                n_pull = self.w_n_pull

            if pull_power is None:

                pull_power = self.w_pull_power

            grid_probability = self.w_grid_probability

            grid_label = self.w_grid_label

            label_colorscale = self.w_label_colorscale

        elif w_or_h == "h":

            element_name = self.h_element_name

            if n_pull is None:

                n_pull = self.h_n_pull

            if pull_power is None:

                pull_power = self.h_pull_power

            grid_probability = self.h_grid_probability

            grid_label = self.h_grid_label

            label_colorscale = self.h_label_colorscale

        plot_gps_map(
            self.nodes,
            self.node_name,
            self.node_x_dimension,
            node_x_predicting_element.columns.tolist(),
            "Predicting {}".format(element_name),
            make_element_x_dimension_from_node_x_element_and_node_dimension(
                node_x_predicting_element.values,
                self.node_x_dimension,
                n_pull,
                pull_power,
            ),
            dimension_grid=self.dimension_grid,
            grid_probability=grid_probability,
            grid_label=grid_label,
            label_colorscale=label_colorscale,
            **plot_gps_map_keyword_arguments,
        )
