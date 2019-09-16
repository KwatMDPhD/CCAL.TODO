from numpy import full, linspace, nan, rot90
from pandas import DataFrame, Index
from scipy.spatial import Delaunay
from scipy.spatial.distance import pdist, squareform

from .cluster_matrix import cluster_matrix
from .compute_element_x_dimension_joint_probability import (
    compute_element_x_dimension_joint_probability,
)
from .compute_information_distance_between_2_vectors import (
    compute_information_distance_between_2_vectors,
)
from .compute_vector_bandwidth import compute_vector_bandwidth
from .DATA_TYPE_COLORSCALE import DATA_TYPE_COLORSCALE
from .make_element_x_dimension_from_element_x_node_and_node_x_dimension import (
    make_element_x_dimension_from_element_x_node_and_node_x_dimension,
)
from .normalize_dataframe import normalize_dataframe
from .plot_gps_map import plot_gps_map
from .plot_heat_map import plot_heat_map
from .RANDOM_SEED import RANDOM_SEED
from .scale_element_x_dimension_dimension import scale_element_x_dimension_dimension
from .unmesh import unmesh


class GPSMap:
    def __init__(
        self,
        element_x_node,
        distance__node_x_node=None,
        distance_function=compute_information_distance_between_2_vectors,
        mds_random_seed=RANDOM_SEED,
        n_pull=None,
        pull_power=None,
    ):

        self.element_x_node = element_x_node

        if distance__node_x_node is None:

            distance__node_x_node = squareform(
                pdist(self.element_x_node.values.T, metric=distance_function)
            )

            plot_heat_map(
                DataFrame(
                    distance__node_x_node,
                    index=self.element_x_node.columns,
                    columns=self.element_x_node.columns,
                ).iloc[
                    cluster_matrix(distance__node_x_node, 0),
                    cluster_matrix(distance__node_x_node, 1),
                ],
                layout={
                    "height": 700,
                    "width": 700,
                    "title": {"text": distance_function.__name__},
                },
            )

        self.node_x_dimension = normalize_dataframe(
            DataFrame(
                scale_element_x_dimension_dimension(
                    2,
                    distance__point_x_point=distance__node_x_node,
                    random_seed=mds_random_seed,
                ),
                index=self.element_x_node.columns,
                columns=Index(("x", "y"), name="Axis"),
            ),
            0,
            "0-1",
        )

        self.element_x_dimension = DataFrame(
            make_element_x_dimension_from_element_x_node_and_node_x_dimension(
                self.element_x_node.values,
                self.node_x_dimension.values,
                n_pull,
                pull_power,
            ),
            index=self.element_x_node.index,
            columns=self.node_x_dimension.columns,
        )

        self.element_label = None

        self.dimension_grid = None

        self.grid_probability = None

        self.grid_label = None

        self.label_colorscale = None

    def plot(self, **plot_gps_map_keyword_arguments):

        plot_gps_map(
            self.node_x_dimension,
            self.element_x_dimension,
            element_label=self.element_label,
            dimension_grid=self.dimension_grid,
            grid_probability=self.grid_probability,
            grid_label=self.grid_label,
            label_colorscale=self.label_colorscale,
            **plot_gps_map_keyword_arguments,
        )

    def set_element_label(
        self,
        element_label,
        n_grid=128,
        bandwidth_factor=1,
        label_colorscale=None,
        plot=True,
    ):

        assert 3 <= element_label.value_counts().min()

        assert not element_label.isna().any()

        self.element_label = element_label

        grid_shape = (n_grid,) * 2

        mask_grid = full(grid_shape, nan)

        triangulation = Delaunay(self.node_x_dimension)

        self.dimension_grid = linspace(-0.001, 1.001, num=n_grid)

        for i in range(mask_grid.shape[0]):

            for j in range(mask_grid.shape[1]):

                mask_grid[i, j] = triangulation.find_simplex(
                    (self.dimension_grid[j], self.dimension_grid[-i])
                )

        label_grid_probability = {}

        dimension_bandwidths = tuple(
            compute_vector_bandwidth(coordinate.values)
            for axis, coordinate in self.element_x_dimension.items()
        )

        for label in element_label.unique():

            label_grid_probability[label] = rot90(
                unmesh(
                    *compute_element_x_dimension_joint_probability(
                        self.element_x_dimension[element_label == label].values,
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

        self.grid_probability = full(grid_shape, nan)

        self.grid_label = full(grid_shape, nan)

        for i in range(mask_grid.shape[0]):

            for j in range(mask_grid.shape[1]):

                if mask_grid[i, j] == -1:

                    continue

                max_probability = 0

                max_label = nan

                for label, grid_probability_ in label_grid_probability.items():

                    probability = grid_probability_[i, j]

                    if max_probability < probability:

                        max_probability = probability

                        max_label = label

                self.grid_probability[i, j] = max_probability

                self.grid_label[i, j] = max_label

        if label_colorscale is None:

            label_colorscale = DATA_TYPE_COLORSCALE["categorical"]

        self.label_colorscale = label_colorscale

        plot_heat_map(
            normalize_dataframe(self.element_x_node, 1, "-0-"),
            row_annotations=self.element_label,
            row_annotation_colorscale=self.label_colorscale,
        )

    def predict(
        self,
        new_element_x_node,
        n_pull=None,
        pull_power=None,
        **plot_gps_map_keyword_arguments,
    ):

        plot_gps_map(
            self.node_x_dimension,
            DataFrame(
                make_element_x_dimension_from_element_x_node_and_node_x_dimension(
                    new_element_x_node.values,
                    self.node_x_dimension.values,
                    n_pull,
                    pull_power,
                ),
                index=new_element_x_node.index,
                columns=self.node_x_dimension.columns,
            ),
            element_label=None,
            dimension_grid=self.dimension_grid,
            grid_probability=self.grid_probability,
            grid_label=self.grid_label,
            label_colorscale=self.label_colorscale,
            **plot_gps_map_keyword_arguments,
        )
