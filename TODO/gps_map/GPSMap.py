from numpy import full, linspace, nan
from pandas import DataFrame
from scipy.spatial import Delaunay

from .compute_bandwidth import compute_bandwidth
from .compute_joint_probability import compute_joint_probability
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
        node_x_node_distance,
        mds_random_seed=RANDOM_SEED,
        node_marker_size=16,
    ):

        self.element_x_node = element_x_node

        self.node_x_dimension = normalize_dataframe(
            DataFrame(
                scale_element_x_dimension_dimension(
                    2,
                    point_x_point_distance=node_x_node_distance,
                    random_seed=mds_random_seed,
                ),
                index=self.element_x_node.columns,
            ),
            0,
            "0-1",
        )

        self.node_marker_size = node_marker_size

        self.element_x_dimension = DataFrame(
            make_element_x_dimension_from_element_x_node_and_node_x_dimension(
                self.element_x_node.values, self.node_x_dimension.values
            ),
            index=self.element_x_node.index,
            columns=self.node_x_dimension.columns,
        )

        self.element_label = None

        self.dimension_grid = None

        self.grid_probability = None

        self.grid_label = None

        self.element_label_colorscale = None

    def plot(self, **plot_gps_map_keyword_arguments):

        plot_gps_map(
            self.node_x_dimension,
            self.element_x_dimension,
            node_marker_size=self.node_marker_size,
            element_label=self.element_label,
            dimension_grid=self.dimension_grid,
            grid_probability=self.grid_probability,
            grid_label=self.grid_label,
            element_label_colorscale=self.element_label_colorscale,
            **plot_gps_map_keyword_arguments,
        )

    def set_element_label(
        self,
        element_label,
        element_label_colorscale=None,
        n_grid=64,
        bandwidth_factor=1,
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
                    (self.dimension_grid[i], self.dimension_grid[j])
                )

        label_grid_probability = {}

        dimension_bandwidths = tuple(
            compute_bandwidth(coordinate.values)
            for axis, coordinate in self.element_x_dimension.items()
        )

        for label in self.element_label.unique():

            label_grid_probability[label] = unmesh(
                *compute_joint_probability(
                    self.element_x_dimension[self.element_label == label].values,
                    plot=False,
                    dimension_bandwidths=dimension_bandwidths,
                    dimension_bandwidth_factors=(bandwidth_factor,) * 2,
                    dimension_grid_mins=(self.dimension_grid.min(),) * 2,
                    dimension_grid_maxs=(self.dimension_grid.max(),) * 2,
                    dimension_fraction_grid_extensions=(0,) * 2,
                    dimension_n_grids=grid_shape,
                )
            )[1]

        self.grid_probability = full(grid_shape, nan)

        self.grid_label = full(grid_shape, nan)

        for i in range(mask_grid.shape[0]):

            for j in range(mask_grid.shape[1]):

                if mask_grid[i, j] != -1:

                    max_probability = 0

                    max_label = nan

                    for label, grid_probability_ in label_grid_probability.items():

                        probability = grid_probability_[i, j]

                        if max_probability < probability:

                            max_probability = probability

                            max_label = label

                    self.grid_probability[i, j] = max_probability

                    self.grid_label[i, j] = max_label

        self.element_label_colorscale = element_label_colorscale

        plot_heat_map(
            normalize_dataframe(self.element_x_node, 1, "-0-").T,
            column_annotations=self.element_label,
            column_annotation_colorscale=self.element_label_colorscale,
        )

    def predict(self, new_element_x_node, **plot_gps_map_keyword_arguments):

        plot_gps_map(
            self.node_x_dimension,
            DataFrame(
                make_element_x_dimension_from_element_x_node_and_node_x_dimension(
                    new_element_x_node.values, self.node_x_dimension.values
                ),
                index=new_element_x_node.index,
                columns=self.node_x_dimension.columns,
            ),
            node_marker_size=self.node_marker_size,
            element_label=None,
            dimension_grid=self.dimension_grid,
            grid_probability=self.grid_probability,
            grid_label=self.grid_label,
            element_label_colorscale=self.element_label_colorscale,
            **plot_gps_map_keyword_arguments,
        )
