from gzip import open as gzip_open
from pickle import dump, load

from numpy import apply_along_axis, full, nan
from pandas import DataFrame
from scipy.spatial import Delaunay

from .array import normalize
from .CONSTANT import RANDOM_SEED
from .kernel_density import get_bandwidth
from .plot import plot_heat_map
from .point import map_point, plot_node_point, pull_point
from .point_x_dimension import get_grid_1ds, make_grid_1d, shape
from .probability import get_probability


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
            point_group=self.point_label,
            group_colorscale=self.point_label_colorscale,
            grid_1d=self.dimension_grid,
            grid_nd_probabilities=self.grid_probability,
            grid_nd_group=self.grid_label,
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

        self.dimension_grid = make_grid_1d(0, 1, 1e-3, n_grid)

        for i in range(n_grid):

            for j in range(n_grid):

                mask_grid[i, j] = triangulation.find_simplex(
                    (self.dimension_grid[i], self.dimension_grid[j])
                )

        label_grid_probability = {}

        bandwidths = tuple(self.point_x_dimension.apply(get_bandwidth))

        grids = (self.dimension_grid,) * 2

        for label in self.point_label.unique():

            grid_point_x_dimension, point_pdf = get_probability(
                self.point_x_dimension[self.point_label == label].values,
                plot=False,
                bandwidths=bandwidths,
                grid_1ds=grids,
            )

            label_grid_probability[label] = shape(
                point_pdf, get_grid_1ds(grid_point_x_dimension)
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
            point_group=None,
            group_colorscale=self.point_label_colorscale,
            grid_1d=self.dimension_grid,
            grid_nd_probabilities=self.grid_probability,
            grid_nd_group=self.grid_label,
            **plot_gps_map_keyword_arguments,
        )


def read_gps_map(pickle_gz_file_path):

    with gzip_open(pickle_gz_file_path) as io:

        return load(io)


def write_gps_map(pickle_gz_file_path, gps_map):

    with gzip_open(pickle_gz_file_path, mode="wb") as io:

        dump(gps_map, io)
