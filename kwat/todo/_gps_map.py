from gzip import open as gzip_open
from pickle import dump, load

from numpy import full, nan, unique
from scipy.spatial import Delaunay

from .CONSTANT import RANDOM_SEED
from .grid import make_g1
from .kernel_density import get_bandwidth
from .plot import CATEGORICAL_COLORSCALE, plot_heat_map
from .point import map_point, plot_node_point, pull_point
from .probability import get_probability


class GPSMap:
    def __init__(
        self,
        node_name,
        node_,
        node_x_node,
        point_name,
        point_,
        point_x_node,
        node_marker_size=24,
        random_seed=RANDOM_SEED,
    ):

        self.node_name = node_name

        self.node_ = node_

        self.node_x_dimension = map_point(node_x_node, 2, random_seed=random_seed)

        self.point_name = point_name

        self.point_ = point_

        self.point_x_node = point_x_node

        self.point_x_dimension = pull_point(self.node_x_dimension, self.point_x_node)

        self.node_marker_size = node_marker_size

        self.group_ = None

        self._1d_grid = None

        self.nd_probability_vector = None

        self.nd_group_vector = None

        self.group_colorscale = None

    def plot(self, **kwarg_):

        plot_node_point(
            self.node_name,
            self.node_,
            self.node_x_dimension,
            self.point_name,
            self.point_,
            self.point_x_dimension,
            group_=self.group_,
            group_colorscale=self.group_colorscale,
            _1d_grid=self._1d_grid,
            nd_probability_vector=self.nd_probability_vector,
            nd_group_vector=self.nd_group_vector,
            node_trace={"marker": {"size": self.node_marker_size}},
            **kwarg_,
        )

    def set_group(self, group_, group_colorscale=CATEGORICAL_COLORSCALE, grid_n=128):

        if group_ == "closest_node":

            group_ = self.point_x_node.argmax(axis=1)

        self.group_ = group_

        self.group_colorscale = group_colorscale

        shape = (grid_n,) * 2

        # TODO: refactor

        mask = full(shape, nan)

        triangulation = Delaunay(self.node_x_dimension)

        self._1d_grid = make_g1(0, 1, 1e-3, grid_n)

        for index_0 in range(grid_n):

            for index_1 in range(grid_n):

                mask[index_0, index_1] = triangulation.find_simplex(
                    (self._1d_grid[index_0], self._1d_grid[index_1])
                )

        group_to_nd_probability_vector = {}

        bandwidth_ = tuple(get_bandwidth(vector) for vector in self.point_x_dimension.T)

        _1d_grid_ = (self._1d_grid,) * 2

        for group in unique(self.group_):

            group_to_nd_probability_vector[group] = get_probability(
                self.point_x_dimension[self.group_ == group],
                plot=False,
                bandwidth_=bandwidth_,
                _1d_grid_=_1d_grid_,
            )[1].reshape(shape)

        self.nd_probability_vector = full(shape, nan)

        self.nd_group_vector = full(shape, nan)

        for index_0 in range(grid_n):

            for index_1 in range(grid_n):

                if mask[index_0, index_1] != -1:

                    best_probability = 0

                    best_group = nan

                    for (
                        group,
                        nd_probability_vector,
                    ) in group_to_nd_probability_vector.items():

                        probability = nd_probability_vector[index_0, index_1]

                        if best_probability < probability:

                            best_probability = probability

                            best_group = group

                    self.nd_probability_vector[index_0, index_1] = best_probability

                    self.nd_group_vector[index_0, index_1] = best_group

        plot_heat_map(
            self.point_x_node.T,
            self.node_,
            self.point_,
            self.node_name,
            self.point_name,
            axis_1_group_=self.group_,
            axis_1_group_colorscale=self.group_colorscale,
            layout={"yaxis": {"dtick": 1}},
        )

    def predict(self, new_point_name, new_point_, new_point_x_node, **kwarg_):

        plot_node_point(
            self.node_name,
            self.node_,
            self.node_x_dimension,
            new_point_name,
            new_point_,
            pull_point(self.node_x_dimension, new_point_x_node),
            group_=None,
            group_colorscale=self.group_colorscale,
            _1d_grid=self._1d_grid,
            nd_probability_vector=self.nd_probability_vector,
            nd_group_vector=self.nd_group_vector,
            node_trace={"marker": {"size": self.node_marker_size}},
            **kwarg_,
        )


def read(
    file_path,
):

    with gzip_open(file_path) as io:

        return load(io)


def write(file_path, gps_map):

    with gzip_open(file_path, mode="wb") as io:

        dump(gps_map, io)
