from gzip import open as gzip_open
from pickle import dump, load

from numpy import apply_along_axis, full, nan
from scipy.spatial import Delaunay

from .array import normalize
from .CONSTANT import RANDOM_SEED
from .grid import make_1d
from .kernel_density import get_bandwidth
from .plot import plot_heat_map
from .point import map_point, plot_node_point, pull_point
from .probability import get_probability


class GPSMap:
    def __init__(
        self,
        node_,
        node_name,
        node_x_node_distance,
        point_,
        point_name,
        point_x_node_pull,
        random_seed=RANDOM_SEED,
    ):

        self.node_ = node_

        self.node_name = node_name

        self.node_x_dimension = map_point(
            node_x_node_distance, 2, random_seed=random_seed
        )

        self.point_ = point_

        self.point_name = point_name

        self.point_x_node_pull = point_x_node_pull

        self.point_x_dimension = pull_point(
            self.node_x_dimension, self.point_x_node_pull
        )

        self.group_ = None

        self._1d_grid = None

        self.nd_probability_vector = None

        self.nd_group_vector = None

        self.group_colorscale = None

    def plot(self, **kwarg_):

        plot_node_point(
            #
            self.node_,
            self.node_name,
            self.node_x_dimension,
            self.point_,
            self.point_name,
            self.point_x_dimension,
            #
            group_=self.group_,
            group_colorscale=self.group_colorscale,
            _1d_grid=self._1d_grid,
            nd_probability_vector=self.nd_probability_vector,
            nd_group_vector=self.nd_group_vector,
            #
            **kwarg_,
        )

    def set_point_group(
        self, point_group, group_colorscale=None, n_grid=64,
    ):

        assert 3 <= point_group.value_counts().min()

        assert not point_group.isna().any()

        self.group_ = point_group

        self.group_colorscale = group_colorscale

        mask = full((n_grid,) * 2, nan)

        triangulation = Delaunay(self.node_x_dimension)

        self._1d_grid = make_1d(0, 1, 1e-3, n_grid)

        for i_0 in range(n_grid):

            for i_1 in range(n_grid):

                mask[i_0, i_1] = triangulation.find_simplex(
                    (self._1d_grid[i_0], self._1d_grid[i_1])
                )

        group_grid_nd_probabilities = {}

        bandwidths = tuple(self.point_x_dimension.apply(get_bandwidth))

        grid_1ds = (self._1d_grid,) * 2

        for group in self.group_.unique():

            group_grid_nd_probabilities[group] = shape(
                get_probability(
                    self.point_x_dimension[self.group_ == group].to_numpy(),
                    plot=False,
                    bandwidths=bandwidths,
                    grid_1ds=grid_1ds,
                )[1],
                grid_1ds,
            )

        self.nd_probability_vector = full((n_grid,) * 2, nan)

        self.nd_group_vector = full((n_grid,) * 2, nan)

        for i_0 in range(n_grid):

            for i_1 in range(n_grid):

                if mask[i_0, i_1] != -1:

                    max_probability = 0

                    max_group = nan

                    for (
                        group,
                        grid_nd_probabilities,
                    ) in group_grid_nd_probabilities.items():

                        probability = grid_nd_probabilities[i_0, i_1]

                        if max_probability < probability:

                            max_probability = probability

                            max_group = group

                    self.nd_probability_vector[i_0, i_1] = max_probability

                    self.nd_group_vector[i_0, i_1] = max_group

        plot_heat_map(
            DataFrame(
                data=apply_along_axis(
                    normalize, 1, self.point_x_node.to_numpy(), "-0-"
                ),
                index=self.point_x_node.index,
                columns=self.point_x_node.columns,
            ).T,
            column_annotations=self.group_,
            column_annotation_colorscale=self.group_colorscale,
            layout={"yaxis": {"dtick": 1}},
        )

    def predict(self, new_point_x_node, **plot_node_point_keyword_arguments):

        plot_node_point(
            self.node_x_dimension,
            DataFrame(
                data=pull_point(
                    self.node_x_dimension.to_numpy(), new_point_x_node.to_numpy()
                ),
                index=new_point_x_node.index,
                columns=self.node_x_dimension.columns,
            ),
            group_=None,
            group_colorscale=self.group_colorscale,
            _1d_grid=self._1d_grid,
            nd_probability_vector=self.nd_probability_vector,
            nd_group_vector=self.nd_group_vector,
            **plot_node_point_keyword_arguments,
        )


def read(file_path):

    with gzip_open(file_path) as io:

        return load(io)


def write(file_path, gps_map):

    with gzip_open(file_path, mode="wb") as io:

        dump(gps_map, io)
