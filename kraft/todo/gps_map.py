from gzip import open as gzip_open
from pickle import dump, load

from numpy import apply_along_axis, full, nan
from pandas import DataFrame
from scipy.spatial import Delaunay

from .array import normalize
from .CONSTANT import RANDOM_SEED
from .grid import make_grid_1d, shape
from .kernel_density import get_bandwidth
from .plot import plot_heat_map
from .point import map_point, plot_node_point, pull_point
from .probability import get_probability


class GPSMap:
    def __init__(self, node_x_node_distance, point_x_node, random_seed=RANDOM_SEED):

        self.point_x_node = point_x_node

        self.node_x_dimension = DataFrame(
            map_point(node_x_node_distance, 2, random_seed=random_seed),
            index=self.point_x_node.columns,
        )

        self.point_x_dimension = DataFrame(
            data=pull_point(
                self.node_x_dimension.to_numpy(), self.point_x_node.to_numpy()
            ),
            index=self.point_x_node.index,
        )

        self.point_group = None

        self.grid_1d = None

        self.grid_nd_probabilities = None

        self.grid_nd_group = None

        self.group_colorscale = None

    def plot(self, **plot_node_point_keyword_arguments):

        plot_node_point(
            self.node_x_dimension,
            self.point_x_dimension,
            groups=self.point_group,
            group_colorscale=self.group_colorscale,
            grid_1d=self.grid_1d,
            grid_nd_probability=self.grid_nd_probabilities,
            grid_nd_group=self.grid_nd_group,
            **plot_node_point_keyword_arguments,
        )

    def set_point_group(
        self, point_group, group_colorscale=None, n_grid=64,
    ):

        assert 3 <= point_group.value_counts().min()

        assert not point_group.isna().any()

        self.point_group = point_group

        self.group_colorscale = group_colorscale

        mask = full((n_grid,) * 2, nan)

        triangulation = Delaunay(self.node_x_dimension)

        self.grid_1d = make_grid_1d(0, 1, 1e-3, n_grid)

        for i_0 in range(n_grid):

            for i_1 in range(n_grid):

                mask[i_0, i_1] = triangulation.find_simplex(
                    (self.grid_1d[i_0], self.grid_1d[i_1])
                )

        group_grid_nd_probabilities = {}

        bandwidths = tuple(self.point_x_dimension.apply(get_bandwidth))

        grid_1ds = (self.grid_1d,) * 2

        for group in self.point_group.unique():

            group_grid_nd_probabilities[group] = shape(
                get_probability(
                    self.point_x_dimension[self.point_group == group].to_numpy(),
                    plot=False,
                    bandwidths=bandwidths,
                    grid_1ds=grid_1ds,
                )[1],
                grid_1ds,
            )

        self.grid_nd_probabilities = full((n_grid,) * 2, nan)

        self.grid_nd_group = full((n_grid,) * 2, nan)

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

                    self.grid_nd_probabilities[i_0, i_1] = max_probability

                    self.grid_nd_group[i_0, i_1] = max_group

        plot_heat_map(
            DataFrame(
                data=apply_along_axis(
                    normalize, 1, self.point_x_node.to_numpy(), "-0-"
                ),
                index=self.point_x_node.index,
                columns=self.point_x_node.columns,
            ).T,
            column_annotations=self.point_group,
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
            groups=None,
            group_colorscale=self.group_colorscale,
            grid_1d=self.grid_1d,
            grid_nd_probability=self.grid_nd_probabilities,
            grid_nd_group=self.grid_nd_group,
            **plot_node_point_keyword_arguments,
        )


def read(file_path):

    with gzip_open(file_path) as io:

        return load(io)


def write(file_path, gps_map):

    with gzip_open(file_path, mode="wb") as io:

        dump(gps_map, io)
