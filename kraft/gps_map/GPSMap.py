from numpy import apply_along_axis, full, nan
from pandas import DataFrame
from scipy.spatial import Delaunay

from .estimate_pdf import estimate_pdf
from .get_bandwidth import get_bandwidth
from .make_grid import make_grid
from .map_points import map_points
from .map_points_by_pull import map_points_by_pull
from .normalize import normalize
from .plot_gps_map import plot_gps_map
from .plot_heat_map import plot_heat_map
from .RANDOM_SEED import RANDOM_SEED
from .unmesh import unmesh


class GPSMap:
    def __init__(self, node_x_node_distance, point_x_node, random_seed=RANDOM_SEED):

        self.point_x_node = point_x_node

        self.node_x_dimension = DataFrame(
            map_points(node_x_node_distance, 2, random_seed=random_seed),
            index=self.point_x_node.columns,
        )

        self.point_x_dimension = DataFrame(
            map_points_by_pull(self.node_x_dimension.values, self.point_x_node.values),
            index=self.point_x_node.index,
        )

        self.point_label = None

        self.dimension_grid = None

        self.grid_probability = None

        self.grid_label = None

        self.point_label_colorscale = None

    def plot(self, **plot_gps_map_keyword_arguments):

        plot_gps_map(
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

        self.dimension_grid = make_grid(0, 1, 1e-3, n_grid)

        for i in range(n_grid):

            for j in range(n_grid):

                mask_grid[i, j] = triangulation.find_simplex(
                    (self.dimension_grid[i], self.dimension_grid[j])
                )

        label_grid_probability = {}

        bandwidths = tuple(self.point_x_dimension.apply(get_bandwidth))

        grids = (self.dimension_grid,) * 2

        for label in self.point_label.unique():

            label_grid_probability[label] = unmesh(
                *estimate_pdf(
                    self.point_x_dimension[self.point_label == label].values,
                    plot=False,
                    bandwidths=bandwidths,
                    grids=grids,
                )
            )[1]

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

        plot_gps_map(
            self.node_x_dimension,
            DataFrame(
                map_points_by_pull(
                    self.node_x_dimension.values, new_point_x_node.values
                ),
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
