from numpy import arange, diag, exp, full, linspace, mean, nan, rot90
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
from .normalize_array_on_axis import normalize_array_on_axis
from .normalize_dataframe import normalize_dataframe
from .pick_colors import pick_colors
from .plot_gps_map import plot_gps_map
from .plot_heat_map import plot_heat_map
from .plot_plotly_figure import plot_plotly_figure
from .RANDOM_SEED import RANDOM_SEED
from .scale_element_x_dimension_dimension import scale_element_x_dimension_dimension
from .train_and_classify import train_and_classify
from .unmesh import unmesh

element_marker_size = 16

grid_label_opacity_without_annotation = 0.64

grid_label_opacity_with_annotation = 0.5

grid_extension = 1 / 1e3


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

            check_dataframe_number(w)

        if h is not None:

            check_dataframe_number(h)

        if w is not None and h is not None:

            if (w.index == h.index).all() and w.index.name == h.index.name:

                self.nodes = w.index.tolist()

                self.node_name = w.index.name

            else:

                raise ValueError

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

        if self.node_x_dimension is None:

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

    def plot(
        self,
        w_or_h,
        grid_label_opacity=None,
        annotation_x_element=None,
        annotation_types=None,
        annotation_std_maxs=None,
        annotation_colorscales=None,
        elements_to_be_emphasized=None,
        element_marker_size=element_marker_size,
        layout_size=880,
        highlight_binary=False,
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

        plot_gps_map(
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
            annotation_colorscales,
            layout_size,
            highlight_binary,
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

            raise ValueError

        if (element_label.value_counts() < 3).any():

            raise ValueError

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

        dimension_bandwidths = (
            compute_vector_bandwidth(element_x_dimension[:, 0]),
            compute_vector_bandwidth(element_x_dimension[:, 1]),
        )

        n_dimension = element_x_dimension.shape[1]

        dimension_bandwidth_factors = (bandwidth_factor,) * n_dimension

        dimension_grid_mins = (0 - grid_extension,) * n_dimension

        dimension_grid_maxs = (1 + grid_extension,) * n_dimension

        dimension_fraction_grid_extensions = (0,) * n_dimension

        dimension_n_grids = (self.n_grid,) * n_dimension

        for label in element_label.sort_values().unique():

            label_grid_probabilities[label] = rot90(
                unmesh(
                    *compute_element_x_dimension_joint_probability(
                        element_x_dimension[element_label == label],
                        plot=False,
                        dimension_bandwidths=dimension_bandwidths,
                        dimension_bandwidth_factors=dimension_bandwidth_factors,
                        dimension_grid_mins=dimension_grid_mins,
                        dimension_grid_maxs=dimension_grid_maxs,
                        dimension_fraction_grid_extensions=dimension_fraction_grid_extensions,
                        dimension_n_grids=dimension_n_grids,
                    )
                )[1]
            )

        grid_values = full(dimension_n_grids, nan)

        grid_labels = full(dimension_n_grids, nan)

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

                dataframe = DataFrame(
                    self.w, index=self.nodes, columns=self.w_elements
                )[column_annotation.index]

                element_name = self.w_element_name

            elif w_or_h == "h":

                column_annotation = self.h_element_label.sort_values()

                dataframe = DataFrame(
                    self.h, index=self.nodes, columns=self.h_elements
                )[column_annotation.index]

                element_name = self.h_element_name

            plot_heat_map(
                normalize_dataframe(dataframe, 0, "-0-"),
                column_annotations=column_annotation,
                # column_annotation_colors=label_colors,
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
        support_vector_parameter_c=1e3,
        n_pull=None,
        pull_power=None,
        grid_label_opacity=None,
        annotation_x_element=None,
        annotation_types=None,
        annotation_std_maxs=None,
        annotation_colorscales=None,
        element_marker_size=element_marker_size,
        layout_size=880,
        highlight_binary=False,
        title=None,
        html_file_path=None,
    ):

        check_dataframe_number(node_x_predicting_element)

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

        predicting_element_x_dimension = make_element_x_dimension_from_node_x_element_and_node_dimension(
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
                name="Predicted {} Label".format(element_name),
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

            title = "{} (predicted)".format(w_or_h.title())

        plot_gps_map(
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
            annotation_colorscales,
            layout_size,
            highlight_binary,
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
                        self.w.T, metric=compute_information_distance_between_2_vectors
                    )
                )

            distance__element_x_element = self.w_distance__element_x_element

            if self.w_distance__node_x_element is None:

                distance__node_x_w_element_ = apply_function_on_slices_from_2_matrices(
                    self.w,
                    diag((1,) * len(self.w_elements)),
                    0,
                    compute_information_distance_between_2_vectors,
                )

                distance__w_element_x_node_ = apply_function_on_slices_from_2_matrices(
                    self.w,
                    diag((1,) * len(self.nodes)),
                    1,
                    compute_information_distance_between_2_vectors,
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
                        self.h.T, metric=compute_information_distance_between_2_vectors
                    )
                )

            distance__element_x_element = self.h_distance__element_x_element

            if self.h_distance__node_x_element is None:

                distance__node_x_h_element_ = apply_function_on_slices_from_2_matrices(
                    self.h,
                    diag((1,) * len(self.h_elements)),
                    0,
                    compute_information_distance_between_2_vectors,
                )

                distance__h_element_x_node_ = apply_function_on_slices_from_2_matrices(
                    self.h,
                    diag((1,) * len(self.nodes)),
                    1,
                    compute_information_distance_between_2_vectors,
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
            apply_function_on_slices_from_2_matrices(
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

                print("\t{}/{}...".format(i + 1, n_iteration))

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
                apply_function_on_slices_from_2_matrices(
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

                    print(
                        "\t\t{:.3e} =(accept)=> {:.3e}...".format(fitness, r__fitness)
                    )

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

        plot_plotly_figure(
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
                    },
                    {"type": "scatter", "name": "Node-Node", "x": x, "y": scores[:, 1]},
                    {
                        "type": "scatter",
                        "name": "Element-Element",
                        "x": x,
                        "y": scores[:, 2],
                    },
                    {
                        "type": "scatter",
                        "name": "Node-Element",
                        "x": x,
                        "y": scores[:, 3],
                    },
                    {"type": "scatter", "name": "Fitness", "x": x, "y": scores[:, 4]},
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
