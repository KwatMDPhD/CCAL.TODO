from numpy import apply_along_axis, asarray, full, isnan, nan, unique, where
from sklearn.manifold import MDS

from .array import normalize
from .CONSTANT import RANDOM_SEED
from .dict_ import merge
from .geometry import get_convex_hull, get_triangulation
from .plot import COLORBAR, get_color, make_colorscale, plot_plotly


def pull_point(node_x_dimension, point_x_node):

    point_x_dimension = full((point_x_node.shape[0], node_x_dimension.shape[1]), nan)

    for point_i in range(point_x_node.shape[0]):

        pulls = point_x_node[point_i, :]

        for dimension_i in range(node_x_dimension.shape[1]):

            point_x_dimension[point_i, dimension_i] = (
                pulls * node_x_dimension[:, dimension_i]
            ).sum() / pulls.sum()

    return point_x_dimension


def map_point(
    point_x_point_distance,
    n_dimension,
    metric=True,
    n_init=int(1e3),
    max_iter=int(1e3),
    verbose=0,
    eps=1e-3,
    n_job=1,
    random_seed=RANDOM_SEED,
):

    point_x_dimension = MDS(
        n_components=n_dimension,
        metric=metric,
        n_init=n_init,
        max_iter=max_iter,
        verbose=verbose,
        eps=eps,
        n_jobs=n_job,
        random_state=random_seed,
        dissimilarity="precomputed",
    ).fit_transform(point_x_point_distance)

    return apply_along_axis(normalize, 0, point_x_dimension, "0-1")


def plot_node_point(
    #
    node_x_dimension,
    point_x_dimension,
    #
    show_node_text=True,
    #
    node_trace=None,
    point_trace=None,
    #
    point_groups=None,
    group_colorscale=None,
    #
    grid_1d=None,
    grid_nd_probability=None,
    grid_nd_group=None,
    #
    point_scores=None,
    score_colorscale=None,
    score_opacity=0.8,
    score_na_opacity=0.1,
    #
    points_to_highlight=(),
    file_path=None,
):

    node_x_x_y = asarray(
        (
            node_x_dimension.iloc[:, 1].to_numpy(),
            1 - node_x_dimension.iloc[:, 0].to_numpy(),
        )
    )

    nodes = node_x_dimension.index.to_numpy()

    node_name = node_x_dimension.index.name

    point_x_x_y = asarray(
        (
            point_x_dimension.iloc[:, 1].to_numpy(),
            1 - point_x_dimension.iloc[:, 0].to_numpy(),
        )
    )

    points = point_x_dimension.index.to_numpy()

    point_name = point_x_dimension.index.name

    title = "{} {} and {} {}".format(nodes.size, node_name, points.size, point_name)

    if point_scores is not None:

        scores = point_scores.to_numpy()

        score_name = point_scores.name

        title = "{}<br>{}".format(title, score_name)

    axis = {
        "showgrid": False,
        "showline": False,
        "zeroline": False,
        "showticklabels": False,
    }

    layout = {
        "height": 880,
        "width": 880,
        "title": {
            "x": 0.5,
            "text": "<b>{}</b>".format(title),
            "font": {
                "size": 24,
                "color": "#4e40d8",
                "family": "Times New Roman, sans-serif",
            },
        },
        "xaxis": axis,
        "yaxis": axis,
        "annotations": [],
    }

    data = []

    triangulation_xs, triangulation_ys = get_triangulation(node_x_x_y)

    convex_hull_xs, convex_hull_ys = get_convex_hull(node_x_x_y)

    data.append(
        {
            "legendgroup": "Node",
            "name": "Line",
            "x": triangulation_xs + convex_hull_xs,
            "y": triangulation_ys + convex_hull_ys,
            "mode": "lines",
            "line": {"color": "#171412"},
        }
    )

    data.append(
        {
            "legendgroup": "Node",
            "name": node_name,
            "x": node_x_x_y[:, 0],
            "y": node_x_x_y[:, 1],
            "text": nodes,
            "mode": "markers",
            "marker": {
                "size": 20,
                "color": "#23191e",
                "line": {"width": 1, "color": "#ebf6f7"},
            },
            "hoverinfo": "text",
        }
    )

    if show_node_text:

        arrowwidth = 1.6

        arrowcolor = "#ebf6f7"

        layout["annotations"] += [
            {
                "x": x,
                "y": y,
                "text": "<b>{}</b>".format(node),
                "font": {
                    "size": 16,
                    "color": "#23191e",
                    "family": "Gravitas One, monospace",
                },
                "borderpad": 2,
                "borderwidth": arrowwidth,
                "bordercolor": arrowcolor,
                "bgcolor": "#ffffff",
                "arrowwidth": arrowwidth,
                "arrowcolor": arrowcolor,
            }
            for node, (x, y) in zip(nodes, node_x_x_y)
        ]

    if grid_nd_group is not None:

        data.append(
            {
                "type": "contour",
                "showlegend": False,
                "x": grid_1d,
                "y": 1 - grid_1d,
                "z": grid_nd_probability,
                "autocontour": False,
                "ncontours": 24,
                "contours": {"coloring": "none"},
            }
        )

        unique_groups = unique(grid_nd_group[~isnan(grid_nd_group)])

        n_unique_group = unique_groups.size

        group_to_color = {
            group: get_color(group_colorscale, group, n_unique_group)
            for group in unique_groups
        }

        for group in unique_groups:

            grid_nd_probability_ = grid_nd_probability.copy()

            grid_nd_probability_[grid_nd_group != group] = nan

            data.append(
                {
                    "type": "heatmap",
                    "x": grid_1d,
                    "y": 1 - grid_1d,
                    "z": grid_nd_probability_,
                    "colorscale": make_colorscale(
                        ("rgb(255, 255, 255)", group_to_color[group])
                    ),
                    "showscale": False,
                    "hoverinfo": "none",
                }
            )

    base = {
        "name": point_name,
        "mode": "markers",
        "marker": {
            "size": 16,
            "color": "#20d9ba",
            "line": {"width": 0.8, "color": "#ebf6f7"},
        },
        "hoverinfo": "text",
    }

    if point_trace is not None:

        base = merge(base, point_trace)

    if point_scores is not None:

        point_scores = point_scores.reindex(index=point_x_y.index)

        point_scores = point_scores[
            point_scores.abs().sort_values(na_position="first").index
        ]

        if score_na_opacity == 0:

            point_scores.dropna(inplace=True)

        else:

            score_opacity = where(point_scores.isna(), score_na_opacity, score_opacity)

        point_x_y = point_x_y.loc[point_scores.index]

        if point_scores.astype(float).map(float.is_integer).all():

            tickvals = point_scores.unique()

            template = "{:.0f}".format

        else:

            tickvals = point_scores.describe()[
                ["min", "25%", "50%", "mean", "75%", "max"]
            ].to_numpy()

            template = "{:.2e}".format

        data.append(
            merge(
                base,
                {
                    "x": point_x_x_y[:, 0],
                    "y": point_x_x_y[:, 1],
                    "marker": {
                        "color": point_scores,
                        "colorscale": score_colorscale,
                        "colorbar": {
                            **COLORBAR,
                            "tickmode": "array",
                            "tickvals": tickvals,
                            "ticktext": tuple(template(number) for number in tickvals),
                        },
                        "opacity": score_opacity,
                    },
                },
            )
        )

    elif point_groups is not None:

        for group in unique_groups:

            name = "Group {}".format(group)

            is_ = point_groups == group

            data.append(
                merge(
                    base,
                    {
                        "legendgroup": name,
                        "name": name,
                        "x": point_x_y[is_, 0],
                        "y": point_x_y[is_, 1],
                        "text": points[is_],
                        "marker": {"color": group_to_color[group]},
                    },
                )
            )
    else:

        data.append(
            {**base, "x": point_x_x_y[:, 0], "y": point_x_x_y[:, 1], "text": points}
        )

    for point in points_to_highlight:

        i = (points == point).nonzero()[-1]

        x, y = point_x_x_y[i]

        layout["annotations"].append(
            {
                "x": x,
                "y": y,
                "text": "<b>{}</b>".format(point),
                "arrowhead": 2,
                "arrowwidth": 2,
                "arrowcolor": "#c93756",
                "standoff": None,
                "clicktoshow": "onoff",
            }
        )

    plot_plotly({"layout": layout, "data": data}, file_path=file_path)
