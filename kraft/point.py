from numpy import (
    absolute,
    apply_along_axis,
    asarray,
    full,
    integer,
    isnan,
    median,
    nan,
    unique,
    where,
)
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
    node_x_dimension,
    point_x_dimension,
    show_node_text=True,
    node_trace=None,
    point_trace=None,
    groups=None,
    group_colorscale=None,
    grid_1d=None,
    grid_nd_probability=None,
    grid_nd_group=None,
    scores=None,
    score_colorscale=None,
    score_opacity=0.8,
    score_nan_opacity=0.1,
    points_to_highlight=(),
    file_path=None,
):

    nodes = node_x_dimension.index.to_numpy()

    node_name = node_x_dimension.index.name

    node_x_dimension = asarray(
        (
            node_x_dimension.iloc[:, 1].to_numpy(),
            1 - node_x_dimension.iloc[:, 0].to_numpy(),
        )
    ).T

    points = point_x_dimension.index.to_numpy()

    point_name = point_x_dimension.index.name

    point_x_dimension = asarray(
        (
            point_x_dimension.iloc[:, 1].to_numpy(),
            1 - point_x_dimension.iloc[:, 0].to_numpy(),
        )
    ).T

    title = "{} {} and {} {}".format(nodes.size, node_name, points.size, point_name)

    if scores is not None:

        score_name = scores.name

        scores = scores.to_numpy()

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

    triangulation_0s, triangulation_1s = get_triangulation(node_x_dimension)

    convex_hull_0s, convex_hull_1s = get_convex_hull(node_x_dimension)

    data.append(
        {
            "legendgroup": "Node",
            "name": "Line",
            "x": triangulation_0s + convex_hull_0s,
            "y": triangulation_1s + convex_hull_1s,
            "mode": "lines",
            "line": {"color": "#171412"},
        }
    )

    base = {
        "legendgroup": "Node",
        "name": node_name,
        "x": node_x_dimension[:, 0],
        "y": node_x_dimension[:, 1],
        "text": nodes,
        "mode": "markers",
        "marker": {
            "size": 20,
            "color": "#23191e",
            "line": {"width": 1, "color": "#ebf6f7"},
        },
        "hoverinfo": "text",
    }

    if node_trace is not None:

        base = merge(base, node_trace)

    data.append(base)

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
            for node, (x, y) in zip(nodes, node_x_dimension)
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

            grid_nd_probability_group = grid_nd_probability.copy()

            grid_nd_probability_group[grid_nd_group != group] = nan

            data.append(
                {
                    "type": "heatmap",
                    "x": grid_1d,
                    "y": 1 - grid_1d,
                    "z": grid_nd_probability_group,
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

    if scores is not None:

        is_ = absolute(scores).argsort()

        scores = scores[is_]

        point_x_dimension = point_x_dimension[is_]

        if score_nan_opacity == 0:

            scores = scores[~isnan(scores)]

        else:

            score_opacity = where(isnan(scores), score_nan_opacity, score_opacity)

        if all(isinstance(score, integer) for score in scores):

            tickvals = unique(scores)

            ticktext = tickvals

        else:

            tickvals = (scores.min(), median(scores), scores.mean(), scores.max())

            ticktext = tuple("{:.2e}".format(number) for number in tickvals)

        data.append(
            merge(
                base,
                {
                    "x": point_x_dimension[:, 0],
                    "y": point_x_dimension[:, 1],
                    "marker": {
                        "color": scores,
                        "colorscale": score_colorscale,
                        "colorbar": {
                            **COLORBAR,
                            "tickmode": "array",
                            "tickvals": tickvals,
                            "ticktext": ticktext,
                        },
                        "opacity": score_opacity,
                    },
                },
            )
        )

    elif groups is not None:

        for group in unique_groups:

            name = "Group {}".format(group)

            is_ = groups == group

            data.append(
                merge(
                    base,
                    {
                        "legendgroup": name,
                        "name": name,
                        "x": point_x_dimension[is_, 0],
                        "y": point_x_dimension[is_, 1],
                        "text": points[is_],
                        "marker": {"color": group_to_color[group]},
                    },
                )
            )
    else:

        data.append(
            {
                **base,
                "x": point_x_dimension[:, 0],
                "y": point_x_dimension[:, 1],
                "text": points,
            }
        )

    for point in points_to_highlight:

        i = (points == point).nonzero()[-1]

        x, y = point_x_dimension[i]

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
