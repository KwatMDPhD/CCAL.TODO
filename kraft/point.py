from numpy import absolute, full, integer, isnan, median, nan, unique, where
from sklearn.manifold import MDS

from .array import normalize
from .CONSTANT import RANDOM_SEED
from .dict import merge
from .geometry import get_convex_hull, get_triangulation
from .plot import COLORBAR, get_color, make_colorscale, plot_plotly


def map_point(point_x_point_distance, dimension_n, random_seed=RANDOM_SEED, **kwarg_):

    point_x_dimension = MDS(
        n_components=dimension_n,
        random_state=random_seed,
        dissimilarity="precomputed",
        **kwarg_,
    ).fit_transform(point_x_point_distance)

    for index in range(dimension_n):

        point_x_dimension[:, index] = normalize(point_x_dimension[:, index], "0-1")

    return point_x_dimension


def pull_point(node_x_dimension, point_x_node_pull):

    dimension_n = node_x_dimension.shape[1]

    point_n = point_x_node_pull.shape[0]

    point_x_dimension = full((point_n, dimension_n), nan)

    for point_index in range(point_n):

        pull_ = point_x_node_pull[point_index, :]

        for dimension_index in range(dimension_n):

            point_x_dimension[point_index, dimension_index] = (
                pull_ * node_x_dimension[:, dimension_index]
            ).sum() / pull_.sum()

    return point_x_dimension


def plot_node_point(
    #
    node_,
    node_name,
    node_x_dimension,
    point_,
    point_name,
    point_x_dimension,
    #
    group_=None,
    group_colorscale=None,
    _1d_grid=None,
    nd_probability_vector=None,
    nd_group_vector=None,
    #
    show_node_text=True,
    node_trace=None,
    point_trace=None,
    scores=None,
    score_colorscale=None,
    score_opacity=0.8,
    score_nan_opacity=0.1,
    points_to_highlight=(),
    file_path=None,
):

    title = "{} {} and {} {}".format(node_.size, node_name, point_.size, point_name)

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
        "text": node_,
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
            for node, (x, y) in zip(node_, node_x_dimension)
        ]

    if nd_group_vector is not None:

        data.append(
            {
                "type": "contour",
                "showlegend": False,
                "x": _1d_grid,
                "y": 1 - _1d_grid,
                "z": nd_probability_vector,
                "autocontour": False,
                "ncontours": 24,
                "contours": {"coloring": "none"},
            }
        )

        unique_groups = unique(nd_group_vector[~isnan(nd_group_vector)])

        n_unique_group = unique_groups.size

        group_to_color = {
            group: get_color(group_colorscale, group, n_unique_group)
            for group in unique_groups
        }

        for group in unique_groups:

            grid_nd_probability_group = nd_probability_vector.copy()

            grid_nd_probability_group[nd_group_vector != group] = nan

            data.append(
                {
                    "type": "heatmap",
                    "x": _1d_grid,
                    "y": 1 - _1d_grid,
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

        i_ = absolute(scores).argsort()

        scores = scores[i_]

        point_x_dimension = point_x_dimension[i_]

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

    elif group_ is not None:

        for group in unique_groups:

            name = "Group {}".format(group)

            i_ = group_ == group

            data.append(
                merge(
                    base,
                    {
                        "legendgroup": name,
                        "name": name,
                        "x": point_x_dimension[i_, 0],
                        "y": point_x_dimension[i_, 1],
                        "text": point_[i_],
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
                "text": point_,
            }
        )

    for point in points_to_highlight:

        i = (point_ == point).nonzero()[-1]

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
