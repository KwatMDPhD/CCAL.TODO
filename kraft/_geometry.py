from scipy.spatial import (
    Delaunay,
)

# TODO: refactor


def triangulation(
    point_x_dimension,
):

    _0 = []

    _1 = []

    triangulation = Delaunay(point_x_dimension)

    for (
        index_0,
        index_1,
        index_2,
    ) in triangulation.simplices:

        point_0 = triangulation.points[index_0]

        point_1 = triangulation.points[index_1]

        point_2 = triangulation.points[index_2]

        _0.append(point_0[0])

        _0.append(point_1[0])

        _0.append(point_2[0])

        _0.append(None)

        _1.append(point_0[1])

        _1.append(point_1[1])

        _1.append(point_2[1])

        _1.append(None)

    return (
        _0,
        _1,
    )


def convex_hull(
    point_x_dimension,
):

    _0 = []

    _1 = []

    triangulation = Delaunay(point_x_dimension)

    for (
        index_0,
        index_1,
    ) in triangulation.convex_hull:

        point_0 = triangulation.points[index_0]

        point_1 = triangulation.points[index_1]

        _0.append(point_0[0])

        _0.append(point_1[0])

        _0.append(None)

        _1.append(point_0[1])

        _1.append(point_1[1])

        _1.append(None)

    return (
        _0,
        _1,
    )
