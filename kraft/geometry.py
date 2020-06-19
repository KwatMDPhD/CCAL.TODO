from scipy.spatial import Delaunay


def get_triangulation(point_x_dimension):

    xs = []

    ys = []

    triangulation = Delaunay(point_x_dimension)

    for point_0_index, point_1_index, point_2_index in triangulation.simplices:

        point_0 = triangulation.points[point_0_index]

        point_1 = triangulation.points[point_1_index]

        point_2 = triangulation.points[point_2_index]

        xs.append(point_0[0])

        xs.append(point_1[0])

        xs.append(point_2[0])

        xs.append(None)

        ys.append(point_0[1])

        ys.append(point_1[1])

        ys.append(point_2[1])

        ys.append(None)

    return xs, ys


def get_convex_hull(point_x_dimension):

    xs = []

    ys = []

    triangulation = Delaunay(point_x_dimension)

    for point_0_index, point_1_index in triangulation.convex_hull:

        point_0 = triangulation.points[point_0_index]

        point_1 = triangulation.points[point_1_index]

        xs.append(point_0[0])

        xs.append(point_1[0])

        xs.append(None)

        ys.append(point_0[1])

        ys.append(point_1[1])

        ys.append(None)

    return xs, ys
