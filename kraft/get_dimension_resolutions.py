from numpy import diff, unique


def get_dimension_resolutions(point_x_dimension):

    return tuple(diff(unique(v).sort()).min() for v in point_x_dimension.T)
