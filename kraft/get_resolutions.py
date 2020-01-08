from numpy import diff, unique


def get_resolutions(point_x_dimension):

    return tuple(diff(unique(vector)).min() for vector in point_x_dimension.T)
