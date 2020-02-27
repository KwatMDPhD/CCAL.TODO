from numpy import full, nan


def map_points_by_pull(node_x_dimension, point_x_node):

    point_x_dimension = full((point_x_node.shape[0], node_x_dimension.shape[1]), nan)

    for point_index in range(point_x_node.shape[0]):

        pulls = point_x_node[point_index, :]

        for dimension_index in range(node_x_dimension.shape[1]):

            point_x_dimension[point_index, dimension_index] = (
                pulls * node_x_dimension[:, dimension_index]
            ).sum() / pulls.sum()

    return point_x_dimension
