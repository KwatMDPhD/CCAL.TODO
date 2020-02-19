from numpy import full, nan


def make_element_x_dimension_from_element_x_node_and_node_x_dimension(
    element_x_node, node_x_dimension
):

    element_x_dimension = full(
        (element_x_node.shape[0], node_x_dimension.shape[1]), nan
    )

    for element_index in range(element_x_node.shape[0]):

        pulls = element_x_node[element_index, :]

        for dimension_index in range(node_x_dimension.shape[1]):

            element_x_dimension[element_index, dimension_index] = (
                pulls * node_x_dimension[:, dimension_index]
            ).sum() / pulls.sum()

    return element_x_dimension