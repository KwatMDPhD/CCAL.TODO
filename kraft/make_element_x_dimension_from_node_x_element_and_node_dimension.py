from numpy import full, nan, sort

from .normalize_array import normalize_array


def make_element_x_dimension_from_node_x_element_and_node_dimension(
    node_x_element, node_x_dimension, n_pull, pull_power
):

    element_x_dimension = full(
        (node_x_element.shape[1], node_x_dimension.shape[1]), nan
    )

    node_x_element = normalize_array(node_x_element, None, "0-1")

    if 3 < node_x_element.shape[0]:

        node_x_element = normalize_array(node_x_element, 0, "0-1")

    for element_index in range(node_x_element.shape[1]):

        pulls = node_x_element[:, element_index]

        if n_pull is not None:

            pulls[pulls < sort(pulls)[-n_pull]] = 0

        if pull_power is not None:

            pulls = pulls ** pull_power

        for dimension_index in range(node_x_dimension.shape[1]):

            element_x_dimension[element_index, dimension_index] = (
                pulls * node_x_dimension[:, dimension_index]
            ).sum() / pulls.sum()

    return element_x_dimension
