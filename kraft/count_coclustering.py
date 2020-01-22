from numpy import triu_indices, zeros
from scipy.spatial.distance import squareform


def count_coclustering(clustering_x_element, r):

    element_0_element_1_index = tuple(
        zip(*triu_indices(clustering_x_element.shape[1], k=1))
    )

    n_colusterings = zeros(len(element_0_element_1_index))

    for clusterings in clustering_x_element:

        for element_0_index, element_1_index in element_0_element_1_index:

            n_colusterings[element_0_index, element_1_index] += int(
                clusterings[element_0_index] == clusterings[element_1_index]
            )

    return squareform(n_colusterings)
