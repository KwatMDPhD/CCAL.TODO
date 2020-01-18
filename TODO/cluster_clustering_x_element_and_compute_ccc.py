from numpy import zeros
from scipy.cluster.hierarchy import cophenet, fcluster, linkage
from scipy.spatial.distance import squareform


def cluster_clustering_x_element_and_compute_ccc(
    clustering_x_element, r, linkage_method
):

    n_clustering, n_element = clustering_x_element.shape

    element_x_element = zeros((n_element, n_element))

    for clustering_index in range(n_clustering):

        for element_index_0 in range(n_element):

            for element_index_1 in range(element_index_0, n_element):

                if element_index_0 == element_index_1:

                    element_x_element[element_index_0, element_index_1] += 1

                elif (
                    clustering_x_element[clustering_index, element_index_0]
                    == clustering_x_element[clustering_index, element_index_1]
                ):

                    element_x_element[element_index_0, element_index_1] += 1

                    element_x_element[element_index_1, element_index_0] += 1

    clustering_distance = squareform(1 - element_x_element / n_clustering)

    clustering_distance_linkage = linkage(clustering_distance, method=linkage_method)

    return (
        fcluster(clustering_distance_linkage, r, criterion="maxclust") - 1,
        cophenet(clustering_distance_linkage, clustering_distance)[0],
    )
