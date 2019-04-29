from numpy import zeros
from scipy.cluster.hierarchy import cophenet, fcluster, linkage
from scipy.spatial.distance import squareform


def cluster_clustering_x_element_and_compute_ccc(
    clustering_x_element, k, linkage_method
):

    n_coclustering__element_x_element = zeros((clustering_x_element.shape[1],) * 2)

    for clustering in range(clustering_x_element.shape[0]):

        for element_0 in range(clustering_x_element.shape[1]):

            for element_1 in range(element_0, clustering_x_element.shape[1]):

                if element_0 == element_1:

                    n_coclustering__element_x_element[element_0, element_1] += 1

                elif (
                    clustering_x_element[clustering, element_0]
                    == clustering_x_element[clustering, element_1]
                ):

                    n_coclustering__element_x_element[element_0, element_1] += 1

                    n_coclustering__element_x_element[element_1, element_0] += 1

    clustering_distance = squareform(
        1 - n_coclustering__element_x_element / clustering_x_element.shape[0]
    )

    clustering_distance_linkage = linkage(clustering_distance, method=linkage_method)

    return (
        fcluster(clustering_distance_linkage, k, criterion="maxclust") - 1,
        cophenet(clustering_distance_linkage, clustering_distance)[0],
    )
