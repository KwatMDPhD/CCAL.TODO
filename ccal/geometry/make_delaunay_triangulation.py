from scipy.spatial import Delaunay


def make_delaunay_triangulation(no_po_di):
    po1_ = []

    po2_ = []

    de = Delaunay(no_po_di)

    for ie1, ie2, ie3 in de.simplices:
        po1 = de.points[ie1]

        po2 = de.points[ie2]

        po3 = de.points[ie3]

        po1_.append(po1[0])

        po1_.append(po2[0])

        po1_.append(po3[0])

        po1_.append(None)

        po2_.append(po1[1])

        po2_.append(po2[1])

        po2_.append(po3[1])

        po2_.append(None)

    return po1_, po2_
