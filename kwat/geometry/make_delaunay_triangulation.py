from scipy.spatial import Delaunay


def make_delaunay_triangulation(no_po_di):

    po0_ = []

    po1_ = []

    de = Delaunay(no_po_di)

    for ie0, ie1, ie2 in de.simplices:

        po0 = de.points[ie0]

        po1 = de.points[ie1]

        po2 = de.points[ie2]

        po0_.append(po0[0])

        po0_.append(po1[0])

        po0_.append(po2[0])

        po0_.append(None)

        po1_.append(po0[1])

        po1_.append(po1[1])

        po1_.append(po2[1])

        po1_.append(None)

    return po0_, po1_
