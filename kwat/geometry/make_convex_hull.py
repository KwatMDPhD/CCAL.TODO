from scipy.spatial import Delaunay


def make_convex_hull(no_po_di):

    po1_ = []

    po2_ = []

    de = Delaunay(no_po_di)

    for ie1, ie2 in de.convex_hull:

        po0 = de.points[ie1]

        po1 = de.points[ie2]

        po1_.append(po0[0])

        po1_.append(po1[0])

        po1_.append(None)

        po2_.append(po0[1])

        po2_.append(po1[1])

        po2_.append(None)

    return po1_, po2_
