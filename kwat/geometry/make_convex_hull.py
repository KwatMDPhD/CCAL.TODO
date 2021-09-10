from scipy.spatial import Delaunay


def make_convex_hull(no_po_di):

    po0_ = []

    po1_ = []

    de = Delaunay(no_po_di)

    for ie0, ie1 in de.convex_hull:

        po0 = de.points[ie0]

        po1 = de.points[ie1]

        po0_.append(po0[0])

        po0_.append(po1[0])

        po0_.append(None)

        po1_.append(po0[1])

        po1_.append(po1[1])

        po1_.append(None)

    return po0_, po1_
