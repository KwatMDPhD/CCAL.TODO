from KDEpy.bw_selection import _bw_methods


def get_bandwidth(nu_po_di, me="ISJ"):

    ba_ = []

    for ie in range(nu_po_di.shape[1]):

        ba_.append(_bw_methods[me](nu_po_di[:, [ie]]))

    return ba_
