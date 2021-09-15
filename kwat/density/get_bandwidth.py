from KDEpy.bw_selection import _bw_methods


def get_bandwidth(nu_po_di, me="ISJ"):

    me = _bw_methods[me]

    return [me(nu_.reshape([-1, 1])) for nu_ in nu_po_di.T]
