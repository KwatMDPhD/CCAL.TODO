from numpy import unique


def get_1d_grid(co_po_di):
    return [unique(co_) for co_ in co_po_di.T]
