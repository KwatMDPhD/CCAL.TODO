from numpy import array, meshgrid


def make_nd_grid(co__):
    return array([co_po_di.ravel() for co_po_di in meshgrid(*co__, indexing="ij")]).T
