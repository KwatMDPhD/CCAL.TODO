from KDEpy.bw_selection import improved_sheather_jones, silvermans_rule


def get_bandwidth(nu_po_di):

    ba_ = []

    for ie in range(nu_po_di.shape[1]):

        nu_ = nu_po_di[:, [ie]]

        try:

            ba = improved_sheather_jones(nu_)

        except ValueError as er:

            ba = silvermans_rule(nu_)

        ba_.append(ba)

    return ba_
