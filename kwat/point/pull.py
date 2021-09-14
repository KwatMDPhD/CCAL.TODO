from numpy import full, nan


def pull(nu_no_di, nu_po_no):

    n_po = nu_po_no.shape[0]

    n_di = nu_no_di.shape[1]

    nu_po_di = full([n_po, n_di], nan)

    for iep in range(n_po):

        pu_ = nu_po_no[iep, :]

        for ied in range(n_di):

            nu_po_di[iep, ied] = (pu_ * nu_no_di[:, ied]).sum() / pu_.sum()

    return nu_po_di
