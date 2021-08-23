def flatten(an_, ty_=(tuple, list, set)):

    an_ = list(an_)

    ie = 0

    while ie < len(an_):

        while isinstance(an_[ie], ty_):

            if len(an_[ie]) == 0:

                an_.pop(ie)

                ie -= 1

                break

            else:

                an_[ie : ie + 1] = an_[ie]

        ie += 1

    return an_
