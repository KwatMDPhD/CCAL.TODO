from numpy import array


def warn_mismatch(la1_, la2_):

    la1_ = array(la1_)

    la2_ = array(la2_)

    mi_ = la1_ != la2_

    if mi_.any():

        print(
            "{}/{} mismatch:\n{}".format(
                mi_.sum(),
                mi_.size,
                "\n".join(
                    (
                        "({}) {} != {}".format(ie, la1, la2)
                        for ie, (la1, la2) in enumerate(zip(la1_[mi_], la2_[mi_]))
                    )
                ),
            )
        )
