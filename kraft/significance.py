from numpy import (
    asarray,
    sqrt,
    where,
)
from scipy.stats import (
    norm,
)
from statsmodels.sandbox.stats.multicomp import (
    multipletests,
)


def get_margin_of_error(
    nu_,
    co=0.95,
):

    return norm.ppf(co) * nu_.std() / sqrt(nu_.size)


def get_p_value(
    nu,
    ra_,
    di,
):

    if di == "<":

        bo_ = ra_ <= nu

    elif di == ">":

        bo_ = nu <= ra_

    return (
        max(
            1,
            bo_.sum(),
        )
        / ra_.size
    )


def get_p_value_q_value(
    nu_,
    ra_,
    di,
    mu="fdr_bh",
):

    if "<" in di:

        pl_ = asarray(
            [
                get_p_value(
                    nu,
                    ra_,
                    "<",
                )
                for nu in nu_
            ]
        )

        ql_ = multipletests(
            pl_,
            method=mu,
        )[1]

    if ">" in di:

        pr_ = asarray(
            [
                get_p_value(
                    nu,
                    ra_,
                    ">",
                )
                for nu in nu_
            ]
        )

        qr_ = multipletests(
            pr_,
            method=mu,
        )[1]

    if di == "<":

        return (
            pl_,
            ql_,
        )

    elif di == ">":

        return (
            pr_,
            qr_,
        )

    elif di == "<>":

        return where(pl_ < pr_, pl_, pr_,), where(
            ql_ < qr_,
            ql_,
            qr_,
        )
