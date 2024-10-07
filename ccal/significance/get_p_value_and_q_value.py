from numpy import array, where

from .get_p_value import get_p_value
from .get_q_value import get_q_value


def get_p_value_and_q_value(ve, ra_, di):
    if "<" in di:
        pl_ = array([get_p_value(nu, ra_, "<") for nu in ve])

        ql_ = get_q_value(pl_)

    if ">" in di:
        pr_ = array([get_p_value(nu, ra_, ">") for nu in ve])

        qr_ = get_q_value(pr_)

    if di == "<":
        return pl_, ql_

    elif di == ">":
        return pr_, qr_

    elif di == "<>":
        return where(pl_ < pr_, pl_, pr_), where(ql_ < qr_, ql_, qr_)
