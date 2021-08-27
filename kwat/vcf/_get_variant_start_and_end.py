def _get_variant_start_and_end(po, re, al):

    if len(re) == len(al):

        st = po

        en = po + len(al) - 1

    elif len(re) < len(al):

        st = po

        en = po + 1

    else:

        st = po + 1

        en = po + len(re) - len(al)

    return st, en
