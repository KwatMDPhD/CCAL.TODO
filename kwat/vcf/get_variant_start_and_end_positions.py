def get_variant_start_and_end_positions(po, re, al):

    if len(re) == len(al):

        st, en = po, po + len(al) - 1

    elif len(re) < len(al):

        st, en = po, po + 1

    else:

        st, en = po + 1, po + len(re) - len(al)

    return st, en
