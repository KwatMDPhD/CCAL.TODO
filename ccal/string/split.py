def split(st, se=" "):
    sp_ = []

    qu = ""

    for sp in st.split(sep=se):
        if '"' in sp or "'" in sp:
            if qu == "":
                qu = sp

            else:
                qu += se + sp

                sp_.append(qu)

                qu = ""

        else:
            if qu == "":
                sp_.append(sp)

            else:
                qu += sp

    if qu != "":
        sp_.append(qu)

    return sp_
