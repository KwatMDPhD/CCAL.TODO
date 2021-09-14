from inspect import stack


def print_stack():

    st = stack()[1]

    print(
        "@{}({})".format(
            st[3],
            ", ".join("{}={}".format(va, an) for va, an in st[0].f_locals.items()),
        )
    )
