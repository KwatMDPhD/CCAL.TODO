from inspect import (
    stack,
)


def print_stack_state():

    st = stack()[1]

    print(
        "@{}({})".format(
            st[3],
            ", ".join(
                "{}={}".format(
                    va,
                    an,
                )
                for (
                    va,
                    an,
                ) in st[0].f_locals.items()
            ),
        ),
    )


def cast_builtin(
    an,
):

    for bu in [
        None,
        False,
        True,
    ]:

        if an is bu or an == str(bu):

            return bu

    for ty in [
        int,
        float,
    ]:

        try:

            return ty(an)

        except ValueError:

            pass

    return an
