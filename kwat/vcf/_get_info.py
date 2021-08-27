def _get_info(io, ke):

    for ios in io.split(sep=";"):

        if "=" in ios:

            ke2, va = ios.split(sep="=")

            if ke2 == ke:

                return va
