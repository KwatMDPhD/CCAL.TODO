from io import UnsupportedOperation

from click import secho


def echo_or_print_str(str_, fg=None):

    try:

        secho(str_, fg=fg)

    except UnsupportedOperation:

        print(str_)
