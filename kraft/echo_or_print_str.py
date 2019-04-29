from io import UnsupportedOperation
from random import choice

from click import secho


def echo_or_print_str(
    str_,
    fg=None,
    bg=None,
    bold=None,
    dim=None,
    underline=None,
    blink=None,
    reverse=None,
    reset=True,
):

    if fg == "random":

        fg = choice(("green", "yellow", "blue", "magenta", "cyan", "white"))

        bg = "black"

    try:

        secho(
            str_,
            fg=fg,
            bg=bg,
            bold=bold,
            dim=dim,
            underline=underline,
            blink=blink,
            reverse=reverse,
            reset=reset,
        )

    except UnsupportedOperation:

        print(str_)
