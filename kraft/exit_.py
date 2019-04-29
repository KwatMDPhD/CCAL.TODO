from sys import exit

from .echo_or_print_str import echo_or_print_str


def exit_(str_, exception=None):

    echo_or_print_str("Uh oh :( ... {}".format(str_), fg="red", bg="black")

    if exception is None:

        exit()

    else:

        raise exception
