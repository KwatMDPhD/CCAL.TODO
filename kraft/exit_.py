from sys import exit

from .echo_or_print_str import echo_or_print_str


def exit_(str_):

    echo_or_print_str(str_, fg="red")

    exit()
