from sys import exit

from .echo_or_print_str import echo_or_print_str


def exit_(exit_message):

    echo_or_print_str(f"Uh oh :( ... {exit_message}", fg="red", bg="black")

    exit()
