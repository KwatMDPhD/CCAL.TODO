from click import get_terminal_size

from kraft import echo_or_print_str


def print_header_in_terminal(str_):

    terminal_width, terminal_height = get_terminal_size()

    spaces = " " * max(0, (terminal_width - len(str_)) // 2)

    str_ = spaces + str_ + spaces

    if not terminal_width % 2:

        str_ += " "

    spacer = "=" * terminal_width

    echo_or_print_str("{0}\n{1}\n{0}".format(spacer, str_))
