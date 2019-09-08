from click import get_terminal_size

from kraft import echo_or_print_str


def print_header_in_terminal(str_):

    terminal_width, terminal_height = get_terminal_size()

    str_length = len(str_)

    if str_length < terminal_width:

        str_ = "{0}{1}{0}".format(" " * (terminal_width - str_length) // 2, str_)

    echo_or_print_str("{0}\n{1}\n{0}".format("=" * terminal_width, str_))
