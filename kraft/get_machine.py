from platform import uname


def get_machine():

    uname_ = uname()

    return f"{uname_.system}_{uname_.machine}"
