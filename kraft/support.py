from inspect import stack
from platform import uname
from subprocess import PIPE, CalledProcessError, run

from .run_command import run_command


def cast_builtin(object_):

    for builtin_object in (
        None,
        False,
        True,
    ):

        if object_ is builtin_object or object_ == str(builtin_object):

            return builtin_object

    for type_ in (int, float):

        try:

            return type_(object_)

        except ValueError:

            pass

    return object_


def flatten(iterable, types=(tuple, list, set)):

    list_ = list(iterable)

    i = 0

    while i < len(list_):

        while isinstance(list_[i], types):

            if len(list_[i]) != 0:

                list_[i : i + 1] = list_[i]

            else:

                list_.pop(i)

                i -= 1

                break

        i += 1

    return list_


def make_unique(iterable):

    iterable_unique = []

    for x in iterable:

        x_original = x

        i = 2

        while x in iterable_unique:

            x = "{}.{}".format(x_original, i)

            i += 1

        iterable_unique.append(x)

    return iterable_unique


def merge_2_dicts(dict_0, dict_1):

    dict_ = {}

    for key in dict_0.keys() | dict_1.keys():

        if key in dict_0 and key in dict_1:

            value_0 = dict_0[key]

            value_1 = dict_1[key]

            if isinstance(value_0, dict) and isinstance(value_1, dict):

                dict_[key] = merge_2_dicts(value_0, value_1)

            else:

                dict_[key] = value_1

        elif key in dict_0:

            dict_[key] = dict_0[key]

        elif key in dict_1:

            dict_[key] = dict_1[key]

    return dict_


def command(command):

    print(command)

    return run(
        command,
        shell=True,
        stdout=PIPE,
        stderr=PIPE,
        check=True,
        universal_newlines=True,
    )


def merge_2_dicts_with_function(dict_0, dict_1, function):

    dict_ = {}

    for key in dict_0.keys() | dict_1.keys():

        if key in dict_0 and key in dict_1:

            dict_[key] = function(dict_0[key], dict_1[key])

        elif key in dict_0:

            dict_[key] = dict_0[key]

        elif key in dict_1:

            dict_[key] = dict_1[key]

    return dict_


def map_objects_to_ints(iterable):

    object_int = {}

    int_object = {}

    for i, object_ in enumerate(sorted(set(iterable))):

        object_int[object_] = i

        int_object[i] = object_

    return object_int, int_object


def get_machine():

    uname_ = uname()

    return "{}_{}".format(uname_.system, uname_.machine)


def get_shell_environment():

    environemnt = {}

    for line in run_command("env").stdout.split(sep="\n"):

        if line and not line.strip().startswith(":"):

            key, value = line.split(sep="=", maxsplit=1)

            environemnt[key.strip()] = value.strip()

    return environemnt


def install_python_libraries(libraries):

    libraries_installed = tuple(
        line.split()[0]
        for line in run_command("pip list").stdout.strip().split(sep="\n")[2:]
    )

    for library in libraries:

        if library not in libraries_installed:

            run_command("pip install {}".format(library))


def is_program(program_name):

    try:

        return bool(run_command("type {}".format(program_name)).stdout.strip())

    except CalledProcessError:

        return False


def print_function_information():

    frame_info = stack()[1]

    try:

        arguments = (
            "{} = {}".format(k, v) for k, v in sorted(frame_info[0].f_locals.items())
        )

        separater = "\n    "

        print("@ {}{}{}".format(frame_info[3], separater, separater.join(arguments)))

    finally:

        del frame_info
