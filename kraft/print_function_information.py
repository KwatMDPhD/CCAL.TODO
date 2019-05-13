from inspect import stack


def print_function_information():

    frame_info = stack()[1]

    try:

        arguments = (f"{k} = {v}" for k, v in frame_info[0].f_locals.items())

        separater = "\n    "

        print(f"@ {frame_info[3]}{separater}{separater.join(arguments)}")

    finally:

        del frame_info


def _function(a, b=1):

    print_function_information()


if __name__ == "__main__":

    _function(2)
