from inspect import stack


def print_function_information():

    frame_info = stack()[1]

    try:

        arguments = (f"{k} = {v}" for k, v in sorted(frame_info[0].f_locals.items()))

        separater = "\n    "

        print(f"@ {frame_info[3]}{separater}{separater.join(arguments)}")

    finally:

        del frame_info
