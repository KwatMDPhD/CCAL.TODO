def merge(dict0, dict1):

    dict01 = {}

    for key in dict0.keys() | dict1.keys():

        if key in dict0 and key in dict1:

            value0 = dict0[key]

            value1 = dict1[key]

            if isinstance(value0, dict) and isinstance(value1, dict):

                dict01[key] = merge(value0, value1)

            else:

                dict01[key] = value1

        elif key in dict0:

            dict01[key] = dict0[key]

        elif key in dict1:

            dict01[key] = dict1[key]

    return dict01


def merge_with_function(dict0, dict1, function):

    dict01 = {}

    for key in dict0.keys() | dict1.keys():

        if key in dict0 and key in dict1:

            dict01[key] = function(dict0[key], dict1[key])

        elif key in dict0:

            dict01[key] = dict0[key]

        elif key in dict1:

            dict01[key] = dict1[key]

    return dict01
