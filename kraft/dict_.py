def merge(dict_0, dict_1, function=None):

    dict_01 = {}

    for key in dict_0.keys() | dict_1.keys():

        if key in dict_0 and key in dict_1:

            if function is None:

                value_0 = dict_0[key]

                value_1 = dict_1[key]

                if isinstance(value_0, dict) and isinstance(value_1, dict):

                    dict_01[key] = merge(value_0, value_1)

                else:

                    dict_01[key] = value_1

            else:

                dict_01[key] = function(dict_0[key], dict_1[key])

        elif key in dict_0:

            dict_01[key] = dict_0[key]

        elif key in dict_1:

            dict_01[key] = dict_1[key]

    return dict_01
