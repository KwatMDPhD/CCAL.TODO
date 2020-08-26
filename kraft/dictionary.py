def summarize(dictionary):

    print(
        "{} keys => {} unique values".format(
            len(dictionary), len(set(dictionary.values()))
        )
    )


def merge(dictionary_0, dictionary_1, function=None):

    dictionary = {}

    for key in sorted(dictionary_0.keys() | dictionary_1.keys()):

        if key in dictionary_0 and key in dictionary_1:

            if function is None:

                value_0 = dictionary_0[key]

                value_1 = dictionary_1[key]

                if isinstance(value_0, dict) and isinstance(value_1, dict):

                    dictionary[key] = merge(value_0, value_1)

                else:

                    dictionary[key] = value_1

            else:

                dictionary[key] = function(dictionary_0[key], dictionary_1[key])

        elif key in dictionary_0:

            dictionary[key] = dictionary_0[key]

        elif key in dictionary_1:

            dictionary[key] = dictionary_1[key]

    return dictionary
