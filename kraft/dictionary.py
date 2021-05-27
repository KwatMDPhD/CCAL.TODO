def summarize(d):

    print("{} k => {} unique value".format(len(d), len(set(d.values()))))


def merge(d1, d2, function=None):

    d3 = {}

    for k in sorted(d1.keys() | d2.keys()):

        if k in d1 and k in d2:

            if function is None:

                v1 = d1[k]

                v2 = d2[k]

                if isinstance(v1, dict) and isinstance(v2, dict):

                    d3[k] = merge(v1, v2)

                else:

                    d3[k] = v2

            else:

                d3[k] = function(d1[k], d2[k])

        elif k in d1:

            d3[k] = d1[k]

        elif k in d2:

            d3[k] = d2[k]

    return d3
