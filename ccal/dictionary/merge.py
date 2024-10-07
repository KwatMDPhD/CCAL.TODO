def merge(di1, di2, fu=None):
    di3 = {}

    for ke in sorted(di1.keys() | di2.keys()):
        if ke in di1 and ke in di2:
            if fu is None:
                va1 = di1[ke]

                va2 = di2[ke]

                if isinstance(va1, dict) and isinstance(va2, dict):
                    di3[ke] = merge(va1, va2)

                else:
                    di3[ke] = va2

            else:
                di3[ke] = fu(di1[ke], di2[ke])

        elif ke in di1:
            di3[ke] = di1[ke]

        elif ke in di2:
            di3[ke] = di2[ke]

    return di3
