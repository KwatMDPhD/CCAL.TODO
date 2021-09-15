def merge(ke_va1, ke_va2, fu=None):

    ke_va3 = {}

    for ke in sorted(ke_va1.keys() | ke_va2.keys()):

        if ke in ke_va1 and ke in ke_va2:

            if fu is None:

                va1 = ke_va1[ke]

                va2 = ke_va2[ke]

                if isinstance(va1, dict) and isinstance(va2, dict):

                    ke_va3[ke] = merge(va1, va2)

                else:

                    ke_va3[ke] = va2

            else:

                ke_va3[ke] = fu(ke_va1[ke], ke_va2[ke])

        elif ke in ke_va1:

            ke_va3[ke] = ke_va1[ke]

        elif ke in ke_va2:

            ke_va3[ke] = ke_va2[ke]

    return ke_va3
