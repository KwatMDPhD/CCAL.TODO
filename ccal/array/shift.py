def shift(nu___, sh):
    if sh == "+1":
        return nu___ + 1

    if isinstance(sh, str) and sh.endswith("<"):
        fl = float(sh[:-1])

        ab___ = fl < nu___

        sh = nu___[ab___].min()

        print("Shifting the minimum to {}".format(sh))

    return nu___ + sh - nu___.min()
