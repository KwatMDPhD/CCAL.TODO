from .check_bad import check_bad


def cast(an):
    if check_bad(an):
        return None

    else:
        for bu in [False, True]:
            if an is bu or an == str(bu):
                return bu

        for ty in [int, float]:
            try:
                return ty(an)

            except ValueError:
                pass

        return an
