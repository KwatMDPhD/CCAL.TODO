def cast(an):

    for bu in [None, False, True]:

        if an is bu or an == str(bu):

            return bu

    for ty in [int, float]:

        try:

            return ty(an)

        except ValueError:

            pass

    return an
