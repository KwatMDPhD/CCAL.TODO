def count(da, ax=0):

    if ax == 0:

        it_ = da.iteritems()

    elif ax == 1:

        it_ = da.iterrows()

    for la, se in it_:

        print()

        print(la)

        print(se.value_counts())

        print()
