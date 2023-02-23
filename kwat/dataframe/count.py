def count(da):
    for co, ro_an in da.iteritems():
        print("-" * 80)

        print(co)

        print(ro_an.value_counts())

        print()
