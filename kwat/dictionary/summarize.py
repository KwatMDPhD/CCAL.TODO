def summarize(di, n_pr=8):

    print("{} keys and {} unique values:\n".format(len(di), len(set(di.values()))))

    for ie, (ke, va) in enumerate(di.items()):

        print("{} => {}".format(ke, va))

        if (ie + 1) == n_pr:

            print("...")

            break
