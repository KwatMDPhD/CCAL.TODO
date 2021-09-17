def summarize(di, n_pa=8):

    print("{} keys and {} unique values:\n".format(len(di), len(set(di.values()))))

    for ke, va in list(di.items())[:n_pa]:

        print("{} => {}".format(ke, va))

    print("...")
