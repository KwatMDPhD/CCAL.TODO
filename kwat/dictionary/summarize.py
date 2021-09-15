def summarize(ke_va, n_pr=8):

    print(
        "{} keys and {} unique values:\n".format(len(ke_va), len(set(ke_va.values())))
    )

    for ke, va in list(ke_va.items())[:n_pr]:

        print("{} => {}".format(ke, va))

    print("...")
