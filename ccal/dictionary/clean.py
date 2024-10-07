from ..python import check_bad


def clean(di):
    print("Before cleaning: {}".format(len(di)))

    dig = {ke: va for ke, va in di.items() if not check_bad(va)}

    print("After: {}".format(len(di)))

    return dig
