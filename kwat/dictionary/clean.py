from ..python import check_bad


def clean(ke_va):

    print("Before: {}".format(len(ke_va)))

    ke_vag = {ke: va for ke, va in ke_va.items() if not check_bad(va)}

    print("After: {}".format(len(ke_va)))

    return ke_vag
