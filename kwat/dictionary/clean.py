from ..python import check_is_bad


def clean(ke_va):

    print(len(ke_va))

    ke_van = {ke: va for ke, va in ke_va.items() if not check_is_bad(va)}

    print(len(ke_van))

    return ke_van
