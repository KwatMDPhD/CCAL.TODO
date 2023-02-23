from .drop import drop


def drop_while(da, ax=None, **ke_ar):
    sh1 = da.shape

    if ax is None:
        ax = int(sh1[0] < sh1[1])

    re = False

    while True:
        da = drop(da, ax, **ke_ar)

        sh2 = da.shape

        if re and sh1 == sh2:
            return da

        sh1 = sh2

        if ax == 0:
            ax = 1

        elif ax == 1:
            ax = 0

        if not re:
            re = True
