from json import dump


def write(pa, di, ie=2):
    with open(pa, mode="w") as io:
        dump(di, io, indent=ie)
