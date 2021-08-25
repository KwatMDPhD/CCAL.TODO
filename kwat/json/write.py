from json import dump


def write(pa, an_an, ie=2):

    with open(pa, mode="w") as io:

        dump(an_an, io, indent=ie)
