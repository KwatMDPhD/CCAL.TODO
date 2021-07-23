from json import dump, load


def read(pa):

    with open(pa) as io:

        return load(io)


def write(pa, an_an, ie=2):

    with open(pa, "w") as io:

        dump(an_an, io, indent=ie)
