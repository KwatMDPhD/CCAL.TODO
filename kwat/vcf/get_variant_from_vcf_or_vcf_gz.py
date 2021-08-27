from os.path import isfile

from ..shell import run


def get_variant_from_vcf_or_vcf_gz(pa, ch, st, en):

    if not isfile("{}.tbi".format(pa)):

        run("tabix {}".format(pa))

    return [
        ro.split(sep="\t")
        for ro in run("tabix {} {}:{}-{}".format(pa, ch, st, en))
        .stdout.strip()
        .splitlines()
        if ro != ""
    ]
