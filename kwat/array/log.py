from numpy import log as loge, log2, log10


def log(ar, ba=2):

    return {2: log2, "e": loge, 10: log10}[
        ba
    ](ar)
