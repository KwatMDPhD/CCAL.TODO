from .normalize import normalize


def accumulate(vector):

    return (
        normalize(vector.cumsum(), "0-1"),
        normalize(vector[::-1].cumsum()[::-1], "0-1"),
    )
