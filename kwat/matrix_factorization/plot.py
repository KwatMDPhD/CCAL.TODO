from ..array import normalize
from ..cluster import cluster
from ..constant import GOLDEN_RATIO
from ..plot import plot_heat_map, plot_plotly


def plot(daw_, dah_, er_ie_it=None, si=640, pa=""):

    sig = si * GOLDEN_RATIO

    ar_ = ["-0-"]

    axisf = {"dtick": 1}

    for ie, daw in enumerate(daw_):

        if pa == "":

            pa2 = pa

        else:

            pa2 = "{}w_{}.html".format(pa, ie)

        plot_heat_map(
            daw.iloc[cluster(daw.values)[0], :].apply(normalize, axis=1, args=ar_),
            layout={
                "height": sig,
                "width": si,
                "title": {"text": "W {}".format(ie)},
                "xaxis": axisf,
            },
            pa=pa2,
        )

    for ie, dah in enumerate(dah_):

        if pa == "":

            pa2 = pa

        else:

            pa2 = "{}h_{}.html".format(pa, ie)

        plot_heat_map(
            dah.iloc[:, cluster(dah.values.T)[0]].apply(normalize, axis=0, args=ar_),
            layout={
                "height": si,
                "width": sig,
                "title": {"text": "H {}".format(ie)},
                "yaxis": axisf,
            },
            pa=pa2,
        )

    if er_ie_it is not None:

        if pa == "":

            pa2 = pa

        else:

            pa2 = "{}error.html".format(pa)

        plot_plotly(
            {
                "data": [{"name": ie, "y": er_} for ie, er_ in enumerate(er_ie_it)],
                "layout": {
                    "yaxis": {"title": "Error"},
                    "xaxis": {"title": "Iteration"},
                    "annotations": [
                        {
                            "y": er_[-1],
                            "x": er_.size - 1,
                            "text": "{:.2e}".format(er_[-1]),
                        }
                        for er_ in er_ie_it
                    ],
                },
            },
            pa=pa2,
        )
