from plotly.io import show, write_html, write_image

from ..dictionary import merge


def plot_plotly(figure, pr=""):

    axis = {"automargin": True}

    figure = merge(
        {
            "layout": {
                "autosize": False,
                "template": "plotly_white",
                "xaxis": axis,
                "yaxis": axis,
            }
        },
        figure,
    )

    config = {
        # "editable": True,
        # "modeBarButtonsToAdd": ["drawopenpath", "eraseshape"],
        "displaylogo": False
    }

    show(figure, config=config)

    if pr != "":

        write_html(figure, "{}.html".format(pr), config=config)

        write_image(figure, "{}.png".format(pr))
