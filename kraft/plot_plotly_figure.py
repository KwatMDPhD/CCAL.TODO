from plotly.io import show, write_html


def plot_plotly_figure(figure, html_file_path=None):

    if "layout" in figure:

        figure["layout"].update(
            {"template": "plotly_white", "autosize": True, "hovermode": "closest"}
        )

        for axis in ("xaxis", "yaxis"):

            if axis in figure["layout"] and figure["layout"][axis] is not None:

                figure["layout"][axis].update({"automargin": True})

    else:

        figure["layout"] = {
            "template": "plotly_white",
            "autosize": True,
            "hovermode": "closest",
            "xaxis": {"automargin": True},
            "yaxis": {"automargin": True},
        }

    config = {"editable": True}

    show(figure, config=config)

    if html_file_path is not None:

        write_html(figure, html_file_path, config=config)
