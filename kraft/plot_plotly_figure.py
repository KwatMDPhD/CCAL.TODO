from plotly.io import show, templates, write_html

templates["kraft"] = {
    "layout": {
        # "autosize": True,
    }
}


def plot_plotly_figure(figure, html_file_path=None):

    # template = "plotly_white+kraft"
    template = "plotly_white"

    if "layout" in figure:

        figure["layout"]["template"] = template
    else:

        figure["layout"] = {"template": template}

    config = {"editable": True}

    show(figure, config=config)

    if html_file_path is not None:

        write_html(figure, html_file_path, config=config)
