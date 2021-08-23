from .plot_plotly import plot_plotly


def plot_point(an_po_pa, annotation_font_size=16, title="", pa=""):

    data = [
        {
            "name": "Point",
            "y": an_po_pa.iloc[:, 0],
            "x": an_po_pa.iloc[:, 1],
            "text": an_po_pa.index,
            "mode": "markers",
            "marker": {
                "size": an_po_pa["Size"],
                "color": an_po_pa["Color"],
                "opacity": an_po_pa["Opacity"],
                "line": {
                    "width": 0,
                },
            },
        }
    ]

    annotations = []

    for text, co_ in an_po_pa.iloc[:, :2].loc[an_po_pa["Annotate"]].iterrows():

        annotations.append(
            {
                "y": co_[0],
                "x": co_[1],
                "text": text,
                "font": {
                    "size": annotation_font_size,
                },
                "arrowhead": 2,
                "arrowsize": 1,
                "clicktoshow": "onoff",
            }
        )

    plot_plotly(
        {
            "data": data,
            "layout": {
                "title": title,
                "yaxis": {
                    "title": an_po_pa.columns[0],
                },
                "xaxis": {
                    "title": an_po_pa.columns[1],
                },
                "annotations": annotations,
            },
        },
        pa=pa,
    )
