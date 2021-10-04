from plotly.colors import make_colorscale, qualitative

NAME_COLORSCALE = {
    "binary": make_colorscale(["#006442", "#ffffff", "#ffb61e"]),
    "categorical": make_colorscale(qualitative.Plotly),
    "continuous": make_colorscale(["#0000ff", "#ffffff", "#ff0000"]),
    "human": make_colorscale(["#4b3c39", "#ffffff", "#ffddca"]),
    "stanford": make_colorscale(["#ffffff", "#8c1515"]),
}
