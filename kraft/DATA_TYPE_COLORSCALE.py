from plotly.colors import make_colorscale, qualitative

DATA_TYPE_COLORSCALE = {
    "continuous": make_colorscale(("#0000ff", "#ffffff", "#ff0000")),
    "categorical": make_colorscale(qualitative.Plotly),
    "binary": make_colorscale(("#ebf6f7", "#171412")),
}
