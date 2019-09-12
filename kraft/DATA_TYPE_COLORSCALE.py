from plotly.colors import make_colorscale

DATA_TYPE_COLORSCALE = {
    "continuous": make_colorscale(("#0000ff", "#ffffff", "#ff0000")),
    "categorical": "portland",
    "binary": make_colorscale(("#ebf6f7", "#171412")),
}
