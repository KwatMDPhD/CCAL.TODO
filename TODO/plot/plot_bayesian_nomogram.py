from numpy import absolute, arange, linspace, log2, sort

from .plot_plotly import plot_plotly


def plot_bayesian_nomogram(
    target,
    target_hit,
    target_miss,
    n_grid,
    conditional_probabilities,
    names,
    html_file_path=None,
):

    p_th = (target == target_hit).sum() / target.size

    p_tm = (target == target_miss).sum() / target.size

    grid_t = linspace(target.min(), target.max(), num=n_grid)

    grid_t_ih = absolute(grid_t - target_hit).argmin()

    grid_t_im = absolute(grid_t - target_miss).argmin()

    layout = {
        "height": 64 * max(8, len(conditional_probabilities)),
        "title": {"text": "Bayesian Nomogram"},
        "xaxis": {"title": {"text": "Log Odds Ratio"}},
        "yaxis": {
            "title": {"text": "Evidence"},
            "showticklabels": False,
            "zeroline": False,
        },
    }

    data = []

    for i, (p_t__, name) in enumerate(zip(conditional_probabilities, names)):

        p_th__ = p_t__[:, grid_t_ih]

        p_tm__ = p_t__[:, grid_t_im]

        log_odds_ratios = log2((p_th__ / p_tm__) / (p_th / p_tm))

        x = arange(n_grid)

        plot_plotly(
            {
                "layout": {"title": {"text": name}},
                "data": [
                    {
                        "type": "scatter",
                        "name": name,
                        "x": x,
                        "y": sort(target)[:: target.size // n_grid],
                    },
                    {
                        "type": "scatter",
                        "name": "P(hit = {} | {})".format(target_hit, name),
                        "x": x,
                        "y": p_th__,
                    },
                    {
                        "type": "scatter",
                        "name": "P(miss = {} | {})".format(target_miss, name),
                        "x": x,
                        "y": p_tm__,
                    },
                    {
                        "type": "scatter",
                        "name": "Log Odds Ratio",
                        "x": x,
                        "y": log_odds_ratios,
                    },
                ],
            },
            None,
        )

        data.append(
            {
                "type": "scatter",
                "name": name,
                "x": (log_odds_ratios.min(), log_odds_ratios.max()),
                "y": (i,) * 2,
            }
        )

    plot_plotly({"layout": layout, "data": data}, html_file_path)
