from .sample_dataframe import sample_dataframe
from .plot_heat_map import plot_heat_map
from .compute_matrix_norm import compute_matrix_norm


def check_mf(v, w, h, *sample_dataframe_arguments):

    wh = w @ h

    norm = compute_matrix_norm(v - wh)

    v_sample = sample_dataframe(v, *sample_dataframe_arguments)

    plot_heat_map(v_sample, layout={"title": {"text": "V"}})

    wh_sample = wh.loc[v_sample.index, v_sample.columns]

    plot_heat_map(wh_sample, layout={"title": {"text": "W * H"}})

    plot_heat_map(
        v_sample - wh_sample,
        layout={"title": {"text": "W * H - V<br>norm = {:.2e}".format(norm)}},
    )
