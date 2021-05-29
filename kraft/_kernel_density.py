from KDEpy import (
    FFTKDE,
)
from statsmodels.nonparametric.kernel_density import (
    KDEMultivariate,
)

from .CONSTANT import (
    FLOAT_RESOLUTION,
)
from .grid import (
    make_g1,
    make_gn,
    plot as grid_plot,
)


def get_bandwidth(
    v,
):

    return KDEMultivariate(
        v,
        "c",
    ).bw[0]


def get_density(pxd, bandwidth_=None, g1_=None, plot=True, **plot_k_):

    dxp = pxd.T

    if bandwidth_ is None:

        bandwidth_ = [get_bandwidth(v) for v in dxp]

    if g1_ is None:

        g1_ = [
            make_g1(
                v.min(),
                v.max(),
                0.1,
                8,
            )
            for v in dxp
        ]

    gn = make_gn(g1_)

    # TODO: consider setting the values whose magnitudes are less than the resolution to be 0
    v = FFTKDE(bw=bandwidth_).fit(pxd).evaluate(gn).clip(FLOAT_RESOLUTION)

    if plot:

        grid_plot(
            gn,
            v,
            value_name="Density",
            **plot_k_,
        )

    return (
        gn,
        v,
    )
