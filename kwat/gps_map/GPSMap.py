from numpy import full, mean, nan, unique
from pandas import DataFrame
from scipy.spatial import Delaunay

from ..constant import RANDOM_SEED
from ..density import get_bandwidth
from ..grid import make_1d_grid
from ..plot import CATEGORICAL_COLORSCALE, plot_heat_map
from ..point import plot, pull, scale
from ..probability import get_probability


class GPSMap:
    def __init__(self, di_no_no, nu_po_no, node_marker_size=24, ra=RANDOM_SEED):

        self.nu_no_di = scale(di_no_no, 2, ra=ra)

        self.nu_po_no = nu_po_no

        self.nu_po_di = pull(self.nu_no_di, self.nu_po_no.values)

        self.node_marker_size = node_marker_size

        self.gr_ = None

        self.co_ = None

        self.bap_ = None

        self.bag_ = None

        self.colorscaleg = None

    def plot(self, **ke_ar):

        if "po_sc" in ke_ar:

            gr_ = None

        else:

            gr_ = self.gr_

        plot(
            DataFrame(data=self.nu_no_di, index=self.nu_po_no.columns),
            DataFrame(data=self.nu_po_di, index=self.nu_po_no.index),
            gr_=gr_,
            colorscaleg=self.colorscaleg,
            co_=self.co_,
            bap_=self.bap_,
            bag_=self.bag_,
            tracen={"marker": {"size": self.node_marker_size}},
            **ke_ar,
        )

    def set_group(self, gr_, colorscale=CATEGORICAL_COLORSCALE, n_co=64):

        if isinstance(gr_, str) and gr_ == "closest_node":

            gr_ = self.nu_po_no.values.argmax(axis=1) + 1

        self.gr_ = gr_

        self.colorscaleg = colorscale

        sh = [n_co] * 2

        ma = full(sh, nan)

        de = Delaunay(self.nu_no_di)

        self.co_ = make_1d_grid(0, 1, 1e-3, n_co)

        for ie1 in range(n_co):

            for ie2 in range(n_co):

                ma[ie1, ie2] = de.find_simplex([self.co_[ie1], self.co_[ie2]])

        gr_bap_ = {}

        ba = mean(get_bandwidth(self.nu_po_di, me="silverman"))

        co__ = [self.co_] * 2

        for gr in unique(self.gr_):

            gr_bap_[gr] = get_probability(
                self.nu_po_di[self.gr_ == gr], co__=co__, pl=False, bw=ba
            )[1].reshape(sh)

        self.bap_ = full(sh, nan)

        self.bag_ = full(sh, 0, dtype=int)

        for ie1 in range(n_co):

            for ie2 in range(n_co):

                if ma[ie1, ie2] != -1:

                    prb = 0.0

                    grb = 0

                    for gr, bap_ in gr_bap_.items():

                        pr = bap_[ie1, ie2]

                        if prb < pr:

                            prb = pr

                            grb = gr

                    self.bap_[ie1, ie2] = prb

                    self.bag_[ie1, ie2] = grb

        plot_heat_map(
            self.nu_po_no.T,
            gr2_=self.gr_,
            colorscale2=self.colorscaleg,
            layout={"yaxis": {"dtick": 1}},
        )

    def predict(self, nap, po_, nu_po_no, **ke_ar):

        plot(
            DataFrame(data=self.nu_no_di, index=self.nu_po_no.columns),
            DataFrame(data=pull(self.nu_no_di, nu_po_no.values), index=nu_po_no.index),
            gr_=None,
            colorscaleg=self.colorscaleg,
            co_=self.co_,
            bap_=self.bap_,
            bag_=self.bag_,
            tracen={"marker": {"size": self.node_marker_size}},
            **ke_ar,
        )
