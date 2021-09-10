from numpy import full, nan, unique
from scipy.spatial import Delaunay

from ..constant import random_seed
from ..density import get_bandwidth
from ..grid import make_1d_grid
from ..plot import categorical_colorscale, plot_heat_map
from ..point import map_point, plot_node_point, pull_point
from ..probability import get_probability


class GPSMap:
    def __init__(
        self,
        non,
        no_,
        nu_no_no,
        pon,
        po_,
        nu_po_no,
        node_marker_size=24,
        ra=random_seed,
    ):

        self.non = non

        self.no_ = no_

        self.nu_no_di = map_point(nu_no_no, 2, ra=ra)

        self.pon = pon

        self.po_ = po_

        self.nu_po_no = nu_po_no

        self.nu_po_di = pull_point(self.nu_no_di, self.nu_po_no)

        self.node_marker_size = node_marker_size

        self.gr_ = None

        self.co_ = None

        self.pr_ = None

        self.go_ = None

        self.grc = None

    def plot(self, **ke_va):

        plot_node_point(
            self.non,
            self.no_,
            self.nu_no_di,
            self.pon,
            self.po_,
            self.nu_po_di,
            gr_=self.gr_,
            grc=self.grc,
            co_=self.co_,
            pr_=self.pr_,
            go_=self.go_,
            node_trace={
                "marker": {
                    "size": self.node_marker_size,
                },
            },
            **ke_va,
        )

    def set_group(self, gr_, grc=categorical_colorscale, n_co=128):

        if gr_ == "closest_node":

            gr_ = self.nu_po_no.argmax(axis=1)

        self.gr_ = gr_

        self.grc = grc

        sh = [n_co] * 2

        ma = full(sh, nan)

        de = Delaunay(self.nu_no_di)

        self.co_ = make_1d_grid(0, 1, 1e-3, n_co)

        for ie1 in range(n_co):

            for ie2 in range(n_co):

                ma[ie1, ie2] = de.find_simplex([self.co_[ie1], self.co_[ie2]])

        gr_pr_ = {}

        ba_ = [get_bandwidth(di) for di in self.nu_po_di.T]

        co_ = [self.co_] * 2

        for gr in unique(self.gr_):

            gr_pr_[gr] = get_probability(
                self.nu_po_di[self.gr_ == gr],
                plot=False,
                ba_=ba_,
                co_=co_,
            )[1].reshape(sh)

        self.pr_ = full(sh, nan)

        self.go_ = full(sh, nan)

        for ie1 in range(n_co):

            for ie2 in range(n_co):

                if ma[ie1, ie2] != -1:

                    prb = 0

                    grb = nan

                    for gr, pr_ in gr_pr_.items():

                        pr = pr_[ie1, ie2]

                        if prb < pr:

                            prb = pr

                            grb = gr

                    self.pr_[ie1, ie2] = prb

                    self.go_[ie1, ie2] = grb

        plot_heat_map(
            self.nu_po_no.T,
            self.no_,
            self.po_,
            self.non,
            self.pon,
            axis_1_group_=self.gr_,
            axis_1_group_colorscale=self.grc,
            layout={
                "yaxis": {
                    "dtick": 1,
                },
            },
        )

    def predict(self, nap, po_, nu_po_no, **ke_va):

        plot_node_point(
            self.non,
            self.no_,
            self.nu_no_di,
            nap,
            po_,
            pull_point(self.nu_no_di, nu_po_no),
            gr_=None,
            grc=self.grc,
            co_=self.co_,
            pr_=self.pr_,
            go_=self.go_,
            node_trace={
                "marker": {
                    "size": self.node_marker_size,
                },
            },
            **ke_va,
        )
