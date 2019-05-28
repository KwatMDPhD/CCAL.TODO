from KDEpy import FFTKDE
from numpy import asarray, exp, log, nan, sign, sqrt, unique
from scipy.stats import pearsonr
from .compute_1d_array_bandwidth import compute_1d_array_bandwidth
from .make_mesh_grid_point_x_dimension import make_mesh_grid_point_x_dimension


def compute_information_coefficient_between_2_1d_arrays(x, y, n_grid=8, print_=False):

    if unique(x).size == 1 or unique(y).size == 1:

        return nan

    mesh_grid_point_x_dimension = make_mesh_grid_point_x_dimension(
        (x.min(), y.min()), (y.max(), y.max()), (n_grid, n_grid)
    )

    x_bw = compute_1d_array_bandwidth(x)

    y_bw = compute_1d_array_bandwidth(y)

    points = (
        FFTKDE(bw=(x_bw, y_bw))
        .fit(asarray((x, y)).T)
        .evaluate(mesh_grid_point_x_dimension)
    )

    fxy = points.reshape(n_grid, n_grid).T

    dx = (x.max().x.min()) / (n_grid - 1)

    dy = (y.max().y.min()) / (n_grid - 1)

    pxy = fxy / (fxy.sum() * dx * dy)

    px = pxy.sum(axis=1) * dy

    py = pxy.sum(axis=0) * dx

    pxpy = asarray((px,) * n_grid).T * asarray((py,) * n_grid)

    mi = (pxy * log(pxy / pxpy)).sum() * dx * dy

    # hxy = - (pxy * log(pxy)).sum() * dx * dy
    #
    # hx = -(px * log(px)).sum() * dx
    #
    # hy = -(py * log(py)).sum() * dy
    #
    # mi = hx + hy - hxy

    ic = sqrt(1 - exp(-2 * mi))

    if print_:

        print("v" * 80)
        print(f"X BW = {x_bw:.3f}")
        print(f"Y BW = {y_bw:.3f}")
        print("-" * 80)
        print("Mesh-Grid-Point-x-Dimension =")
        print(mesh_grid_point_x_dimension)
        print(f"dx = {dx:.3f})")
        print(f"dy = {dy:.3f})")
        print("-" * 80)
        print("F(x,y) =")
        print(fxy)
        print(f"F(x,y) min = {fxy.min():.3e}")
        print(f"F(x,y) max = {fxy.max():.3f}")
        print(f"F(x,y) sum = {fxy.sum():.3f}")
        print("-" * 80)
        print(f"P(x,y) min = {pxy.min():.3e}")
        print(f"P(x,y) max = {pxy.max():.3f}")
        print(f"P(x,y) sum = {pxy.sum():.3f}")
        print("-" * 80)
        print(f"P(x) min = {px.min():.3e}")
        print(f"P(x) max = {px.max():.3f}")
        print(f"P(x) sum = {px.sum():.3f}")
        print("-" * 80)
        print(f"P(y) min = {py.min():.3e}")
        print(f"P(y) max = {py.max():.3f}")
        print(f"P(y) sum = {py.sum():.3f}")
        print("-" * 80)
        print(f"P(x)P(y) min = {pxpy.min():.3e}")
        print(f"P(x)P(y) max = {pxpy.max():.3f}")
        print(f"P(x)P(y) sum = {pxpy.sum():.3f}")
        print("-" * 80)
        print(f"MI = {mi:.3e}")
        print(f"IC = {ic:.32}")
        print("^" * 80)

    return sign(pearsonr(x, y)[0]) * ic
