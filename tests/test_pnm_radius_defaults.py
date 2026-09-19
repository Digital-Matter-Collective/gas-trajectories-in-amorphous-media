from inspect import signature

import pytest

from base.reader import Reader
from processes.pnm_walk_simulator import load_pnm_graph
from scripts.distr_pnm_element_size_plotter import (
    build_3d_distributions,
    compute_density,
)
from scripts.simulate_pnm_trajectory import run as simulate_pnm_trajectory
from scripts.stationarity import analysis as stationarity_analysis


@pytest.mark.parametrize(
    ("function", "parameter"),
    [
        (Reader.read_pnm_data, "border"),
        (compute_density, "x_min"),
        (build_3d_distributions, "border"),
        (build_3d_distributions, "x_min"),
        (load_pnm_graph, "min_radius"),
        (simulate_pnm_trajectory, "min_radius"),
        (stationarity_analysis, "x_min"),
    ],
)
def test_pnm_min_radius_defaults_are_consistent(
    function: object, parameter: str
) -> None:
    assert Reader.PNM_MIN_RADIUS_NM == 0.003
    assert signature(function).parameters[parameter].default == 0.003
