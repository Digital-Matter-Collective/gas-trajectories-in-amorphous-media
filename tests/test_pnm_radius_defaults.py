from inspect import signature
from typing import Any, Callable

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
        (compute_density, "x_min"),
        (build_3d_distributions, "border"),
        (build_3d_distributions, "x_min"),
        (simulate_pnm_trajectory, "min_radius"),
        (stationarity_analysis, "x_min"),
    ],
)
def test_data_specific_pnm_min_radius_defaults_are_consistent(
    function: Callable[..., Any], parameter: str
) -> None:
    assert Reader.PNM_MIN_RADIUS_NM == 0.003
    assert signature(function).parameters[parameter].default == 0.003


@pytest.mark.parametrize(
    ("function", "parameter"),
    [
        (Reader.read_pnm_data, "border"),
        (load_pnm_graph, "min_radius"),
    ],
)
def test_library_pnm_min_radius_defaults_do_not_filter(
    function: Callable[..., Any], parameter: str
) -> None:
    assert signature(function).parameters[parameter].default == 0.0
