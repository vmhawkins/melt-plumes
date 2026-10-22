"""Test cases for the bpt module."""

import numpy as np
import scipy.integrate as itgr
import scipy.interpolate as itpl
import seawater as sw
import pytest

from melt_plumes import bpt, helpers, solve


def test_linearized_freezing_point_equation() -> None:
    pass


def test_plume_homogeneous_ocean_without_melt() -> None:
    """"""
    w, R, T_o, S_o, z_out = solve.plume(-165, 100, 0.5)


def test_plume_stratified_ocean_without_melt() -> None:
    """"""
    d, S, T = helpers.load_LeConte_CTD()

    S = itpl.interp1d(-d, S)
    T = itpl.interp1d(-d, T)

    w, R, T_o, S_o, z_out = solve.plume(-165, 100, 0.5, -d[0], T, S)
    np.testing.assert_allclose(w[-1], 1.0694846682525059, rtol=1e-4)
    np.testing.assert_allclose(R[-1], 19.458316949436956, rtol=1e-4)
    np.testing.assert_allclose(T_o[-1], 7.038140255058137, rtol=1e-4)
    np.testing.assert_allclose(S_o[-1], 26.881824068698358, rtol=1e-4)


def test_melt_rate() -> None:
    pass

def test_point_plume() -> None:
    w, R, T_o, S_o, z_out = solve.plume(-165, 100, 0.5, geometry="point")

    for arr in (w, R, T_o, S_o, z_out):
        assert np.all(np.isfinite(arr))
    # entrainment should make the plume grow with height
    assert np.all(np.diff(R) > 0)


def test_plume_point_differs_from_line() -> None:
    line = solve.plume(-165, 100, 0.5, geometry="line")
    point = solve.plume(-165, 100, 0.5, geometry="point")
    assert not np.allclose(line[1][-1], point[1][-1])


def test_plume_invalid_geometry_raises() -> None:
    with pytest.raises(ValueError):
        solve.plume(-165, 100, 0.5, geometry="infinite")