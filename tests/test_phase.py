import pytest

pytest.importorskip("scipy")

from fefos.phase import HullCalculator  # noqa: E402


def _toy():
    # A-B-O system: binary oxides AO (-2.0 eV/atom) and BO2 (-2.4), ternary ABO3 (-2.5),
    # an unstable polymorph-free ternary A2BO4 (-2.0) and an intermetallic AB (-0.3)
    return HullCalculator(
        [
            ({"A": 1, "O": 1}, -2.0),
            ({"B": 1, "O": 2}, -2.4),
            ({"A": 1, "B": 1, "O": 3}, -2.5),
            ({"A": 2, "B": 1, "O": 4}, -2.0),
            ({"A": 1, "B": 1}, -0.3),
        ]
    )


def test_hull_energy_of_binary_mixture_and_ternary():
    h = _toy()
    # AO + BO2 has the ABO3 composition: lever rule (2 atoms * -2.0 + 3 atoms * -2.4) / 5
    assert h.hull_energy({"A": 1, "B": 1, "O": 3}, exclude_self=True) == pytest.approx(-2.24)
    assert h.hull_energy({"A": 1, "B": 1, "O": 3}) == pytest.approx(-2.5)
    assert h.hull_energy({"A": 1}) == pytest.approx(0.0)
    assert h.hull_energy({"A": 2, "B": 2}) == pytest.approx(-0.3)


def test_decomposition_energy_sign():
    h = _toy()
    # ABO3 is stable: 0.26 eV/atom below AO + BO2
    assert h.decomposition_energy({"A": 1, "B": 1, "O": 3}, -2.5) == pytest.approx(-0.26)
    # A2BO4 = AO + ABO3 (2 + 5 atoms): hull -15.5/7; the entry at -2.0 lies above it
    e_hull = (2 * -2.0 + 5 * -2.5) / 7
    assert h.decomposition_energy({"A": 2, "B": 1, "O": 4}, -2.0) == pytest.approx(-2.0 - e_hull)
    assert h.decomposition_energy({"A": 2, "B": 1, "O": 4}, -2.0) > 0
