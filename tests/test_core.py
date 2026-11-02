import numpy as np
import pytest

from fefos import (
    FEFOS,
    BinaryHull,
    IonicComposition,
    Parabola,
    fit_reference_curves,
    lower_convex_hull,
    mix,
    parse_formula,
)
from fefos.legacy import PaperFEFOS


# ---------------------------------------------------------------- composition
def test_parse_formula_with_groups():
    assert parse_formula("Dy(BiO2)3") == {"Dy": 1, "Bi": 3, "O": 6}
    assert parse_formula("Cr0.5Mo0.5O2") == {"Cr": 0.5, "Mo": 0.5, "O": 2}
    with pytest.raises(ValueError):
        parse_formula("Sr(TiO3")


def test_ionic_composition():
    sto = IonicComposition.from_composition("SrTiO3")
    assert sto.anion == "O"
    assert sto.cations == {"Sr": 0.5, "Ti": 0.5}
    assert sto.mean_oxidation_state == pytest.approx(3.0)
    lfp = IonicComposition.from_composition("LiFePO4")
    assert lfp.mean_oxidation_state == pytest.approx(8 / 3)
    assert lfp.has_nonmetal_cation
    assert IonicComposition.from_composition("KMgF3").mean_oxidation_state == pytest.approx(1.5)


# ----------------------------------------------------------------------- hull
def _toy_hull(element="A", points=((1, -2.0), (2, -3.0), (3, -3.2), (4, -3.0))):
    # points are (oxidation state, energy per cation) for an anion of valence 2
    entries = []
    for q, e in points:
        n_x = q / 2
        entries.append((f"{element}{q}", f"{element}X{n_x}", 1.0, n_x, e / (1 + n_x)))
    return BinaryHull.from_entries(element, "X", 2, entries)


def test_lower_convex_hull_removes_points_above():
    hull = lower_convex_hull([(0, 0), (1, -1), (2, -1.5), (1.5, 0), (3, -1.6)])
    assert (1.5, 0.0) not in hull
    assert hull[0] == (0.0, 0.0) and hull[-1] == (3.0, -1.6)


def test_binary_hull_is_flat_beyond_most_stable_compound():
    hull = _toy_hull()
    assert hull.max_stable_oxidation_state == 3  # the q=4 point is above A(3) + X2
    assert hull.energy_per_cation(5.0) == pytest.approx(-3.2)
    # per-atom energy reproduces the input formation energy of a hull compound
    assert hull.energy_per_atom(2.0) == pytest.approx(-3.0 / 2)
    assert hull.energy_per_cation(1.5) == pytest.approx(-2.5)


# --------------------------------------------------------------------- curves
def test_parabola_fit_and_vertex():
    x = np.linspace(0.1, 3, 20)
    p = Parabola.fit(x, 0.3 * x**2 - 0.6 * x, x_max=3)
    assert (p.a, p.b) == pytest.approx((0.3, -0.6))
    assert p.minimum() == pytest.approx((1.0, -0.3))
    v = Parabola.through_minimum(2.0, -1.0)
    assert v(2.0) == pytest.approx(-1.0) and v(4.0) == pytest.approx(0.0)
    concave = Parabola.fit(x, -0.2 * x**2 + x)
    assert concave.a == 0.0  # curvature is never negative


def test_reference_curves_follow_hull():
    hull = _toy_hull()
    r = fit_reference_curves(hull, 2.0, window=1.0)
    assert r.reference_energy == pytest.approx(-3.0)
    # reduction from q=2 to q=1 costs +1 eV per cation along a straight hull segment
    assert r.red(1.0) == pytest.approx(1.0, abs=1e-6)
    assert r.ox(1.0) == pytest.approx(-0.2, abs=1e-6)


# --------------------------------------------------------------------- mixing
def test_paper_rule_matches_published_binary_mixing():
    paper = PaperFEFOS.load()
    a, b, q = "Cr", "Mo", 4
    for z_a in (0.2, 0.5, 0.7):
        ox = [Parabola(*paper.ox[(a, q)], 10.0), Parabola(*paper.ox[(b, q)], 10.0)]
        red = [Parabola(*paper.red[(a, q)], 10.0), Parabola(*paper.red[(b, q)], 10.0)]
        general = mix([z_a, 1 - z_a], ox, red, rule="paper").delta_d
        legacy = paper.delta_d(a, b, q, z_a, snap=False)
        assert general == pytest.approx(legacy, abs=1e-6)


def test_mixing_vanishes_for_pure_end_members_and_is_charge_neutral():
    ox = [Parabola(0.2, -0.5, 4), Parabola(0.1, 0.3, 4), Parabola(0.4, -0.1, 4)]
    red = [Parabola(0.3, 0.2, 3), Parabola(0.2, -0.4, 3), Parabola(0.1, 0.5, 3)]
    for rule in ("paper", "thermodynamic"):
        assert mix([1 - 1e-12, 5e-13, 5e-13], ox, red, rule=rule).delta_d == pytest.approx(
            0, abs=1e-6
        )
    c = np.array([0.5, 0.3, 0.2])
    res = mix(c, ox, red, rule="thermodynamic")
    assert np.dot(c, res.shifts) == pytest.approx(0.0, abs=1e-9)


def test_thermodynamic_mixing_matches_brute_force():
    ox = [Parabola(0.2, -0.5, 4), Parabola(0.1, 0.3, 4)]
    red = [Parabola(0.3, 0.2, 3), Parabola(0.2, 0.1, 3)]
    c = np.array([0.4, 0.6])

    def f(i, d):
        return ox[i](d) if d >= 0 else red[i](-d)

    grid = np.linspace(-3, 3, 60001)
    best = min(
        c[0] * f(0, d0) + c[1] * f(1, -c[0] * d0 / c[1])
        for d0 in grid
        if -3 <= -c[0] * d0 / c[1] <= 4
    )
    assert mix(c, ox, red, rule="thermodynamic").delta_c == pytest.approx(best, abs=1e-6)


# ---------------------------------------------------------------------- model
def _toy_model(mode="linear"):
    hulls = {
        ("A", "X"): _toy_hull("A", ((2, -4.0), (3, -4.4))),
        ("B", "X"): _toy_hull("B", ((2, -2.0), (4, -5.0), (5, -5.3))),
    }
    return FEFOS(hulls, mode=mode)


def test_linear_mode_equals_brute_force_binary_mixture():
    model = _toy_model()
    ionic = IonicComposition.from_composition({"A": 1, "B": 1, "X": 3}, anion="X", valence=2)
    energy, states = model.decompose(ionic)
    c, q = 0.5, ionic.mean_oxidation_state
    ha, hb = model.hulls[("A", "X")], model.hulls[("B", "X")]
    qa = np.linspace(0, q / c, 200001)
    qb = (q - c * qa) / c
    ok = qb >= 0
    e = c * ha.energy_per_cation(qa[ok]) + c * hb.energy_per_cation(qb[ok])
    exact = energy * ionic.atoms_per_cation
    assert exact <= np.nanmin(e) + 1e-12  # exact optimum is never above the grid search
    assert exact == pytest.approx(np.nanmin(e), abs=1e-4)
    assert c * states["A"] + c * states["B"] == pytest.approx(q)


@pytest.mark.parametrize("mode", ["linear", "quadratic", "paper"])
def test_predict_runs_and_reduces_to_reference_for_one_cation(mode):
    model = _toy_model(mode)
    two = model.predict({"A": 1, "B": 1, "X": 3}, anion="X")
    assert two.formation_energy <= two.reference_energy + 1e-9
    one = model.predict({"B": 1, "X": 2}, anion="X")
    assert one.formation_energy == pytest.approx(one.reference_energy)
    assert one.formation_energy == pytest.approx(-5.0 / 3)


# --------------------------------------------------------------------- legacy
def test_published_predictions_are_reproduced():
    from fefos.legacy import reproduce_published

    rep = reproduce_published()
    assert len(rep) == 2327
    diff = (rep.reproduced_prediction - rep.published_prediction).abs()
    assert diff.max() < 2e-4  # published values are rounded to 4 decimals
