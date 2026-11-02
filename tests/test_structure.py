import numpy as np
import pandas as pd
import pytest

from fefos.data import cation_anion_bonds
from fefos.structure import BondFeatures, BondLengthCorrection, BondLengthTable, bond_features


def _refs():
    return pd.DataFrame(
        [
            # element, anion, formal OS, BV charge, bond length, E_hull
            ("Fe", "O", 2.0, 2.0, 2.16, 0.00),
            ("Fe", "O", 3.0, 3.0, 2.02, 0.00),
            ("Fe", "O", 3.0, 3.0, 2.10, 0.05),  # less stable polymorph at the same charge
            ("Li", "O", 1.0, 1.0, 1.99, 0.00),
            ("Li", "O", 2.0, 1.0, 2.15, 0.02),  # Li2O2: formal +2 but really Li+ (peroxide)
            ("Mg", "O", 2.0, np.nan, 2.11, 0.00),  # no bond-valence charge: formal is used
        ],
        columns=[
            "element",
            "anion",
            "formal_oxidation_state",
            "bv_oxidation_state",
            "bond_length",
            "energy_above_hull",
        ],
    )


def test_table_uses_bond_valence_charge_and_most_stable_reference():
    table = BondLengthTable.from_reference_bonds(_refs())
    qs, ds = table.points[("Fe", "O")]
    assert list(qs) == [2.0, 3.0] and list(ds) == [2.16, 2.02]
    # Li2O2 is filed under Li+ (bond-valence charge); Li2O is more stable and wins
    assert table.points[("Li", "O")] == (pytest.approx([1.0]), pytest.approx([1.99]))
    assert table.assumed_bond_length("Mg", "O", 2.0) == pytest.approx(2.11)


def test_assumed_bond_length_interpolates_and_clamps():
    table = BondLengthTable.from_reference_bonds(_refs())
    assert table.assumed_bond_length("Fe", "O", 2.5) == pytest.approx(2.09)
    assert table.assumed_bond_length("Fe", "O", 1.0) == pytest.approx(2.16)  # below range
    assert table.assumed_bond_length("Fe", "O", 6.0) == pytest.approx(2.02)  # above range
    assert table.assumed_bond_length("Li", "O", 1.22) == pytest.approx(1.99)
    roundtrip = BondLengthTable.from_frame(table.to_frame())
    assert roundtrip.assumed_bond_length("Fe", "O", 2.5) == pytest.approx(2.09)


def test_bond_features():
    table = BondLengthTable.from_reference_bonds(_refs())
    f = bond_features(table, "O", {"Li": 0.5, "Fe": 0.5}, {"Li": 1.0, "Fe": 3.0})
    assert f.bond_lengths == pytest.approx({"Li": 1.99, "Fe": 2.02})
    phi = np.array([1 / 1.99, 3 / 2.02])
    assert f.ionic_potentials["Fe"] == pytest.approx(phi[1])
    assert f.ionic_potential_spread == pytest.approx(abs(phi[1] - phi[0]) / 2)
    assert f.size_mismatch == pytest.approx(0.015 / 2.005)
    same = bond_features(table, "O", {"Fe": 1.0}, {"Fe": 3.0})
    assert same.size_mismatch == 0 and same.ionic_potential_spread == 0


def test_cation_anion_bonds_parses_mp_bond_types():
    bond_types = [
        ("Fe-O", [2.0, 2.2]),
        ("O-Li", [1.9]),
        ("O-O", [1.5]),  # anion-anion: ignored
        ("Fe-Li", [2.9]),  # cation-cation: ignored
        ("Fe-O-x", [9.9]),  # malformed key: ignored
        ("Mn-O", []),  # no bonds: ignored
    ]
    out = cation_anion_bonds(bond_types, "O")
    assert out == {"Fe": (pytest.approx(2.1), 2), "Li": (pytest.approx(1.9), 1)}
    assert cation_anion_bonds({"Fe-O": [2.0]}, "O") == {"Fe": (2.0, 1)}
    assert cation_anion_bonds(None, "O") == {}


def test_structure_bonds_rock_salt():
    pytest.importorskip("pymatgen")
    from pymatgen.core import Lattice, Structure

    from fefos.structure import structure_bonds

    mgo = Structure.from_spacegroup(
        "Fm-3m", Lattice.cubic(4.2), ["Mg", "O"], [[0, 0, 0], [0.5, 0.5, 0.5]]
    )
    out = structure_bonds(mgo, "O")
    assert set(out) == {"Mg"}
    assert out["Mg"]["bond_length"] == pytest.approx(2.1)
    assert out["Mg"]["coordination"] == pytest.approx(6)
    assert out["Mg"]["bv_oxidation_state"] == pytest.approx(2)
    # the dict form (as stored in the MP parquet) works too
    assert structure_bonds(mgo.as_dict(), "O")["Mg"]["bond_length"] == pytest.approx(2.1)


def test_correction_fit_recovers_coefficients_per_anion():
    rng = np.random.default_rng(0)
    n = 400
    feats = rng.uniform(0, 1, (n, 2))
    anions = np.where(np.arange(n) < 200, "O", "S")
    true = {"O": np.array([-0.05, -0.1, -0.4]), "S": np.array([0.02, -0.3, 0.1])}
    y = np.array([true[a] @ [1, *f] for a, f in zip(anions, feats)])
    corr = BondLengthCorrection.fit(anions, feats, y)
    for an in ("O", "S"):
        assert corr.coefficients[an] == pytest.approx(true[an], abs=1e-9)
    f = BondFeatures({}, {}, size_mismatch=0.2, ionic_potential_spread=0.5)
    assert corr("O", f) == pytest.approx(-0.05 - 0.1 * 0.5 - 0.4 * 0.2)
    assert corr("F", f) == 0.0  # no coefficients for this anion
    back = BondLengthCorrection.from_frame(corr.to_frame())
    assert back("S", f) == pytest.approx(corr("S", f))


def test_correction_skips_anions_with_too_few_samples():
    corr = BondLengthCorrection.fit(["O"] * 5, np.zeros((5, 2)), np.zeros(5), min_samples=10)
    assert corr.coefficients == {}


def test_predict_with_bonds_adds_correction_at_fefos_states():
    from fefos import FEFOS, BinaryHull
    from fefos.structure import predict_with_bonds

    def hull(el, pts):
        rows = [(f"{el}{q}", "", 1.0, q / 2, e / (1 + q / 2)) for q, e in pts]
        return BinaryHull.from_entries(el, "O", 2, rows)

    model = FEFOS(
        {("Li", "O"): hull("Li", [(1, -3.0)]), ("Fe", "O"): hull("Fe", [(2, -5.0), (3, -7.0)])}
    )
    table = BondLengthTable.from_reference_bonds(_refs())
    corr = BondLengthCorrection({"O": np.array([-0.01, -0.1, -0.5])})
    r = predict_with_bonds(model, table, corr, {"Li": 1, "Fe": 1, "O": 2})
    assert r.prediction.oxidation_states == pytest.approx({"Fe": 3.0, "Li": 1.0})
    assert r.features.bond_lengths == pytest.approx({"Fe": 2.02, "Li": 1.99})
    expected = -0.01 - 0.1 * r.features.ionic_potential_spread - 0.5 * r.features.size_mismatch
    assert r.correction == pytest.approx(expected)
    assert r.formation_energy == pytest.approx(r.prediction.formation_energy + expected)
    with pytest.raises(ValueError):
        predict_with_bonds(model.with_mode("quadratic"), table, corr, "LiFeO2")
