"""Collect the numbers and samples used in the validation report.

    python scripts/make_report_data.py   # after scripts/run_benchmark.py

Writes ``results/report_data.json``.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from fefos import IonicComposition, parse_formula
from fefos.benchmark import metrics
from fefos.cli import load_model
from fefos.data import DEFAULT_MP_VERSION, load_dataset, parse_composition_column
from fefos.elements import electronegativity

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
MODELS = ["fefos_2023", "fefos_paper", "fefos_quadratic", "fefos_linear", "reference_average"]


def table(df: pd.DataFrame, models=MODELS) -> dict:
    out = {}
    for m in models:
        met = metrics(df[m], df["mp"])
        if met.n:
            out[m] = {
                k: round(v, 4) if isinstance(v, float) else v for k, v in met.as_dict().items()
            }
    return out


OER_METALS = {"Ir", "Ru", "Rh", "Pd", "Pt", "Co", "Ni", "Fe", "Mn", "Cu"}


def screening_summary(s: pd.DataFrame) -> dict:
    new = s[~s.in_mp]
    known = s[s.in_mp & s.charge_consistent & s.in_domain]
    stable = known.mp_dhd < 0
    calib = [
        {
            "threshold": None,
            "n": int(len(known)),
            "stable": round(float(stable.mean()), 3),
            "near_hull": round(float((known.mp_dhd < 0.05).mean()), 3),
        }
    ]
    for thr in (0.0, -0.05, -0.10):
        m = known.dhd_pred < thr
        calib.append(
            {
                "threshold": thr,
                "n": int(m.sum()),
                "stable": round(float(stable[m].mean()), 3),
                "near_hull": round(float((known.mp_dhd[m] < 0.05).mean()), 3),
            }
        )
    short = s[s.shortlist].copy()
    xcheck_path = RES / "screening_crosscheck.csv.gz"
    if xcheck_path.exists():
        x = pd.read_csv(xcheck_path).set_index("formula")
        short["residual_model_dhd"] = x.residual_model_dhd.reindex(short.formula).to_numpy()
        short["consensus"] = (
            x.consensus.reindex(short.formula).fillna(False).astype(bool).to_numpy()
        )
    cols = [
        "formula",
        "A",
        "B",
        "dhd_pred",
        "fefos_bonds",
        "hull_energy",
        "fefos_oxidation_states",
        "n_known_ternaries",
        "toxic",
        "precious",
    ]
    short["oer_metal"] = [bool({a, b} & OER_METALS) for a, b in zip(short.A, short.B)]
    return {
        "n_candidates": int(len(s)),
        "n_in_mp": int(s.in_mp.sum()),
        "n_new": int(len(new)),
        "n_new_consistent_in_domain": int((new.charge_consistent & new.in_domain).sum()),
        "n_shortlist": int(len(short)),
        "n_systems": int((short.A + "-" + short.B).nunique()),
        "calibration_in_mp": calib,
        "shortlist": short[
            [c for c in cols + ["oer_metal", "residual_model_dhd", "consensus"] if c in short]
        ]
        .round(4)
        .to_dict("records"),
    }


def main() -> None:
    r = pd.read_csv(RES / "benchmark_predictions.csv.gz")
    summary = json.loads((RES / "summary.json").read_text())
    binaries, compounds = load_dataset(ROOT / "data" / f"mp_{DEFAULT_MP_VERSION}")
    q = r.mean_oxidation_state
    ionic = r[(q >= 1) & ~r.nonmetal_cation]
    out: dict = {
        "mp_version": DEFAULT_MP_VERSION,
        "n_binaries": int(len(binaries)),
        "n_hulls": int(binaries.groupby(["element", "anion"]).ngroups),
        "n_compounds": int(len(compounds)),
        "n_representable": int(len(r)),
        "reproduction": summary["paper_2023_reproduction"],
    }

    # 1. head-to-head on the 2023 model's own domain
    common = r[r.paper_scope & r.fefos_2023.notna()]
    out["head_to_head"] = {
        "all": table(common),
        "ICSD (experimental)": table(common[~common.theoretical]),
        "stable (on MP hull)": table(common[common.is_stable]),
        "near hull (<0.1 eV/atom)": table(common[common.energy_above_hull < 0.1]),
    }

    # 2. the extension
    ext = MODELS[1:]
    out["scope"] = {
        "ionic, metal cations": table(ionic, ext),
        "stable (on MP hull)": table(ionic[ionic.is_stable], ext),
        "ICSD (experimental)": table(ionic[~ionic.theoretical], ext),
        "near hull (<0.1 eV/atom)": table(ionic[ionic.energy_above_hull < 0.1], ext),
    }
    out["by_anion"] = {
        an: {"n": int(len(g)), **{m: round(float((g[m] - g.mp).abs().mean()), 4) for m in ext}}
        for an, g in ionic.groupby("anion")
    }
    nc = ionic.n_cations.clip(upper=5)
    out["by_n_cations"] = {
        int(k): {"n": int(len(g)), **{m: round(float((g[m] - g.mp).abs().mean()), 4) for m in ext}}
        for k, g in ionic.groupby(nc)
    }
    classes = {
        "ionic, metal cations": ionic,
        "polyanion / mixed anion": r[(q >= 1) & r.nonmetal_cation],
        "intermetallic-like (mean OS < 1)": r[q < 1],
    }
    out["classes"] = {k: table(g, ext) for k, g in classes.items()}

    # 3. parity samples
    rng = np.random.default_rng(0)
    s = ionic.iloc[rng.choice(len(ionic), 1500, replace=False)]
    out["parity_extension"] = [
        [
            row.formula,
            row.anion,
            int(row.n_cations),
            round(row.mp, 3),
            round(row.fefos_linear, 3),
            round(row.reference_average, 3),
        ]
        for row in s.itertuples(index=False)
    ]
    out["parity_paper"] = [
        [row.formula, round(row.mp, 3), round(row.fefos_2023, 3), round(row.fefos_linear, 3)]
        for row in common.itertuples(index=False)
    ]

    # 4. Cr–Mo worked example (Fig. 1 of the paper)
    models = {m: load_model(mode=m) for m in ("linear", "quadratic", "paper")}
    hull_cr, hull_mo = models["linear"].hulls[("Cr", "O")], models["linear"].hulls[("Mo", "O")]
    curves_cr = models["paper"].curves("Cr", "O", 4.0)
    curves_mo = models["paper"].curves("Mo", "O", 4.0)
    zs = np.linspace(0, 1, 41)
    dd = {}
    for m, model in models.items():
        vals = []
        for z in zs:
            if z in (0.0, 1.0):
                vals.append(0.0)
                continue
            p = model.predict({"Mo": 1 - z, "Cr": z, "O": 2})
            vals.append(round(p.delta_d, 4))
        dd[m] = vals
    mp_points = []
    for row in compounds[(compounds.anion == "O") & (compounds.n_cations == 2)].itertuples(
        index=False
    ):
        comp = parse_composition_column(row.composition)
        if set(comp) == {"Cr", "Mo", "O"} and abs(row.mean_oxidation_state - 4) < 1e-6:
            p = models["linear"].predict(comp)
            ion = IonicComposition.from_composition(comp)
            mp_points.append(
                [
                    row.formula,
                    round(ion.cations["Cr"], 4),
                    round(row.formation_energy_per_atom - p.reference_energy, 4),
                ]
            )
    out["cr_mo_example"] = {
        "hull_cr": [
            [round(a, 3), round(float(hull_cr.energy_per_atom(a)), 4)] for a in hull_cr.vertices_q
        ],
        "hull_mo": [
            [round(a, 3), round(float(hull_mo.energy_per_atom(a)), 4)] for a in hull_mo.vertices_q
        ],
        "cr_red": [curves_cr.red.a, curves_cr.red.b, curves_cr.red.x_max],
        "mo_ox": [curves_mo.ox.a, curves_mo.ox.b, curves_mo.ox.x_max],
        "paper_cr_red": [0.319, -0.638],
        "paper_mo_ox": [0.022, 0.003],
        "z": [round(z, 4) for z in zs],
        "delta_d": dd,
        "mp_points": mp_points,
    }

    # 5. example compounds
    lin = models["linear"]

    def key(comp: dict) -> tuple:
        total = sum(comp.values())
        return tuple(sorted((el, round(n / total, 6)) for el, n in comp.items()))

    by_comp = {
        key(parse_composition_column(c)): e
        for c, e in zip(compounds.composition, compounds.formation_energy_per_atom)
    }
    examples = []
    for f in [
        "SrTiO3",
        "BaTiO3",
        "LiCoO2",
        "MgAl2O4",
        "NiFe2O4",
        "LiNbO3",
        "Ba2YCu3O7",
        "CuFeS2",
        "AgInSe2",
        "Li2ZrN2",
        "KMgF3",
        "Cs2AgBiBr6",
        "LiAlH4",
        "Sr2FeMoO6",
    ]:
        p = lin.predict(f)
        mp_e = by_comp.get(key(parse_formula(f)))
        examples.append(
            {
                "formula": f,
                "anion": p.anion,
                "mean_os": round(p.mean_oxidation_state, 3),
                "mp": None if mp_e is None else round(float(mp_e), 3),
                "linear": round(p.formation_energy, 3),
                "quadratic": round(models["quadratic"].predict(f).formation_energy, 3),
                "paper": round(models["paper"].predict(f).formation_energy, 3),
                "reference": round(p.reference_energy, 3),
                "oxidation_states": {k: round(v, 2) for k, v in p.oxidation_states.items()},
            }
        )
    out["examples"] = examples

    # 6. what is left: residual vs electronegativity spread
    def chi_var(formula_row):
        comp = parse_composition_column(formula_row)
        ion = IonicComposition.from_composition(comp)
        c = np.array(list(ion.cations.values()))
        x = np.array([electronegativity(e) for e in ion.cations])
        return float(np.sum(c * (x - c @ x) ** 2))

    comp_lookup = compounds.set_index("material_id").composition
    sub = ionic.sample(8000, random_state=1)
    xv = np.array([chi_var(comp_lookup[m]) for m in sub.material_id])
    res = (sub.mp - sub.fefos_linear).to_numpy()
    out["residual_vs_chi"] = {
        "pearson_r": round(float(np.corrcoef(res, xv)[0, 1]), 3),
        "mean_residual": round(float(res.mean()), 4),
        "n": int(len(sub)),
    }
    # 7. oxide screen (scripts/screen_oxides.py) and the ML comparison (benchmarks/run_related.py)
    if (RES / "screening.csv.gz").exists():
        out["screening"] = screening_summary(pd.read_csv(RES / "screening.csv.gz"))
    for name in ("related_work", "screening_calibration", "screening_crosscheck"):
        if (RES / f"{name}.json").exists():
            out[name] = json.loads((RES / f"{name}.json").read_text())
    (RES / "report_data.json").write_text(json.dumps(out, default=float))
    print(
        json.dumps({k: v for k, v in out.items() if not k.startswith("parity")}, default=float)[
            :3000
        ]
    )


if __name__ == "__main__":
    main()
