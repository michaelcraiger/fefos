# FEFOS: Formation Energies From Oxidation States

<p align="center"><img src="docs/paper_2023_mixing.jpg" width="320" alt="Mixing two oxides"></p>

FEFOS predicts the formation energy of a mixed compound from nothing but the
binary compounds of its elements. Each element's formation energy is described
as a function of its oxidation state, and the compound's energy is the
charge-neutral combination of those curves that minimises the total.

The method was introduced for binary-cation oxides in
[Craig, Kleuker, Bajdich & García-Melchor, *Catal. Sci. Technol.* 2023](https://doi.org/10.1039/D3CY00107E).
**Version 2** is a complete rewrite as a tested Python package and extends the
method from two cations sharing O²⁻ to:

* **any number of cations** (ternary, quaternary, … high-entropy compounds);
* **any anion**: O, S, Se, Te, N, P, C, H, F, Cl, Br, I;
* **any stoichiometry**: e.g. spinels (mean oxidation state 8/3), not just MO, M₂O₃, MO₂;
* **curves fitted automatically** to each element's convex hull, so no hand-tuned coefficients remain.

The code and data exactly as published in 2023 are kept under the git tag
[`paper-2023`](https://github.com/michaelcraiger/fefos/tree/paper-2023), and
`fefos reproduce-paper` recomputes all 2,327 published predictions from the new
code (max. deviation 1×10⁻⁴ eV/atom, the rounding of the published table).

## How it works

1. **One hull per element and anion.** A binary compound M_aX_b sits at the formal
   oxidation state *q* = *v*·*b*/*a* (with *v* the anion valence). Its formation
   energy per cation, *e*(*q*), combines linearly, so the lower convex hull of
   (*q*, *e*) is the thermodynamic hull of the M–X system. Beyond the most
   anion-rich stable compound the hull is flat (excess anion leaves as X₂).
2. **A common reference.** For a compound with cation fractions *cᵢ* and mean
   oxidation state *q̄*, every cation starts at *q̄*. Its reference energy is
   read off its hull, so the 2023 hand-picked or "synthetic" reference oxides
   are no longer needed. This gives the reference average Σ*cᵢ*·Δ*E*(MᵢX_{q̄/v}).
3. **Charge-neutral redistribution (Eq. 2 of the paper, now n cations).** Cations
   shift their oxidation states by *dᵢ*, subject to Σ*cᵢdᵢ* = 0. Each shift is
   priced by that element's oxidation or reduction curve. The minimum is the
   mixing energy ΔD, and Δ*E*_f = reference average + ΔD (Eq. 6).

Three curve and merging modes are provided:

| mode | curves | merging | notes |
|---|---|---|---|
| `linear` (default) | the hull itself (piecewise linear) | concentration-weighted, exact | parameter-free; equals the energy of the most stable mixture of binary compounds (verified against pymatgen to 10⁻¹⁴ eV/atom) |
| `quadratic` | Eq. 1 parabolas *a*₁*x*² + *a*₂*x* least-squares fitted to the hull (per cation, ±2 OS) | concentration-weighted | smooth, two coefficients per side |
| `paper` | Eq. 1 parabolas in the 2023 convention (per atom, symmetric through the minimum) | the 2023 rule: unweighted sum + end-member correction (Eq. 5), generalised to n cations | reproduces e.g. *f*_red,CrO₂ = 0.319*x*² − 0.638*x* of Fig. 1 automatically |

The merging is solved exactly: by bisection on the Lagrange multiplier (the anion
chemical potential) for parabolas, and by a greedy fill of hull segments in
order of slope (a fractional knapsack problem) for the linear mode.

## Results (Materials Project, 2026-09-28 release)

Benchmark: every multi-cation, single-anion compound in MP (lowest-energy
polymorph per composition), 107,600 of 108,808 representable. MAE in eV/atom.

**On the 2023 model's own domain** (two-cation oxides with mean OS 2/3/4, the
2,735 compounds the published coefficients cover):

| | all | ICSD | near hull (<0.1 eV/atom) |
|---|---|---|---|
| FEFOS 2023 (published coefficients) | 0.161 | 0.160 | 0.132 |
| 2023 rule, refitted automatically to the hulls (`paper`) | 0.160 | 0.151 | 0.124 |
| FEFOS v2 `quadratic` | 0.148 | 0.134 | 0.101 |
| FEFOS v2 `linear` | **0.140** | **0.117** | **0.091** |
| reference average (no charge transfer) | 0.212 | 0.255 | 0.192 |

**The extension** (56,428 ionic compounds with metal cations, all twelve anions, 2–6 cations):

| | all | stable | ICSD | near hull |
|---|---|---|---|---|
| `linear` | **0.121** | **0.112** | **0.120** | **0.096** |
| `quadratic` | 0.137 | 0.133 | 0.153 | 0.119 |
| `paper` | 0.160 | 0.167 | 0.167 | 0.146 |
| reference average | 0.234 | 0.242 | 0.276 | 0.234 |

The error does not grow with the number of cations (`linear`: 0.129, 0.115,
0.108, 0.108 eV/atom for 2, 3, 4 and 5+ cations) and is 0.07–0.19 eV/atom across
the twelve anions. Two classes are harder:

* **polyanions and mixed anions**, where a nonmetal acts as a cation (phosphates,
  sulfates, carbonates, oxyhalides): `linear` 0.231 vs reference 0.448;
* **intermetallic-like compounds** with mean OS < 1: 0.55. These are outside
  the oxidation-state picture.

Full tables: `results/summary.json`; per-compound predictions:
`results/benchmark_predictions.csv.gz`.

## Install

```bash
git clone https://github.com/michaelcraiger/fefos && cd fefos
pip install -e ".[dev]"      # Python >= 3.10; core needs only numpy and pandas
```

## Use

```console
$ fefos predict SrTiO3 LiFePO4 KMgF3 Li2ZrN2
formula          anion mean OS      dEf      ref       dD  oxidation states
SrTiO3               O    3.00   -3.335   -2.928   -0.407  Sr +2.00, Ti +4.00
LiFePO4              O    2.67   -2.090   -1.676   -0.414  Fe +2.00, Li +1.00, P +5.00
KMgF3                F    1.50   -3.460   -2.938   -0.522  K +1.00, Mg +2.00
Li2ZrN2              N    2.00   -1.122   -0.812   -0.310  Li +1.00, Zr +4.00
$ fefos curves Cr O --q 4 --mode paper     # hull + Eq. 1 parabolas
$ fefos reproduce-paper                    # check against the 2023 predictions
```

```python
from fefos.cli import load_model

model = load_model(mode="linear")          # hulls from data/mp_2026-09-28/binaries.csv
p = model.predict("Ba2YCu3O7")
p.formation_energy, p.reference_energy, p.delta_d, p.oxidation_states

# or from your own binary data (any source: MP, OQMD, your DFT)
from fefos import FEFOS
model = FEFOS.from_binary_entries(
    [{"composition": "TiO2", "formation_energy_per_atom": -3.51}, ...], mode="quadratic"
)
```

## Reproduce the benchmark

```bash
python scripts/fetch_mp.py          # MP summary from AWS Open Data (no API key) -> data/mp_<version>/
python scripts/run_benchmark.py     # -> results/benchmark_predictions.csv.gz, results/summary.json
python scripts/make_report_data.py  # -> results/report_data.json
pytest
```

The `quadratic`/`paper` fitting window (±2 oxidation states) was chosen on the
2023 domain only (two-cation oxides) and then applied unchanged to every other
anion and cation count.

## Layout

```
src/fefos/
  composition.py  formula parsing; cation/anion view, mean oxidation state
  hull.py         1-D convex hull of an M–X system on the oxidation-state axis
  curves.py       Eq. 1 parabolas fitted to a hull around any reference state
  mixing.py       charge-neutral merging of curves for n cations (Eqs. 2–5)
  model.py        FEFOS model (linear / quadratic / paper modes)
  legacy.py       exact 2023 model with the published coefficients
  data.py         Materials Project open-data download and dataset building
  benchmark.py    benchmark runner and metrics
data/
  paper_2023/     published coefficients, references, predictions, benchmark set
  mp_2026-09-28/  binaries.csv (hull inputs), compounds.csv.gz (benchmark targets)
results/          benchmark outputs
```

## Citation

```bibtex
@article{craig2023fefos,
  title   = {FEFOS: a method to derive oxide formation energies from oxidation states},
  author  = {Craig, Michael John and Kleuker, Felix and Bajdich, Michal and Garc{\'i}a-Melchor, Max},
  journal = {Catalysis Science \& Technology},
  volume  = {13},
  pages   = {3427},
  year    = {2023},
  doi     = {10.1039/D3CY00107E}
}
```

Materials Project data: A. Jain *et al.*, *APL Materials* **1**, 011002 (2013),
used under CC BY 4.0.
