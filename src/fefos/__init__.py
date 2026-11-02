"""FEFOS: Formation Energies From Oxidation States.

Predict formation energies of multi-cation compounds (oxides, sulfides,
nitrides, halides, ...) by merging quadratic oxidation/reduction curves fitted
to the convex hulls of the constituent binary systems.
"""

from .composition import IonicComposition, parse_formula
from .curves import Parabola, ReferenceCurves, fit_reference_curves
from .hull import BinaryHull, lower_convex_hull
from .legacy import PaperFEFOS
from .mixing import MixingResult, mix
from .model import FEFOS, Prediction

__all__ = [
    "FEFOS",
    "BinaryHull",
    "IonicComposition",
    "MixingResult",
    "PaperFEFOS",
    "Parabola",
    "Prediction",
    "ReferenceCurves",
    "fit_reference_curves",
    "lower_convex_hull",
    "mix",
    "parse_formula",
]

__version__ = "2.0.0"
