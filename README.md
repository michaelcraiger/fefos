# Formation energy mixing
Creating binary oxide phase diagrams is an extremely time consuming process both experimentally and computationally through DFT. Exploring all of chemical space is a daunting task but may be necessary if we are to discover novel functional materials.

## Data mining

To analyse formation energies in oxides, we must first get information on unary and binary oxides themselves from OQMD and Materials Project. In `data_gather` we calculate formation energies and collect structures contained for a given set of elements and their oxides.  

## High-level procedure overview

When we mix two elements into an oxide, the two elements can have different oxidation states. 

This model attempts to describe how this stabilises binary oxides.

If creating oxide from elements A–B in stoichiometry (A+B):O=2, for instance, we need to make quadratic equations which fit through the most stable points on the A–O and B–O phase diagrams, where the (0, 0) value is AO2 and BO2, and we need quadratic equations for reduction and oxidation. This general representation can be applied to any element, and the procedure for creating the quadratics, which can be element specific, is found in the `make_quadratics/oqmd_analysis.ipynb` notebook.

We then enforce that the average oxidation state of non-oxygen elements in ABO4 be +4, and take a weighted sum of the oxidation and reduction equations for any combination of elements in formation_energy_mixer.ipynb
