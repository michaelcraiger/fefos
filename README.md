# Formation energy mixing
<p align="center">
  <img src="https://github.com/michaelcraiger/oxide_formation_energy/blob/main/mixing_pic.jpg" alt="Sublime's custom image"/>
</p>

<p style='text-align: justify;'> 

Creating binary oxide phase diagrams is an extremely time consuming process both experimentally and computationally through DFT. Exploring all of chemical space is a daunting task but may be necessary if we are to discover novel functional materials.

## Overview

When we mix two elements into an oxide, the two elements can have different oxidation states. 

This model attempts to describe how this stabilises binary oxides.

If creating an oxide from elements A–B in stoichiometry (A+B):O=2, for instance, we need to make quadratic equations which fit through the most stable points on the A–O and B–O phase diagrams, where the (0, 0) value is AO2 and BO2, and we need quadratic equations for reduction and oxidation. This general representation can be applied to any element, and the procedure for creating the quadratics, which can be element specific, is found in the notebooks in the `make_quadratics` directory.

We then enforce that the average oxidation state of non-oxygen elements in ABO4 be +4, and take a weighted sum of the oxidation and reduction equations for any combination of elements in formation_energy_mixer.ipynb

## Installing

To use the code, use python version 3.8.12, other versions can have difficulty using the necessary version of pymatgen. Using conda; 

```
conda create -n oxide_mixer python=3.8.12
```
Then activate the environment

```
conda activate oxide_mixer
```

```
git clone https://github.com/michaelcraiger/oxide_formation_energy
``` 

Then in the environment:

```
conda install matplotlib==3.2.1 numpy==1.22.3 scipy==1.7.3 pymatgen==2022.4.19 notebook ipykernel
```
Finally to be able to use the environment in ipykernel
```
python -m ipykernel install --user --name=oxide_mixer
```

## File system description

The directory `data_gather` contains the code to collect the necessary data from Materials Project in `mp_data_save.ipynb`. This is saved as pickle files since there contains pymatgen structure objects and they are under `unary_oxide_data.p` for unary oxide information, `ele2gs.p` which is the single element energy data, required to calculate formation energies and the binary oxide data under `binary_oxide_data.p`. These files are needed to fit the coefficients for the parabolas of each of the 3 considered binary oxide types i.e. MO, M2O3 and MO2-type oxides under the `make_quadratics` directory. Within `data_gather` another file which saves all the MO/M2O3/MO2-type binary oxides with the appropriate pairs of elements, and formation energy infomation required by FEFOS, is seen in `binary_pairing_data.ipynb` and saved as `binary_pairing_data.p` which is required to plot Figures 2 and 4.

In `make_quadratics` there is code to save the oxidation and reduction parabola coefficients for each considered element, oxidation state pairing under `mp_quadratic_equations_ox.p` and `mp_quadratic_equations_red.p` denoting oxidation and reduction respectively.

`roost_test_data` contains the csv files required to plot the learning curves for Figure 3 in the manuscript, found in the Goodall, Lee Nature Communications paper under 'Data availability'.

The code to plot Figure 1, 2 and 4 is in the notebook `formation_energy_mixer.ipynb` as this includes the code which carries out the FEFOS procedure once the parabola coefficients and binary oxide data from Materials Project are saved.

</p>
