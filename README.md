#  A coarse-grained model for disordered proteins under crowded conditions

This repository contains code and data for:

Rauh A.S., Tesei, G. & Lindorff-Larsen, K. (2025) _A coarse-grained model for disordered proteins under crowded conditions._ [bioRxiv](https://doi.org/10.1101/2025.03.01.640997)

The simulation data can be retrieved from the Electronic Research Data Archive at UCPH (ERDA): [here](https://sid.erda.dk/cgi-sid/ls.py?share_id=B2pqRhqcsE)

## Running CALVADOS simulations with PEG
PEG simulations can be run either with the code provided here or using the [CALVADOS package](https://github.com/KULL-Centre/CALVADOS) ([Example](https://github.com/KULL-Centre/CALVADOS/tree/main/examples/slab_IDR_PEG))

## File structure
- **data**: Pandas DataFrames with processed data from the simulations.
- **single_chain_PEG**: Scripts used in parameter scan for reproduction of single-chain properties of linear PEG chains with different molecular weight.
- **protein_crowding**: Scripts used in for tuning $\lambda_{PEG}$ in a comparison with PEG-induced compaction of IDPs.
- **ps_crowding**: Scripts used to run PEG-crowding slab simulations for phase separation systems.
- **figures.ipynb**: Jupyter Notebook with code to reproduce the figures in the manuscript.
