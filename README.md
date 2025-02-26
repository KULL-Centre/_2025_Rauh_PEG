#  A coarse-grained model for disordered proteins under crowded conditions

This repository contains code and data for:

Rauh A.S., Tesei, G. & Lindorff-Larsen, K. (2025) _A coarse-grained model for disordered proteins under crowded conditions_

## Running CALVADOS simulations with PEG
PEG simulations can be run either with the code provided here or using the [CALVADOS package](https://github.com/KULL-Centre/CALVADOS)

## File structure
- **data**: Pandas DataFrames with processed data from the simulations.
- **single_chain_PEG**: Scripts used in parameter scan for reproduction of single-chain properties of linear PEG chains with different molecular weight.
- **protein_crowding**: Scripts used in for tuning $\lambda_{PEG}$ in a comparison with PEG-induced compaction of IDPs.
- **ps_crowding**: Scripts used to run PEG-crowding slab simulations for phase separation systems.
- **figures.ipynb**: Jupyter Notebook with code to reproduce the figures in the manuscript.


