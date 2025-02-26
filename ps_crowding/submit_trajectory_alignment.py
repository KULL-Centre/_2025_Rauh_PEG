#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 13:04:10 2023

@author: Arriën Symon Rauh
"""
from analyse_mixture import *
import os
import numpy as np
import pandas as pd
from scipy.constants import Avogadro
import subprocess
from jinja2 import Template


proteins = pd.read_pickle('proteins.pkl')


def get_number_PEG_chains_wv(wv_fraction, mw_peg, volume_simulation_box, NAv=Avogadro):
    molarity_peg = (wv_fraction*10)/mw_peg
    return int(np.round(molarity_peg*volume_simulation_box*NAv))


def get_number_PEG_chains_vv(vol_fraction, mw, v, density=1.12, NA=Avogadro):
    return int(np.round(NA*((density*(max(vol_fraction/100*v,0)))/mw))) # /mol * ((g/cm^3*cm^3)*))



submission_alignment = Template("""#!/bin/bash
#PBS -W group_list=ku_10001 -A ku_10001
#PBS -N {{name}}_align
### Only send mail when job is aborted or terminates abnormally
#PBS -M arrien.rauh@bio.ku.dk
#PBS -m n
### Number of nodes
##PBS -l nodes=1:thinnode:ppn=20:
### Memory & Walltime
#PBS -l mem=100gb
#PBS -l walltime=02:00:00
### Output
#PBS -e {{name}}_{{temp}}_align.err
#PBS -o {{name}}_{{temp}}_align.out

source /home/people/arrrau/.bashrc
/home/projects/ku_10001/people/arrrau/software/miniconda3/condabin/conda activate calvados
module purge
module load cuda/toolkit/10.1/10.1.168 openmpi/gcc/64/4.0.2

cd $PBS_O_WORKDIR

/home/projects/ku_10001/people/arrrau/software/miniconda3/envs/calvados/bin/python align_slabs_peg.py --protein {{protein}} --peg {{peg}} --number {{n_peg}} --temp {{temp}}
""")


n_protein = 100

systems = {'A1':{
               298:{
                  # 'PEG400':[0.0,2.53293,5.02651,7.55944,10.053],
                  # 'PEG2000':[0.0,2.5,5.0],
                  # 'PEG8000':[0.0,2.5,5.0,7.5,10.0],#12.5,15.0,20.0],
                   'PEG8000':[12.5,15.0,20.0],
                  # 'PEG12000':[0.0,2.5,5.0],
                  # 'PEG20000':[0.0,2.5,5.0,7.5,10.0]}
              }},
          # 'AroMM':{
          #     298:{
          #         'PEG8000':[0.0,2.5,5.0,7.5,10.0]}},
          # 'AroM':{
          #     298:{
          #         'PEG8000':[0.0,2.5,5.0,7.5,10.0]}},
          # 'aSyn':{
          #     298:{
          #         'PEG8000':[0.0,2.5,5.0,7.5,10.0,12.5,15.0,20.0]}},
           # 'Ddx4':{
           #     293:{}}
           }


box_sizes = {'A1':(15,15,150),
             'AroMM':(15,15,150),
             'AroM':(15,15,150),
             'aSyn':(15,15,150),
             'Ddx4':(17,17,300)
    }


# Protein
for protein in systems.keys():
    for temp in systems[protein].keys():
        for peg in systems[protein][temp].keys():
            # How much PEG?
            for peg_perc in systems[protein][temp][peg]:
                box_size = box_sizes[protein] # nm
                box_vol = (box_size[0]*box_size[1]*box_size[2])*1e-24 # L
                n_peg = get_number_PEG_chains_wv(wv_fraction=peg_perc,
                                                 mw_peg=int(peg[3:]),
                                                 volume_simulation_box=box_vol,
                                                 NAv=6.022e23)
                name = F"{protein}-{peg}_{temp}_{n_peg}"

                with open(F'{name:s}_{temp:d}_align.sh','w') as submit:
                    submit.write(submission_alignment.render(name=name,
                                                             temp=F'{temp:d}',
                                                             protein=F'{protein:s}',
                                                             peg=F'{peg:s}',
                                                             n_peg=F'{n_peg:d}'))
                print("Submitting: ",name)
                subprocess.run(['qsub',F'{name:s}_{temp:d}_align.sh'])


