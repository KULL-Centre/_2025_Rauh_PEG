#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 14:36:49 2023

@author: A.S. Rauh
"""
from analyse_protein_scan import *
import os
import subprocess
from jinja2 import Template

proteins = initProteins()
proteins.to_pickle('proteins.pkl')

submission_SLURM = Template("""#!/bin/bash
#SBATCH --job-name={{name}}_{{dirchains}}_{{lambda_peg}}
#SBATCH --nodes=1
#SBATCH --partition=qgpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=18
#SBATCH --mem=50GB
#SBATCH -t 24:30:00
#SBATCH -o ./out/{{name}}_{{dirchains}}.out
#SBATCH -e ./err/{{name}}_{{dirchains}}.err

source /home/asrauh/.bashrc
conda activate hoomd
module purge
module load cmake/3.9.4 gcc/6.5.0 openmpi/4.0.3 llvm/7.0.0 cuda/9.2.148

echo $SLURM_CPUS_PER_TASK
echo $SLURM_JOB_NODELIST

/home/asrauh/miniconda3/envs/hoomd/bin/python ./simulate_protein_scan.py --dirname {{name}} --proteins {{proteins}} --numbers {{numbers}} --temp {{temp}} --ionic {{ionic}} --cutoff {{cutoff}} --boxlength {{boxlength}} --lambda_peg {{lambda_peg}}""")

submission_PBS = Template("""#!/bin/bash
#PBS -W group_list=ku_10001 -A ku_10001
#PBS -N {{name}}_{{dirchains}}_{{lambda_peg}}
### Only send mail when job is aborted or terminates abnormally
#PBS -M arrien.rauh@bio.ku.dk
#PBS -m n
### Number of nodes
#PBS -l nodes=1:ppn=2:gpus=1
### Memory & Walltime
#PBS -l mem=10gb
#PBS -l walltime=24:30:00
### Output
#PBS -e ./out/{{name}}_{{dirchains}}.err
#PBS -o ./err/{{name}}_{{dirchains}}.out

source /home/people/arrrau/.bashrc
conda activate calvados
module purge
module load cuda/toolkit/10.1/10.1.168 openmpi/gcc/64/4.0.2

cd $PBS_O_WORKDIR

/home/projects/ku_10001/people/arrrau/software/miniconda3/envs/calvados/bin/python ./simulate_protein_scan.py --dirname {{name}} --proteins {{proteins}} --numbers {{numbers}} --temp {{temp}} --ionic {{ionic}} --cutoff {{cutoff}} --boxlength {{boxlength}} --lambda_peg {{lambda_peg}}""")


###############################################################################


# Set the interaction cut-off and box size (cubic).
cutoff = 2.0
boxlength = 16 #nm
submission_system = 'SLURM'


# Set the lambda to scan.
lambda_peg = ...


# Define the contents of the system.
names  = [
    'ACTR PEG400',
    'ACTR PEG8000',
    'IN PEG400',
    'IN PEG8000',
    'ProTaCfull PEG400',
    'ProTaCfull PEG8000'
          ]

# number of PEG chains = 6.022 * box_volume * phi_peg / (mw / 1000 * 10)
# we assume that w/v = w/w and that density is 1.12 kg/l
chains = {
    'ACTR PEG400': ['1 0', '1 174', '1 626', '1 967', '1 1324', '1 2647'],
    'ACTR PEG8000': ['1 0', '1 20', '1 36', '1 48', '1 69'],
    'IN PEG400': ['1 0', '1 256', '1 537', '1 1159', '1 1742', '1 2484'],
    'IN PEG8000': ['1 0', '1 16', '1 28', '1 43', '1 60'],
    'ProTaCfull PEG400': ['1 0', '1 270', '1 521', '1 816', '1 1212', '1 2341'],
    'ProTaCfull PEG8000': ['1 0', '1 7', '1 22', '1 36', '1 51', '1 60']
    }


if not os.path.isdir("out"): os.mkdir('./out')
if not os.path.isdir("err"): os.mkdir('./err')


for name,ionic in zip(names,[110]*len(names)):
    print(name)
    dir_name = name.replace(' ','-')

    if not os.path.isdir(dir_name):
        os.mkdir(dir_name)

    for temp in [295]:
        if not os.path.isdir(F'{dir_name}/{temp:d}'):
            os.mkdir(F'{dir_name}/{temp:d}')

        for N_chains in chains[name]:
            dir_chains = N_chains.split(' ')[-1]

            if not os.path.isdir(F"{dir_name}/{temp:d}/{dir_chains}"):
                os.mkdir(F'{dir_name}/{temp:d}/{dir_chains}')

            if submission_system == 'SLURM':
                with open(F'{dir_name:s}_{temp:d}_{dir_chains}.sh', 'w') as submit:
                    submit.write(submission_SLURM.render(name=dir_name,
                                                   lambda_peg=lambda_peg,
                                                   dirchains=dir_chains,
                                                   proteins=name,
                                                   ionic=ionic,
                                                   numbers='{}'.format(N_chains),
                                                   temp='{:d}'.format(temp),
                                                   cutoff='{:.1f}'.format(cutoff),
                                                   boxlength='{:.1f}'.format(boxlength)))
                subprocess.run(['sbatch','{:s}_{:d}_{:s}.sh'.format(dir_name,temp,dir_chains)])
            elif submission_system == 'PBS':
                with open(F'{dir_name:s}_{temp:d}_{dir_chains}.sh', 'w') as submit:
                    submit.write(submission_PBS.render(name=dir_name,
                                                   lambda_peg=lambda_peg,
                                                   dirchains=dir_chains,
                                                   proteins=name,
                                                   ionic=ionic,
                                                   numbers='{}'.format(N_chains),
                                                   temp='{:d}'.format(temp),
                                                   cutoff='{:.1f}'.format(cutoff),
                                                   boxlength='{:.1f}'.format(boxlength)))
                subprocess.run(['qsub','{:s}_{:d}_{:s}.sh'.format(dir_name,temp,dir_chains)])







