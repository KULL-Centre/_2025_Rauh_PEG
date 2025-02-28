from analyse_mixture import *
import os
import subprocess
from jinja2 import Template

proteins = initProteins()
proteins.to_pickle('proteins.pkl')

submission = Template("""#!/bin/bash
#PBS -W group_list=ku_10001 -A ku_10001
#PBS -N {{name}}_{{temp}}_{{dirchains}}
### Only send mail when job is aborted or terminates abnormally
#PBS -M arrien.rauh@bio.ku.dk
#PBS -m n
### Number of nodes
#PBS -l nodes=1:gpu:ppn=15:gpus=1
### Memory & Walltime
#PBS -l mem=50gb
#PBS -l walltime=72:00:00
### Output
#PBS -e {{name}}_{{temp}}_{{dirchains}}.err
#PBS -o {{name}}_{{temp}}_{{dirchains}}.out

source /home/people/arrrau/.bashrc
/home/projects/ku_10001/people/arrrau/software/miniconda3/condabin/conda activate calvados
module purge
module load cuda/toolkit/10.1/10.1.168 openmpi/gcc/64/4.0.2

cd $PBS_O_WORKDIR

/home/projects/ku_10001/people/arrrau/software/miniconda3/envs/calvados/bin/python ./simulate_mixture_slab.py --dirname {{name}} --proteins {{proteins}} --numbers {{numbers}} --temp {{temp}} --ionic {{ionic}} --cutoff {{cutoff}} --boxlength {{boxlength}}""")

# Set the interaction cut-off.
cutoff = 2.0
boxlength = 15 #nm

# Define the contents of the system.

names  = ["A1 PEG400",
          "A1 PEG8000",
          "AroM PEG8000",
          "AroMM PEG8000",
          "aSyn PEG8000",
          "aSyn2745 PEG8000"
]

temperature  = {
          'A1 PEG400':298,
          'A1 PEG8000':298,
          "AroM PEG8000":298,
          "AroMM PEG8000":298,
          'aSyn PEG8000':298,
          'aSyn2745 PEG8000':298
}

# number of PEG chains = 6.022 * boxlength**3 * 0.02 / (8000 / 1000 * 10)
# we assume that w/v = w/w and that density is 1 kg/l
#["100 508", "100 381","100 318","100 254","100 191","100 127"]
chains= {
    "A1 PEG400":["100 0","100 1287","100 2554","100 3841","100 5108"],
    "A1 PEG8000":["100 0","100 64","100 127","100 191","100 254"],
    "AroM PEG8000":["100 0","100 64","100 127","100 191","100 254"],
    "AroMM PEG8000":["100 0","100 64","100 127","100 191","100 254"],
    "aSyn PEG8000":["100 0","100 64","100 127","100 191","100 254","100 318","100 381","100 508"],
    "aSyn2745 PEG8000":["100 0","100 64","100 127","100 191","100 254"]
} 

#
for name,ionic in zip(names,[150]*len(names)):
    print(name)
    dir_name = name.replace(' ','-')
    if not os.path.isdir(dir_name):
        os.mkdir(dir_name)
    for temp in [298]:
        if not os.path.isdir(F'{dir_name}/{temp:d}'):
            os.mkdir(F'{dir_name}/{temp:d}')
        for N_chains in chains[name]:
            dir_chains = N_chains.split(' ')[-1] # N_chains.replace(' ','-')
            if not os.path.isdir(F"{dir_name}/{temp:d}/{dir_chains}"):
                os.mkdir(F'{dir_name}/{temp:d}/{dir_chains}')
            with open(F'{dir_name:s}_{temp:d}_{dir_chains}.sh', 'w') as submit:
                submit.write(submission.render(name=dir_name,
                                               dirchains=dir_chains,
                                               proteins=name,
                                               ionic=ionic,
                                               numbers='{}'.format(N_chains),
                                               temp='{:d}'.format(temp),
                                               cutoff='{:.1f}'.format(cutoff),
                                               boxlength='{:.1f}'.format(boxlength)))
            print(F'Submitting: {dir_name:s}_{temp:d}_{dir_chains}.sh')
            subprocess.run(['qsub','{:s}_{:d}_{:s}.sh'.format(dir_name,temp,dir_chains)])
