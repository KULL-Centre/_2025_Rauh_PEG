from analyse_mixture import *
import os
import subprocess
from jinja2 import Template

proteins = initProteins()
proteins.to_pickle('proteins.pkl')

submission = Template("""#!/bin/bash
#PBS -W group_list=ku_10001 -A ku_10001
#PBS -N sc_peg_{{name}}
### Only send mail when job is aborted or terminates abnormally
#PBS -M arrien.rauh@bio.ku.dk
#PBS -m n
### Number of nodes
#PBS -l nodes=1:ppn=2:gpus=1
### Memory & Walltime
#PBS -l mem=10gb
#PBS -l walltime=72:00:00
### Output
#PBS -e {{name}}.err
#PBS -o {{name}}.out

source /home/people/arrrau/.bashrc
conda activate calvados
module purge
module load cuda/toolkit/10.1/10.1.168 openmpi/gcc/64/4.0.2

cd $PBS_O_WORKDIR

for l in 0.0 0.1 0.2 0.3 0.4 0.5
do
    /home/projects/ku_10001/people/arrrau/software/miniconda3/envs/calvados/bin/python ./simulate_peg_scan.py --dirname {{name}} --peg {{name}} --numbers 1 --temp {{temp}} --ionic {{ionic}} --cutoff {{cutoff}} --boxlength {{boxlength}} --lambda_peg $l
done

""")

# Set the interaction cut-off.
cutoff = 2.0
boxlength = 20 #nm

# Define the contents of the system.

names  = ['PEG400']
#, 'PEG1000', 'PEG1560', 'PEG2000', 'PEG3000', 'PEG4000','PEG50001',
#          'PEG50002', 'PEG8000', 'PEG10000', 'PEG16000', 'PEG20000','PEG21000', 'PEG25000',
#          'PEG30000', 'PEG35000', 'PEG40000', 'PEG73000','PEG150000']

#
for name in names:
    print(name)
    if not os.path.isdir(name):
        os.mkdir(name)
    temp  = proteins.loc[name,'temp']
    ionic = proteins.loc[name,'ionic']*1000
    with open(F'submit_scan_{name:s}.sh', 'w') as submit:
        submit.write(submission.render(name=name,
                                       ionic=ionic,
                                       numbers='1',
                                       temp=temp,
                                       cutoff=F'{cutoff:.1f}',
                                       boxlength=F'{boxlength:.1f}'))
    subprocess.run(['qsub',F'submit_scan_{name:s}.sh'])

