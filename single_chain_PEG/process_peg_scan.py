import os
import sys
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import seaborn as sns
import mdtraj as md

# if os.path.isdir("/projects/prism/people/mhz916/software/BLOCKING"):
sys.path.append("/projects/prism/people/mhz916/software/BLOCKING")
from main import BlockAnalysis


def parse_lower_lambda_files(name,lambda_values,sigma_values,rounder=1):
    data = {
        "l_value":np.round(lambda_values,rounder),
        "s_value":np.round(sigma_values,rounder),
    }
    rgArrays = []
    simRgs  = []
    simRgEs = []
    for l,s in zip(lambda_values,sigma_values):
        if str(l)[-2:] == ".0":
            file = F'{name}/rg_{int(l)}_{int(s)}.npy'
        else:
            file = F'./lower_lambdas/{name}/rg_{np.round(l,rounder)}_{np.round(s,rounder)}.npy'
        print(file)
        if os.path.isfile(file):
            print("Si")
            rg_array = np.load(file)
            rgArrays.append(rg_array)
            simRgs.append(rg_array.mean())
            simRgEs.append(np.std(rg_array)/np.sqrt(rg_array.size))
    print('\n')
    data[(np.round(l,rounder),np.round(s,rounder))] = {}
    data[(np.round(l,rounder),np.round(s,rounder))]["rgArray"]=rgArrays
    data[(np.round(l,rounder),np.round(s,rounder))]["simRg"]=simRgs
    data[(np.round(l,rounder),np.round(s,rounder))]["simRgE"]=simRgEs
    return data


def calcRg(t,residues,fasta):
    masses = residues.loc[fasta,'MW'].values
    # calculate the center of mass
    cm = np.sum(t.xyz*masses[np.newaxis,:,np.newaxis],axis=1)/masses.sum()
    # calculate residue-cm distances
    si = np.linalg.norm(t.xyz - cm[:,np.newaxis,:],axis=2)
    # calculate rg
    rgarray = np.sqrt(np.sum(si**2*masses,axis=1)/masses.sum())
    return rgarray


def extract_Rg(traj_file, top_file, directory, residues, fasta):
    #
    traj = md.load(traj_file, top=top_file)#[10:]
    traj = traj.image_molecules(inplace=False, anchor_molecules=[set(traj.top.chain(0).atoms)], make_whole=True)
    traj.center_coordinates()
    traj.xyz += traj.unitcell_lengths[0,0]/2
    # Cacluate Rg
    rgarray = calcRg(traj,residues,fasta)
    np.save(directory+F"/rg_{residues.loc['J','lambdas']:g}_{residues.loc['J','lambdas']:g}.npy",rgarray)
    return rgarray


def calculate_relative_error(simulation, experiment):
    return np.abs((simulation - experiment) / experiment * 100)


def calculate_rmse(simulation, experiment):
    return np.sqrt(((simulation - experiment)**2).sum())


def calculate_chi_squared(simulation,experiment, expError=None, error_perc=0.1):
    if not expError:
        expError = experiment*error_perc
    return np.sum((simulation-experiment)**2 / expError**2)


def calculate_reduced_chi_squared(simulation,experiment,n_par=0, expError=None, error_perc=0.1,df=0):
    if expError != None:
        expError = experiment*error_perc
    if df == 0:
        df = len(simulation)-n_par
    return calculate_chi_squared(simulation,experiment, expError=expError) / df


if __name__=="__main__":
    check_output = True
    h_par = "8030/150"
    df_peg_simulated = pd.read_csv("./experimental-data__peg-compaction__used-subset.csv",index_col=0)

    pegs = ['PEG400','PEG1000', 'PEG1560', 'PEG2000', 'PEG3000', 'PEG4000', 'PEG50001',
            'PEG8000', 'PEG10000', 'PEG16000', 'PEG20000', 'PEG21000', 'PEG25000',
            'PEG30000']
    lambdas = [0.0,0.1,0.2,0.3,0.4]
    sigmas  = [0.35,0.4,0.43,0.46]

    cols = ['simRg','simRgSEM','expRg','simRgE','rgArray']

    indices = []
    for name in pegs:
        for l in lambdas:
            for s in sigmas:
                indices.append((name, l, s))
    rg_values_8030 = pd.DataFrame(columns=cols,index=pd.MultiIndex.from_tuples(indices, names=['PEG','lambda','sigma']))

    residues = pd.read_csv(F"./{h_par}/residues.csv").set_index('one',drop=False)
    components = pd.read_pickle(F"./{h_par}/proteins.pkl")
    block_analysis = {}

    for name in pegs:
        path = F"./{h_par}/{name}"
        for l in lambdas:
            for s in sigmas:
                # Set the parameters
                residues.loc['J','lambdas'] = l
                residues.loc['J','sigmas'] = s
                
                if check_output:
                    print(F"{h_par}/{name}/t_{l}_{s}.dcd: {os.path.isfile(F'{h_par}/{name}/t_{l}_{s}.dcd')}")
                try:
                    rg_array = extract_Rg(traj_file=F"{path}/t_{l}_{s}.dcd",
                                          top_file=F"{path}/top.pdb",
                                          directory=path,
                                          residues=residues,
                                          fasta=components.loc[name,'fasta'])
                    if check_output:
                        print(F'Number of frames used: {rg_array.size:d}')
                        print(rg_array)
                    block = BlockAnalysis(rg_array)
                    block.SEM()
                    rg_values_8030.loc[(name,l,s),'simRg'] = rg_array.mean()
                    rg_values_8030.loc[(name,l,s),'simRgSEM'] = block.sem
                    rg_values_8030.loc[(name,l,s),'expRg'] = df_peg_simulated.loc[name,'Rg']
                    rg_values_8030.loc[(name,l,s),'simRgE'] = np.std(rg_array)/np.sqrt(rg_array.size)
                    rg_values_8030.loc[(name,l,s),'rgArray'] = rg_array
                    block_analysis[(name,l,s)] = (block)
                except OSError:
                    print(F"Can't find: {h_par}/{name}/t_{l}_{s}.dcd:")
                    rg_values_8030.drop(index=[(name,l,s)])


    data_8030 = {'chi2':pd.DataFrame(columns=lambdas,index=sigmas),
                 'chi2Red':pd.DataFrame(columns=lambdas,index=sigmas),
                 'aveRelError':pd.DataFrame(columns=lambdas,index=sigmas),
                 'rmse':pd.DataFrame(columns=lambdas,index=sigmas)}
    for l in lambdas:
        for s in sigmas:
            simulation = []
            experiment = []
            for name in names:
                simulation.append(rg_values_8030.loc[(name,l,s), 'simRg'])
                experiment.append(rg_values_8030.loc[(name,l,s), 'expRg'])

            simulation = np.array(simulation)
            experiment = np.array(experiment)
            data_8030['chi2'].loc[s,l]    = calculate_chi_squared(simulation, experiment, expError=None, error_perc=0.1)
            data_8030['chi2Red'].loc[s,l] = calculate_reduced_chi_squared(simulation, experiment, expError=None, error_perc=0.1) 
            data_8030['aveRelError'].loc[s,l]= np.mean(calculate_relative_error(simulation, experiment))
            data_8030['rmse'].loc[s,l]    = calculate_rmse(simulation, experiment)

    rg_values_8030.to_pickle("./single-chain-PEG__rg-data.pkl")
    with open(F"./single-chain-PEG__block-analysis.pkl", 'wb') as pf:
        pickle.dump(block_analysis, pf)
    with open(F"./single-chain-PEG__metrics.pkl", 'wb') as pf:
        pickle.dump(data_8030, pf)
