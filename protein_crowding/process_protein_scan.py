import os
import sys
import pickle
from scipy import stats
import pandas as pd
import numpy as np
import pickle
import warnings
from scipy.stats import pearsonr
from scipy.optimize import curve_fit
warnings.filterwarnings('ignore') # make the notebook nicer
import mdtraj as md

sys.path.append("/home/mhz916/software/BLOCKING")
from main import BlockAnalysis


def to_pickle(filename, object):
    with open(filename, 'ab') as pf:
        # source, destination
        pickle.dump(object, pf)


def from_pickle(filename):
    with open(filename, 'rb') as pf:
        # for reading also binary mode is important
        unpickled_object = pickle.load(pf)
    return unpickled_object


def calcRg(t,residues,fasta):
    masses = residues.loc[fasta,'MW'].values
    # calculate the center of mass
    cm = np.sum(t.xyz*masses[np.newaxis,:,np.newaxis],axis=1)/masses.sum()
    # calculate residue-cm distances
    si = np.linalg.norm(t.xyz - cm[:,np.newaxis,:],axis=2)
    # calculate rg
    rg_array = np.sqrt(np.sum(si**2*masses,axis=1)/masses.sum())
    return rg_array


def calcRee(t,pos1=0,pos2=-1):
    # calculate rg
    ree_array = np.linalg.norm(t.xyz[:,pos1,:]-t.xyz[:,pos2,:], axis=1)
    return ree_array


def calculate_relative_crowding(Rgs):
    if type(Rgs) != np.array: Rgs=np.array(Rgs)
    return (1-(Rgs/Rgs[0]))*100


def calc_chi_squared(sim_data, exp_data, exp_err=None, reduced=False):
    if exp_err==None: exp_err = 0.1*exp_data
    if reduced:
        
        return  (np.sum((sim_data - exp_data)**2 / (exp_err)**2))/ (len(sim_data)-n_par)
    else:
        return  (np.sum((sim_data - exp_data)**2 / (exp_err)**2))


def calculate_relative_error(experiment, simulation):
    return np.abs((simulation - experiment) / experiment * 100)


def load_protein_crowding_data(systems,lambdas,percentages,percentage_to_numbers,data,temp=295,start=0,stop=-1,step=1):
    blocking_analysis = {}
    
    for protein, peg in systems:
        # Load the data on the molecule of interest.
        residues = pd.read_csv(F"./residues.csv").set_index('one',drop=False)
        proteins = pd.read_pickle(F"./proteins.pkl").loc[protein]
        fasta = proteins.fasta
        
        for l in lambdas:
            residues.loc['J','lambdas'] = l
            
            for perc in percentages[(protein,peg)]:
                data_dir = F"{lambda_to_str[l]}/{protein}-{peg}/{temp}/{percentage_to_numbers[(protein,peg)][perc]}"
                # print(F"./{path}/{percentage_to_numbers[(protein,peg)][perc]}/top.pdb")
                traj = md.load(F"./{data_dir}/t.dcd",
                               top=F"./{data_dir}/top.pdb")[start:stop:step]

                # Image the selected chain back into the original box.
                traj = traj.image_molecules(inplace=False,
                                            anchor_molecules=[set(traj.top.chain(0).atoms)],
                                            make_whole=True)
                traj.center_coordinates()
                traj.xyz += traj.unitcell_lengths[0,0]/2
                print('Number of frames: {:d}'.format(traj.n_frames))
                print(traj.atom_slice(traj.top.select("chainid 0")))
                rgarray_temp = calcRg(traj.atom_slice(traj.top.select("chainid 0")),residues,fasta)

                # Save array of raw Rg data to directory.
                np.save(F"./{data_dir}/rg_data_{protein}-{peg}_{temp}_{percentage_to_numbers[(protein,peg)][perc]}.npy",
                        rgarray_temp)
                block_rg = BlockAnalysis(rgarray_temp)
                block_rg.SEM()

                # Ree calculations            
                reearray_temp = calcRee(traj.atom_slice(traj.top.select("chainid 0")),pos1=dye_pos[protein][0],pos2=dye_pos[protein][1])
                # Save array of raw Ree data to directory.
                np.save(F"./{data_dir}/ree_data_{protein}-{peg}_{temp}_{percentage_to_numbers[(protein,peg)][perc]}.npy",
                        reearray_temp)
                block_ree = BlockAnalysis(reearray_temp)
                block_ree.SEM()
                
                print(type(block_rg.av))

                data.loc[(protein, peg, l, perc),'simRg']     = block_rg.av
                data.loc[(protein, peg, l, perc),'simRgSEM']  = block_rg.sem
                data.loc[(protein, peg, l, perc),'simRee']    = block_ree.av
                data.loc[(protein, peg, l, perc),'simReeSEM'] = block_ree.sem
                data.loc[(protein, peg, l, perc),'N_chains']  = percentage_to_numbers[(protein,peg)][perc]
                data.loc[(protein, peg, l, perc),'simRgA']    = rgarray_temp
                data.loc[(protein, peg, l, perc),'simReeA']   = reearray_temp
                
                blocking_analysis[(protein, peg, l, perc)]    = {'rg':block_rg,
                                                                 'ree':block_ree}
    return data, blocking_analysis


if __name__=='__main__':
    ##############################
    ###  Process trajectories  ###
    ##############################
    lambda_to_str = {0.0:'000',0.1:'010',0.2:'020',0.3:'030'}

    dye_pos = {'ACTR':(0,-1),
            'IN':(0,-1)}

    temp = 295
    lambdas = [0.0,0.1,0.2,0.3]
    systems = [('ACTR', 'PEG400'),
            ('ACTR', 'PEG8000'),
            ('IN', 'PEG400'),
            ('IN', 'PEG8000')]

    cols = ['simRg','simRgSEM','simRee','simReeSEM','N_chains','simRgA','simReeA']

    indices = []
    for system in systems:
        for l in lambdas:
            for perc in (vol_fractions_split[system]):
                indices.append((system[0], system[1], l, perc))

    data = pd.DataFrame(columns=cols,index=pd.MultiIndex.from_tuples(indices, names=['protein','peg','lambda','percentage']))

    data, blocking_analysis = load_protein_crowding_data(systems=systems,
                                                        lambdas=lambdas,
                                                        percentages=vol_fractions_split,
                                                        percentage_to_numbers=percentage_to_numbers,
                                                        data=data,
                                                        temp=temp,
                                                        start=200,stop=-1,step=1)


    ###########################
    ###  Process scan data  ###
    ###########################


    ACTR = {}
    IN = {}

    ACTR_exp = {
        'PEG400':{
            0.:2.481434565226884,
            2.513025017902774:2.4364521906407877,
            9.070974615307184:2.4426807913458326,
            14.007858020094648:2.409571935712151,
            19.165795905693635:2.356623239221165,
            27.860605484274497:2.2675814940933483,
            38.3238509093467:2.1522333103815288},
    # PEG8000
        'PEG8000':{
            0.:2.481434565226884,
            5.6772127298160475:2.397956993548617,
            10.347101295279533:2.33020381651242,
            13.814139775699514:2.320014182628568,
            19.899146088273163:2.2786536417444667}	
    }

    IN_exp = {
        'PEG400':{
            0.:2.017507901491959,
            3.707575138455711:2.0148292506243295,
            7.7762373054322875:2.0070836315539986,
            16.785859379067215:1.98901676655914,
            25.221738483332455:1.9660303100529586,
            35.969183208222816:1.953439092759093},
    # PEG8000
        'PEG8000':{
            0.:2.017673196337158,
            4.524801758448074:2.002519926148217,
            8.11644979887675:2.0006777386400754,
            12.389195930006638:1.9621176344048348,
            17.495086963814828:1.9536524534361166}
    }

    vol_fractions_systems = {
        'ACTR PEG400':[0.0, 2.513025017902774, 9.070974615307184, 14.007858020094648, 19.165795905693635, 38.3238509093467],
        'ACTR PEG8000':[0.0, 5.6772127298160475, 10.347101295279533, 13.814139775699514, 19.899146088273163],
        'IN PEG400':[0.0, 3.707575138455711, 7.7762373054322875, 16.785859379067215, 25.221738483332455, 35.969183208222816],
        'IN PEG8000':[0.0, 4.524801758448074, 8.11644979887675, 12.389195930006638, 17.495086963814828],

        'ProTaCfull PEG400':[0.0, 3.500291491848216, 6.78629121907027,10.679216739212201, 18.824786382893887, 31.406888773045576],
        'ProTaCfull PEG8000':[0.0, 1.82679404453762, 5.620378828289341, 9.42840148326031, 13.430483963787246, 15.67488745743924],
        'ProTaCfullw PEG400': [0.0,3.9039286051722417, 7.53924991079708, 11.80938477246735, 17.548162840836117, 33.89815280798841],
        'ProTaCfullw PEG8000': [0.0, 2.0415339753526602, 6.252653549726252, 10.441671733341522, 14.80355928080827, 17.231747116241387]
    }

    vol_fractions_split = {
        ('ACTR','PEG400'):[0.0, 2.513025017902774, 9.070974615307184, 14.007858020094648, 19.165795905693635, 38.3238509093467],
        ('ACTR','PEG8000'):[0.08749762873090164, 5.6772127298160475, 10.347101295279533, 13.814139775699514, 19.899146088273163],
        ('IN','PEG400'):[0.0, 3.707575138455711, 7.7762373054322875, 16.785859379067215, 25.221738483332455, 35.969183208222816],
        ('IN','PEG8000'):[0.0, 4.524801758448074, 8.11644979887675, 12.389195930006638, 17.495086963814828],
        ('ProTaCfull','PEG400'):[0.0, 3.500291491848216, 6.78629121907027,10.679216739212201, 18.824786382893887, 31.406888773045576],
        ('ProTaCfull','PEG8000'):[0.0, 1.82679404453762, 5.620378828289341, 9.42840148326031, 13.430483963787246, 15.67488745743924],
        
        ('ProTaCfullw', 'PEG400'):[0.0, 3.9039286051722417, 7.53924991079708, 11.80938477246735, 17.548162840836117, 33.89815280798841],
        ('ProTaCfullw', 'PEG8000'):[0.0, 2.0415339753526602, 6.252653549726252, 10.441671733341522, 14.80355928080827, 17.231747116241387]
    }




    ACTR[('exp','PEG400')]  = {
        'vv':np.array(vol_fractions_systems['ACTR PEG400']),
        'expRg':np.array( [ ACTR_exp['PEG400'][perc] for perc in vol_fractions_systems['ACTR PEG400'] ]),
        'expRee':np.array([ ACTR_exp['PEG400'][perc] for perc in vol_fractions_systems['ACTR PEG400'] ])*np.sqrt(5)}
    ACTR[('exp','PEG8000')] = {
        'vv':np.array(vol_fractions_systems['ACTR PEG8000']),
        'expRg':np.array( [ ACTR_exp['PEG8000'][perc] for perc in vol_fractions_systems['ACTR PEG8000'] ]),
        'expRee':np.array([ ACTR_exp['PEG8000'][perc] for perc in vol_fractions_systems['ACTR PEG8000'] ])*np.sqrt(5)
    }

    IN[('exp','PEG400')]  = {
        'vv':np.array(vol_fractions_systems['IN PEG400']),
        'expRg':np.array( [ IN_exp['PEG400'][perc] for perc in vol_fractions_systems['IN PEG400'] ]),
        'expRee':np.array([ IN_exp['PEG400'][perc] for perc in vol_fractions_systems['IN PEG400'] ])*np.sqrt(5)}
    IN[('exp','PEG8000')] = {
        'vv':np.array(vol_fractions_systems['IN PEG8000']),
        'expRg':np.array( [ IN_exp['PEG8000'][perc] for perc in vol_fractions_systems['IN PEG8000'] ]),
        'expRee':np.array([ IN_exp['PEG8000'][perc] for perc in vol_fractions_systems['IN PEG8000'] ])*np.sqrt(5)}




    for protein,data_dict in zip(['ACTR','IN'],[ACTR, IN]):
        
        for l in [0.0, 0.1, 0.2, 0.3]:
            
            for peg in ["PEG400","PEG8000"]:
                data_dict[(l, peg)] = {
                    'vv':data.loc[(protein,peg,l)].index.to_numpy(),
                    'simRg':data.loc[(protein,peg,l),'simRg'].to_numpy(),
                    'simRgSEM':data.loc[(protein,peg,l),'simRgSEM'].to_numpy(),
                    'simRee':data.loc[(protein,peg,l),'simRee'].to_numpy(),
                    'simReeSEM':data.loc[(protein,peg,l),'simReeSEM'].to_numpy(),
                    'relcom_rg':calculate_relative_crowding(data.loc[(protein,peg,l),'simRg'].to_numpy()),
                    'relcom_ree':calculate_relative_crowding(data.loc[(protein,peg,l),'simRee'].to_numpy())
                }


            data_dict[(l,'PEG400')]['chi2'] = {}
            data_dict[(l,'PEG8000')]['chi2'] = {}
            data_dict[(l,'both')] = {'chi2':{}}

            for obs,obs_exp in zip(['simRg','simRee','relcom_rg','relcom_ree'],['expRg','expRee','relcom_rg','relcom_ree']):
                for peg in ["PEG400","PEG8000"]:
                    chi2 = calc_chi_squared(data_dict[(l,peg)][obs][1:], data_dict[('exp',peg)][obs_exp][1:], reduced=True)
                    data_dict[(l,peg)]['chi2'][obs]  = chi2
                
                sim = np.concatenate((data_dict[(l,"PEG400")][obs][1:], data_dict[(l,"PEG8000")][obs][1:]), axis=None)
                exp = np.concatenate((data_dict[('exp',"PEG400")][obs_exp][1:], data_dict[('exp',"PEG8000")][obs_exp][1:]), axis=None)
                chi2 = calc_chi_squared(sim, exp, reduced=True)
                data_dict[(l,"both")]['chi2'][obs]  = chi2
        print(protein,data_dict)

        with open(F"./protein-crowding__data__{protein}.pkl", "wb") as pf:
            pickle.dump(data_dict,pf)
