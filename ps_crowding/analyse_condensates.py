#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 12:37:05 2023

@author: Arrien S. Rauh
"""
import os
import sys
import math
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import MDAnalysis as mda
import mdtraj as md
import string
from scipy.optimize import least_squares, curve_fit
from mpl_toolkits.axes_grid1 import make_axes_locatable
from cycler import cycler
import pickle


sys.path.append("/home/people/arrrau/software/BLOCKING/")
from main import BlockAnalysis


#######################
#### Plot Settings ####
#######################
params = {'legend.fontsize': 'x-large',
          'figure.figsize': (15, 10),
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large' }
plt.rcParams.update(params)
plt.rc('axes', prop_cycle = cycler('color', ['#377eb8', '#ff7f00', '#4daf4a',
                  '#e41a1c', '#984ea3', '#f781bf', '#a65628', 
                  '#999999', '#dede00']))
cm = ['#377eb8', '#ff7f00', '#4daf4a', '#e41a1c', '#984ea3',
                  '#f781bf', '#a65628', 
                  '#999999', '#dede00']
#######################


def get_number_PEG_chains_wv(wv_fraction, mw_peg, volume_simulation_box, NAv=6.022e23):
    """
    Calculate the number of polymer chains.

    Parameters
    ----------
    wv_fraction : TYPE
        DESCRIPTION.
    mw_peg : TYPE
        DESCRIPTION.
    volume_simulation_box : TYPE
        DESCRIPTION.
    NAv : TYPE, optional
        DESCRIPTION. The default is 6.022e23.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    molarity_peg = (wv_fraction*10)/mw_peg
    return int(np.round(molarity_peg*volume_simulation_box*NAv))


def get_number_PEG_chains_vv(vol_fraction, mw, v, density=1.12, NA=6.022e23):
    return int(np.round(NA*((density*(max(vol_fraction/100*v,0)))/mw))) # /mol * ((g/cm^3*cm^3)*))


def calculate_molar_volume(van_der_waals_radius):
    avogadro_number = 6.022e23  # Avogadro's number
    molar_volume = (4/3) * math.pi * (van_der_waals_radius**3) * avogadro_number
    return molar_volume


def molar_volume_sequencec(sequence,df_residues):
    if type(sequence) != list:
        sequence = list(sequence)
    return np.sum(df_residues.loc[sequence,'sigmas'].apply(calculate_molar_volume))


def molar_from_n(n_chains, volume, NA=6.022e23):
    return n_chains/(NA*volume)


def phi_from_molar(sequence,df_residues,prefix=1):
    MW = np.sum(df_residues.loc[sequence,'MW'])
    if 'J' in sequence: # peg
        return (prefix*MW)/1120
    else: # protein
        return (prefix*MW)/1310


f_lin = lambda x,a,b: a*x+b


def fit_with_errors(x, y, xerr, yerr):
    # Define the errors for the curve_fit function
    weights = 1.0 / np.sqrt(xerr**2 + yerr**2)  # Combine errors in x and y
    # Perform the curve fitting
    popt, pcov = curve_fit(f_lin, x, y, [1,-4], sigma=weights)
    # Get the standard deviations of the parameters
    perr = np.sqrt(np.diag(pcov))[0]
    return popt, perr


def fit_without_errors(x, y):
    # Perform the curve fitting
    popt, pcov = curve_fit(f_lin, x, y, [1,-4])
    # Get the standard deviations of the parameters
    perr = np.sqrt(np.diag(pcov))
    return popt, perr


def calculate_chi_delta(k,phi1,N1):
    denominator = 2*phi1*k*N1
    return -0.5-(1/denominator)


def calculate_chi_deltaE(k,phi1,N1,k_err=0,phi1_err=0,N1_err=0):
    chi = -0.5-(1/(2*phi1*k*N1))
    partial_k    = 1/(2*phi1*(k**2)*N1)
    partial_phi1 = 1/(2*(phi1**2)*k*N1)
    partial_N1   = 1/(2*phi1*k*(N1**2))
    sum_parts = np.sum([(partial_k**2*k_err**2),
                        (partial_phi1**2*phi1_err**2),
                        (partial_N1**2*N1_err**2)])
    return np.abs(chi)*np.sqrt(sum_parts)




def determine_cutoffs(h, protein):
    lz = (h.shape[1]+1)
    edges = np.arange(-lz/2.,lz/2.,1)/10
    dz = (edges[1]-edges[0])/2.
    z = edges[:-1]+dz
    profile = lambda x,a,b,c,d : .5*(a+b)+.5*(b-a)*np.tanh((np.abs(x)-c)/d)
    residuals = lambda params,*args: (args[1] - profile(args[0], *params))
    hm = np.mean(h,axis=0)
    z1 = z[z>0]
    h1 = hm[z>0]
    z2 = z[z<0]
    h2 = hm[z<0]
    # p0=[hm.min(),hm.max(),3,1]
    p0=[1,1,1,1]
    # res1 = least_squares(residuals, x0=p0, args=[z1, h1], bounds=([0]*4,[1e3]*4))
    # res2 = least_squares(residuals, x0=p0, args=[z2, h2], bounds=([0]*4,[1e3]*4))
    res1 = least_squares(residuals, x0=p0, args=[z1, h1], bounds=([0]*4,[100]*4))
    res2 = least_squares(residuals, x0=p0, args=[z2, h2], bounds=([0]*4,[100]*4))
    
    if (res1.x[3]>res1.x[2]) or (res2.x[3]>res2.x[2]):
        zDS = res1.x[2] if res1.x[2]>res2.x[2] else res2.x[2]
        print(zDS)
        zDS = 10 if zDS<1 else zDS
        cutoffs1 = [zDS,-zDS]
        cutoffs2 = [zDS+25,-zDS-25]
    else:
        cutoffs1 = [res1.x[2]-.5*res1.x[3],-res2.x[2]+.5*res2.x[3]]
        #cutoffs1 = [res1.x[2]-res1.x[3],-res2.x[2]+res2.x[3]]
        cutoffs2 = [res1.x[2]+10*res1.x[3],-res2.x[2]-10*res2.x[3]]
    
    # cutoffs1 = [res1.x[2]-res1.x[3],-res2.x[2]+res2.x[3]]
    # cutoffs2 = [res1.x[2]+6*res1.x[3],-res2.x[2]-6*res2.x[3]]
    # cutoffs1 = [7,-7]
    
    if np.abs(cutoffs2[1]/cutoffs2[0]) > 2:
        print('WRONG',protein,cutoffs1,cutoffs2)
        print(res1.x,res2.x)
    if np.abs(cutoffs2[1]/cutoffs2[0]) < 0.5:
        print('WRONG',protein,cutoffs1,cutoffs2)
        print(res1.x,res2.x)
        plt.plot(z1, h1)
        plt.plot(z2, h2)
        plt.plot(z1,profile(z1,*res1.x),color='tab:blue')
        plt.plot(z2,profile(z2,*res2.x),color='tab:orange')
        cutoffs2[0] = -cutoffs2[1]
        print(cutoffs2)
    return cutoffs1, cutoffs2, z


def plot_profile(hm, c1, c2, z, l):
    plt.figure(figsize=(20,15))
    plt.plot(z,hm)
    plt.ylim(0.01,100)

    plt.vlines(c1[0],0,hm.max()+10,ls='--',color='r')
    plt.vlines(c1[1],0,hm.max()+10,ls='--',color='r')
    plt.vlines(c2[0],0,hm.max()+10,ls='--')
    plt.vlines(c2[1],0,hm.max()+10,ls='--')

    plt.hlines(1,-75,75,ls='--',color='black')
    
    plt.vlines(0,0,hm.max()+10,ls='--',color='black')
    plt.yscale('log')
    plt.ylim(1e-2,1e2)
    plt.ylabel('[A1]  /  mM', fontsize=25)
    plt.xlim(-75,75)
    plt.xlabel('$z$  /  nm', fontsize=25)
    plt.savefig(F"profile_{l}.pdf",dpi=300)
    plt.close()


def extract_concentrations(h, bool_den, bool_dil):
    denarray = np.apply_along_axis(lambda a: a[bool_den].mean(), 1, h)
    dilarray = np.apply_along_axis(lambda a: a[bool_dil].mean(), 1, h)
    if h.mean()==0.0:
        ratioarray = np.zeros([denarray.size])
    else:
        ratioarray = dilarray/denarray
    den   = np.mean(h,axis=0)[bool_den].mean()
    dil   = np.mean(h,axis=0)[bool_dil].mean()
    ratio = np.mean(ratioarray)
    # 
    block_den   = BlockAnalysis(denarray)
    block_dil   = BlockAnalysis(dilarray)
    block_ratio = BlockAnalysis(ratioarray)

    # Perform block averaging analysis.
    block_den.SEM()
    block_dil.SEM()
    block_ratio.SEM()
    return denarray, dilarray, ratioarray, den, dil, ratio, block_den, block_dil, block_ratio



def calcProfiles_peg(T,L,systems,data,nskip=1200, wdir=".",n_protein=100,
                     proteins_pkl="./proteins.pkl",report_cutoffs=False,
                     den_cutoffs=None,dil_cutoffs=None,dt=0.0005):
    
    """
    Calculates profiles based on multi-chain simulations.
    
    INPUT PARAMETERS
    -----------    
    T : int
        Temperature of simulation in K.
    L : int
        Box size (nm).
    proteins : list
     c   
    data : 
        
    nskip : float
        Number of frames to skip.
    proteins_pkl="./proteins.pkl"
    
    
    RETURNS
    ------------
    
    
    """
    df_proteins = pd.read_pickle(proteins_pkl)
    block_averages = {}
    for i, system in enumerate(systems):
        protein, peg, percentage = system
        Lx, Ly, Lz = L[protein]
        box_vol = (Lx*Ly*Lz)*1e-24 # With the boxlengths in nm.
        n_chains = get_number_PEG_chains_wv(wv_fraction=percentage,
                                            mw_peg=int(peg[3:]),
                                            volume_simulation_box=(Lx*Ly*Lz)*1e-24,
                                            NAv=6.022e23)
        print(F"Analysing {protein} with {n_chains} chains of {peg}:")
        try:
            # Load histogram data
            h = np.load(F'{wdir}/{protein}-{peg}/{T:d}/{n_chains}/{protein}-{peg}_{T}_{n_chains}_protein.npy')
            if percentage == 0.0:
                hp = np.zeros([h.shape[0],h.shape[1]])
            else:
                hp = np.load(F'{wdir}/{protein}-{peg}/{T:d}/{n_chains}/{protein}-{peg}_{T}_{n_chains}_peg.npy')


            # Extract the chain sequence
            fasta = df_proteins.loc[protein].fasta
            fasta_p =  df_proteins.loc[peg].fasta
            # Sequence length
            n_res = len(fasta)
            n_resp = len(fasta_p)

            # Conversion number from count to concentration in mM
            # conv = 100/6.022/n_res/L/L/1e3
            conv  = 1e5/6.022/n_res/Lx/Ly
            convp = 1e5/6.022/n_resp/Lx/Ly

            # Take the frames after nskip and convert bin counts to concentrations.
            h = h[nskip:]*conv
            hp = hp[nskip:]*convp

            cutoffs_den, cutoffs_dil, z = determine_cutoffs(h, protein)
            # if type(den_cutoffs)==pd.core.series.Series and type(dil_cutoffs)==pd.core.series.Series:
            #     cutoffs_den = den_cutoffs.loc[percentage]
                # cutoffs_dil = dil_cutoffs.loc[percentage]
            if type(den_cutoffs[(protein, peg)])==list and type(dil_cutoffs[(protein, peg)])==list:
                cutoffs_den = den_cutoffs[(protein, peg)]
                # cutoffs_dil = dil_cutoffs
            if cutoffs_dil[0]>=75 or cutoffs_dil[1]<=-75:
                cutoffs_dil = [65.0,-65.0]
            if report_cutoffs:
                print(F"Dense phase: {cutoffs_den:.2f}\nDilute phase: {cutoffs_dil:.2f}")

            bool_den = np.logical_and(z<cutoffs_den[0],z>cutoffs_den[1])
            bool_dil = np.logical_or(z>cutoffs_dil[0],z<cutoffs_dil[1])

            # Protein concentrations
            denarray, dilarray, ratioarray, den, dil, ratio, block_den, block_dil, block_ratio = extract_concentrations(h, bool_den, bool_dil)
            # PEG concentrations
            denarrayp, dilarrayp, ratioarrayp, den, dil, ratio, block_denp, block_dilp, block_ratiop = extract_concentrations(hp, bool_den, bool_dil)

            # Overall concentrations in the system
            denarrays   = denarray + denarrayp
            dilarrays   = dilarray + dilarrayp
            ratioarrays = dilarrays/denarrays
            block_dens   = BlockAnalysis(denarrays)
            block_dils   = BlockAnalysis(dilarrays)
            block_ratios = BlockAnalysis(ratioarrays)

            block_dens.SEM()
            block_dils.SEM()
            block_ratios.SEM()

            C_protein = molar_from_n(n_protein,volume=(15*15*150*1e-24), NA=6.022e23)*1000
            C_peg     = molar_from_n(n_chains, volume=(15*15*150*1e-24), NA=6.022e23)*1000
            print(block_dil.av,block_den.av)
            # Add the data to the dataframe.
            # Protein concentrations
            data.loc[(protein,peg,percentage),'Cdil']      = block_dil.av
            data.loc[(protein,peg,percentage),'Cden']      = block_den.av
            data.loc[(protein,peg,percentage),'Ratio']     = block_ratio.av
            data.loc[(protein,peg,percentage),'CdilSEM']   = block_dil.sem
            data.loc[(protein,peg,percentage),'CdenSEM']   = block_den.sem
            data.loc[(protein,peg,percentage),'RatioSEM']  = block_ratio.sem
            # PEG concentrations
            data.loc[(protein,peg,percentage),'CdilP']      = block_dilp.av
            data.loc[(protein,peg,percentage),'CdenP']      = block_denp.av
            data.loc[(protein,peg,percentage),'RatioP']     = block_ratiop.av
            data.loc[(protein,peg,percentage),'CdilPSEM']   = block_dilp.sem
            data.loc[(protein,peg,percentage),'CdenPSEM']   = block_denp.sem
            data.loc[(protein,peg,percentage),'RatioPSEM']  = block_ratiop.sem
            # Overall concentrations
            data.loc[(protein,peg,percentage),'CdilS']      = block_dils.av
            data.loc[(protein,peg,percentage),'CdenS']      = block_dens.av
            data.loc[(protein,peg,percentage),'RatioS']     = block_ratios.av
            data.loc[(protein,peg,percentage),'CdilSSEM']   = block_dils.sem
            data.loc[(protein,peg,percentage),'CdenSSEM']   = block_dens.sem
            data.loc[(protein,peg,percentage),'RatioSSEM']  = block_ratios.sem
            #
            data.at[(protein,peg,percentage),'Cdilarray'] = dilarray
            data.at[(protein,peg,percentage),'Cdenarray'] = denarray
            data.at[(protein,peg,percentage),'ratioarray']= ratioarray
            #
            data.at[(protein,peg,percentage),'CdilarrayP'] = dilarrayp
            data.at[(protein,peg,percentage),'CdenarrayP'] = denarrayp
            data.at[(protein,peg,percentage),'ratioarrayP']= ratioarrayp
            #
            data.at[(protein,peg,percentage),'CdilarrayS'] = dilarrays
            data.at[(protein,peg,percentage),'CdenarrayS'] = denarrays
            data.at[(protein,peg,percentage),'ratioarrayS']= ratioarrays
            #
            data.loc[(protein,peg,percentage),'C_protein'] = C_protein
            data.loc[(protein,peg,percentage),'C_peg']     = C_peg
            # Phi data
            data.loc[(protein,peg,percentage),'phi1'] = C_protein * phi_from_molar(fasta,  df_residues,prefix=1e-3)
            data.loc[(protein,peg,percentage),'phi2'] = C_peg     * phi_from_molar(fasta_p,df_residues,prefix=1e-3)
            data.loc[(protein,peg,percentage),'phi0'] = 1-data.loc[(protein,peg,percentage),'phi1']-data.loc[(protein,peg,percentage),'phi2']
            ## Phi1
            data.loc[(protein,peg,percentage),'phi1_dil']      = block_dil.av  * phi_from_molar(fasta,df_residues,prefix=1e-3)
            data.loc[(protein,peg,percentage),'phi1_den']      = block_den.av  * phi_from_molar(fasta,df_residues,prefix=1e-3)
            data.loc[(protein,peg,percentage),'phi1_dilSEM']   = block_dil.sem * phi_from_molar(fasta,df_residues,prefix=1e-3)
            data.loc[(protein,peg,percentage),'phi1_denSEM']   = block_den.sem * phi_from_molar(fasta,df_residues,prefix=1e-3)
            ## Phi2
            data.loc[(protein,peg,percentage),'phi2_dil']      = block_dilp.av  * phi_from_molar(fasta_p,df_residues,prefix=1e-3)
            data.loc[(protein,peg,percentage),'phi2_den']      = block_denp.av  * phi_from_molar(fasta_p,df_residues,prefix=1e-3)
            data.loc[(protein,peg,percentage),'phi2_dilSEM']   = block_dilp.sem * phi_from_molar(fasta_p,df_residues,prefix=1e-3)
            data.loc[(protein,peg,percentage),'phi2_denSEM']   = block_denp.sem * phi_from_molar(fasta_p,df_residues,prefix=1e-3)
            # General info
            data.at[(protein,peg,percentage),'dilcutoff'] = cutoffs_dil
            data.at[(protein,peg,percentage),'dencutoff'] = cutoffs_den
            data.loc[(protein,peg,percentage),'n_protein'] = n_protein
            data.loc[(protein,peg,percentage),'n_peg']     = n_chains
            data.loc[(protein,peg,percentage),'n_frames'] = len(ratioarray)
            data.loc[(protein,peg,percentage),'sampling'] = len(ratioarray)*dt
            data.at[(protein,peg,percentage),'h']= h
            data.at[(protein,peg,percentage),'hp']= hp

            tie_line = np.array([[data.loc[(protein,peg,percentage),'phi1_dil'],data.loc[(protein,peg,percentage),'phi1'],data.loc[(protein,peg,percentage),'phi1_den']],
                                 [data.loc[(protein,peg,percentage),'phi2_dil'],data.loc[(protein,peg,percentage),'phi2'],data.loc[(protein,peg,percentage),'phi2_den']]])
            popt_tie, perr_tie = curve_fit(f=f_lin,xdata=tie_line[0,:],ydata=tie_line[1,:],p0=[-1,4])
            perr_tie = np.sqrt(np.diag(perr_tie))
            data.at[(protein,peg,percentage),'tie_line']   = tie_line
            data.loc[(protein,peg,percentage),'tie_slope']  = popt_tie[0]
            data.loc[(protein,peg,percentage),'tie_slopeE'] = perr_tie[0]
            data.loc[(protein,peg,percentage),'tie_int']    = popt_tie[1]
            data.loc[(protein,peg,percentage),'tie_intE']   = perr_tie[1]
            # Chi delta

            data.loc[(protein,peg,percentage),'chi_d']  = calculate_chi_delta(k=popt_tie[0],
                                                                              phi1=data.loc[(protein,peg,percentage),'phi1'],
                                                                              N1=n_res)
            #data.loc[(protein,peg,percentage),'chi_dE'] = calculate_chi_deltaE(k=popt_tie[0],
            #                                                                   phi1=block_dil.av,
            #                                                                   N1=n_res,
            #                                                                   k_err=perr_tie[0],
            #                                                                   phi1_err=block_dil.sem * phi_from_molar(fasta,df_residues,prefix=1e-3))
            print(data.loc[(protein,peg,percentage),])

#            if not np.isclose(den, block_den.av, 0.1):
#                print(F" - Cden: Average and Block Average differ: {block_den.av-den}")
#            if not np.isclose(dil, block_dil.av, 0.1):
#                print(F" - Cdil: Average and Block Average differ: {block_dil.av-dil}")
#            if not np.isclose(ratio, block_ratio.av, 0.1):
#                print(F" - Ratio: Average and Block Average differ: {block_ratio.av-ratio}")
#            if not np.isclose(ratio, block_dil.av/block_den.av, 0.1):
#                print(F" - Ratio: Timeseries based does not match ratio of Block Averages: {block_ratio.av-(block_dil.av/block_den.av)}")


            block_averages[(protein, peg, percentage)] = {'den':block_den,'dil':block_dil,'ratio':block_ratio,
                                                          'denp':block_denp,'dilp':block_dilp,'ratiop':block_ratiop,
                                                          'dens':block_dens,'dils':block_dils,'ratios':block_ratios}
            
        except FileNotFoundError:
            print(F'DATA NOT FOUND FOR {protein} with {n_chains} chains of {peg}')
            data.drop(index=[(protein,peg,percentage)])
    return data, block_averages



def dG_transfer(cdil, cden, T=300.00, R=8.31446261815324):
    """
    Calculate the transfer energy of chains from the dense phase to the dilute phase.
    """
    return R*T*np.log(cdil/cden)


def error_dG_transfer(cdil, cden, cdilE, cdenE, T=300.00, R=8.31446261815324):
    """
    Propagate the error for a calculation of the transfer energy of chains from the dense phase to the dilute phase.
    """
    cden_part = cdenE**2 * (-R*T/cden)**2
    cdil_part = cdilE**2 * ( R*T/cdil)**2
    return np.sqrt(cden_part + cdil_part)


def dG_transfer_ratio(ratio, T=300.00, R=8.31446261815324):
    """
    Calculate the transfer energy of chains from the dense phase to the dilute phase.
    """
    return R*T*np.log(ratio)


def error_dG_transfer_ratio(ratio, ratioE, T=300.00, R=8.31446261815324):
    """
    Propagate the error for a calculation of the transfer energy of chains from the error of the ratio.
    """
    return np.abs((R*T)*(ratioE/ratio))


def ddG_transfer(dG_begin, dG_end):
    """
    Calculate an energy difference.
    """
    return dG_end - dG_begin


def error_ddG_transfer(dG_beginE, dG_endE):
    """
    Propogate the error in energies when calcluating an energy difference.
    """
    return np.sqrt(dG_beginE**2 + dG_endE**2)




############
# Gas Constant for energy calculations:
R = 8.31446261815324/1000 # kJ/(mol K)


temp = 298
df_proteins = pd.read_pickle("proteins.pkl")
df_residues = pd.read_csv("./residues.csv").set_index('one')#"calvados2_chvar.csv")#.set_index('one')

# systems  = [('A1','PEG8000'),('aSyn','PEG8000'),#('aSynM5DP5A' ,'PEG8000'),
#             ('A1','PEG400'), #('aSyn','PEG400'), ('aSynM5DP5A' ,'PEG400'),
#             ("AroM", "PEG8000"),("AroMM","PEG8000"),('A1','PEG20000')
#         ]

# peg_numbers = {('A1','PEG8000'):[[100,508],[100,381],[100,318],[100,254],[100,191],[100,127],[100,64],[100,0]],
#                ('aSyn','PEG8000'):[[100,508],[100,381],[100,318],[100,254],[100,191],[100,127],[100,64],[100,0]],
#                ('aSynM5DP5A' ,'PEG8000'):[[100,508],[100,381],[100,318],[100,254],[100,191],[100,127],[100,64],[100,0]],
#                ("A1","PEG400"):[[100, 1287],[100,2554],[100,3841],[100,5108]],#[100,6395],[100,7662],[100 ,10216]]
#                ("aSyn","PEG400"):[[100, 1287],[100,2554],[100,3841],[100,5108]],#[100,6395],[100,7662],[100 ,10216]]
#                ("aSynM5DP5A","PEG400"):[[100, 1287],[100,2554],[100,3841],[100,5108]],#[100,6395],[100,7662],[100 ,10216]]
#                ("AroM", "PEG8000"):[[100,0],[100,64],[100,191],[100,254]], #,[100,127]
#                ("AroMM","PEG8000"):[[100,0],[100,64],[100,127],[100,191],[100,254]],
#                ("A1","PEG20000"):[[100, 0],[100,26],[100,51],[100,76],[100,101]]
#               }
# peg_percentages = {('A1','PEG8000'):[0,2.5,5,7.5,10,12.5,15,20],
#                    ('aSyn','PEG8000'):[0,2.5,5,7.5,10,12.5,15,20],
#                    ('aSynM5DP5A' ,'PEG8000'):[0,2.5,5,7.5,10,12.5,15,20],
#                    ("A1","PEG400"):[0,2.5,5,7.5,10],
#                    ("aSyn","PEG400"):[0,2.5,5,7.5,10],
#                    ("aSynM5DP5A","PEG400"):[0,2.5,5,7.5,10],
#                    ("AroM", "PEG8000"):[0,2.5,5,7.5,10], # ,[100,127]
#                    ("AroMM","PEG8000"):[0,2.5,5,7.5,10],
#                    ("A1","PEG20000"):[0,2.5,5,7.5,10],
#               }


n_protein = 100

systems = {'A1':{
               298:{
                   'PEG400':[0.0,2.53293,5.02651,7.55944,10.053],
                   'PEG2000':[0.0,2.5,5.0],
                   'PEG8000':[0.0,2.5,5.0,7.5,10.0,12.5,15.0,20.0],
                   'PEG12000':[0.0,2.5,5.0],
                   'PEG20000':[0.0,2.5,5.0,7.5,10.0]}},
           'AroMM':{
               298:{
                   'PEG8000':[0.0,2.5,5.0,7.5,10.0]}},
           'AroM':{
               298:{
                   'PEG8000':[0.0,2.5,5.0,7.5,10.0]}},
           'aSyn':{
               298:{
                   'PEG8000':[0.0,2.5,5.0,7.5,10.0,12.5,15.0,20.0]}},
           # 'Ddx4':{
           #     293:{}}
           }


box_sizes = {'A1':(15,15,150),
             'AroMM':(15,15,150),
             'AroM':(15,15,150),
             'aSyn':(15,15,150),
             'Ddx4':(17,17,300)
    }


cols = ['Cden', 'CdenSEM','Cdil', 'CdilSEM','Ratio', 'RatioSEM',
        'CdenP', 'CdenPSEM','CdilP', 'CdilPSEM','RatioP', 'RatioPSEM',
        'CdenS', 'CdenSSEM','CdilS', 'CdilSSEM','RatioS', 'RatioSSEM',
        'Cdilarray', 'Cdenarray','ratioarray',
        'CdilarrayP', 'CdenarrayP','ratioarrayP',
        'CdilarrayS', 'CdenarrayS','ratioarrayS',
        'phi0','phi1','phi2',
        'phi1_dil', 'phi1_dilSEM', 'phi2_dil', 'phi2_dilSEM',
        'phi1_den', 'phi1_denSEM', 'phi2_den', 'phi2_denSEM',
        'tie_line','tie_slope', 'tie_slopeE','tie_int','tie_intE','chi_d','chi_dE',
        'n_protein', 'n_peg', 'C_protein', 'C_peg',
        'phi1_array','phi2_array',
        'dilcutoff', 'dencutoff','n_frames', 'sampling',
        'h','hp']

indices = []
for protein in systems.keys():
    for peg in (systems[protein][temp].keys()):
        for peg_perc in (systems[protein][temp][peg]):
            indices.append((protein, peg, peg_perc))

data = pd.DataFrame(columns=cols,
                    index=pd.MultiIndex.from_tuples(indices,names=['protein',
                                                                   'peg',
                                                                   'percentage']))

cutoff_den_dict = {('A1','PEG8000'):[5.0,-5.0],
                   ('aSyn','PEG8000'):(0.0),
                   ('A1','PEG400'):(0.0),
                   ("AroM", "PEG8000"):(0.0),
                   ("AroMM","PEG8000"):(0.0),
                   ('A1','PEG20000'):[5.0,-5.0],
                   ('A1','PEG2000'):(0.0),
                   ('A1','PEG12000'):(0.0)}
cutoff_dil_dict = {('A1','PEG8000'):[0.0,-0.0],
                   ('aSyn','PEG8000'):(0.0),
                   ('A1','PEG400'):(0.0),
                   ("AroM", "PEG8000"):(0.0),
                   ("AroMM","PEG8000"):(0.0),
                   ('A1','PEG20000'):[0.0,-0.0],
                   ('A1','PEG2000'):(0.0),
                   ('A1','PEG12000'):(0.0)}

t0 = time.time()
data, block_averages = calcProfiles_peg(T=temp,L=box_sizes,systems=indices,
                                        data=data, nskip=2000, wdir=".",
                                        proteins_pkl="./proteins.pkl",
                                        n_protein=n_protein,
                                        den_cutoffs=cutoff_den_dict,
                                        dil_cutoffs=cutoff_dil_dict) # nskip=2000 means skipping the first µs


# Energy Calculations
for var in ['','P','S']:
    data[F'dGtransfer{var}'] = data.apply(lambda x: dG_transfer(x[F'Cdil'],x[F'Cden'],
                                                          T=temp, R=R), axis=1)
    data[F'dGtransfer{var}E']= data.apply(lambda x: error_dG_transfer(x[F'Cdil{var}'],
                                                                x[F'Cden{var}'],
                                                                x[F'Cdil{var}SEM'],
                                                                x[F'Cden{var}SEM'],
                                                                T=temp, R=R), axis=1)


for var in ['Ratio','RatioP','RatioS']:
    data[F'dGtransfer{var}'] = data.apply(lambda x: dG_transfer_ratio(x[F'{var}'],
                                                                     T=temp, R=R), axis=1)
    data[F'dGtransfer{var}E']= data.apply(lambda x: error_dG_transfer_ratio(x[F'{var}'],
                                                                           x[F'{var}SEM'],
                                                                           T=temp, R=R), axis=1)




# data['dGtransferP'] = data.apply(lambda x: dG_transfer(x['CdilP'],x['CdenP'],
#                                                       T=temp, R=R), axis=1)
# data['dGtransferPE']= data.apply(lambda x: error_dG_transfer(x['CdilP'],
#                                                             x['CdenP'],
#                                                             x['CdilPSEM'],
#                                                             x['CdenPSEM'],
#                                                             T=temp, R=R), axis=1)

# data['dGtransferRatioP'] = data.apply(lambda x: dG_transfer_ratio(x['ratioP'],
#                                                                  T=temp, R=R), axis=1)
# data['dGtransferRatioPE']= data.apply(lambda x: error_dG_transfer_ratio(x['ratioP'],
#                                                                        x['ratioPSEM'],
#                                                                        T=temp, R=R), axis=1)



# data['dGtransferS'] = data.apply(lambda x: dG_transfer(x['CdilS'],x['CdenS'],
#                                                       T=temp, R=R), axis=1)
# data['dGtransferSE']= data.apply(lambda x: error_dG_transfer(x['CdilS'],
#                                                             x['CdenS'],
#                                                             x['CdilSSEM'],
#                                                             x['CdenSSEM'],
#                                                             T=temp, R=R), axis=1)

# data['dGtransferRatioS'] = data.apply(lambda x: dG_transfer_ratio(x['ratioS'],
#                                                                  T=temp, R=R), axis=1)
# data['dGtransferRatioSE']= data.apply(lambda x: error_dG_transfer_ratio(x['ratioS'],
#                                                                        x['ratioSSEM'],
#                                                                        T=temp, R=R), axis=1)

####################


for i,idx in enumerate(indices):
    if idx[0]=='A1' and idx[1]=='PEG400':
        percentage = float(str(idx[-1])[:3])
        indices[i] = ('A1','PEG400',percentage)
data.index = pd.MultiIndex.from_tuples(indices,names=['protein',
                                                      'peg',
                                                      'percentage'])

data = data[['Cden', 'CdenSEM','Cdil', 'CdilSEM','Ratio', 'RatioSEM',
             'CdenP', 'CdenPSEM','CdilP', 'CdilPSEM','RatioP', 'RatioPSEM',
             'CdenS', 'CdenSSEM','CdilS', 'CdilSSEM','RatioS', 'RatioSSEM',
             'dGtransferRatio', 'dGtransferRatioE',
             'dGtransferRatioP', 'dGtransferRatioPE',
             'dGtransferRatioS', 'dGtransferRatioSE',
             'phi0','phi1','phi2',
             'phi1_dil', 'phi1_dilSEM', 'phi2_dil', 'phi2_dilSEM',
             'phi1_den', 'phi1_denSEM', 'phi2_den', 'phi2_denSEM',
             'tie_line','tie_slope', 'tie_slopeE','tie_int','tie_intE',
             'chi_d','chi_dE',
             'n_protein', 'n_peg', 'C_protein', 'C_peg',
             'Cdilarray', 'Cdenarray','ratioarray',
             'CdilarrayP', 'CdenarrayP','ratioarrayP',
             'CdilarrayS', 'CdenarrayS','ratioarrayS',
             'dGtransfer', 'dGtransferE','dGtransferP',
             'dGtransferPE','dGtransferS', 'dGtransferSE',
             'dilcutoff', 'dencutoff','n_frames', 'sampling', 'h','hp',]]


# Writing the results of the analysis to pickle files.
data.to_pickle(F"./processed_data/slab_analysis__data__OG__09112023.pkl")
with open(F"./processed_data/slab_analysis__block-analysis__OG__09112023.pkl", 'wb') as pf:
    pickle.dump(block_averages, pf)

print(F'Timing {time.time()-t0:.3f}')
