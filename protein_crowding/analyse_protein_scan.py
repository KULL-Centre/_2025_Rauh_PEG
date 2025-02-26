#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 15:30:55 2023

@author: mhz916
"""
import pandas as pd
import numpy as np
import mdtraj as md
import itertools
import os
import MDAnalysis
from MDAnalysis import transformations


def initProteins():
    sequences = pd.DataFrame(columns=['pH','ionic','fasta','temp','N_res'])
    # Sequences
    fasta_ACTR = """GTQNRPLLRNSLDDLVGPPSNLEGQSDERALLDQLHTLLSNTDATGLEEIDRALGIPELVNQGQALEPKQD""".replace('\n', '') # DOI: 10.1073/pnas.1322611111
    fasta_aSyn = """MDVFMKGLSKAKEGVVAAAEKTKQGVAEAAGKTKEGVLYVGSKTKEGVVHGVATVAEKTKEQVT
NVGGAVVTGVTAVAQKTVEGAGSIAAATGFVKKDQLGKNEEGAPQEGILEDMPVDPDNEAYEMPSEEGYQDYEPEA""".replace('\n', '')
    fasta_aSynM5DP5A = """MDVFMKGLSKAKEGVVAAAEKTKQGVAEAAGKTKEGVLYVGSKTKEGVVHGVATVAEKTKEQVT
NVGGAVVTGVTAVAQKTVEGAGSIAAATGFVKKAQLGKNEEGAPQEGILEAMPVAPANEAYEMPSEEGYQAYEPEA""".replace('\n', '')
    fasta_A1 = """GSMASASSSQRGRSGSGNFGGGRGGGFGGNDNFGRGGNFSGRGGFGGSRGGGGYGGSGDGYNGFGNDGSNFGGGGSYNDFGNYNNQ
SSNFGPMKGGNFGGRSSGGSGGGGQYFAKPRNQGGYGGSSSSSSYGSGRRF""".replace('\n', '')
    fasta_protac =  """CEEGGEEEEEEEEGDGEEEDGDEDEEAESATGKRAAEDDEDDDVDTKKQKTDED""".replace('\n', '') # DOI: 10.1073/pnas.1322611111
    fasta_protan =  """CDAAVDTSSEITTKDLKEKKEVVEEAENGRDAPANGNAENEENGEQEADNEVDEE""".replace('\n', '') # DOI: 10.1073/pnas.1322611111
    fasta_protacF = """SDAAVDTSSEITTKDLKEKKEVVEEAENGRDAPANGNAENEENGEQEADNEVDEECEEGGEEEEEEEEGDGEEEDGDEDEEAESATGKRAAEDDEDDDVDTKKQKTDEDC""".replace('\n', '') # DOI: 10.1073/pnas.1322611111
    fasta_protanF = """CDAAVDTSSEITTKDLKEKKEVVEEAENGRDAPANGNAENEENGEQEADNEVDEECEEGGEEEEEEEEGDGEEEDGDEDEEAESATGKRAAEDDEDDDVDTKKQKTDEDD""".replace('\n', '') # DOI: 10.1073/pnas.1322611111
    fasta_in =      """CAQEEHEKAHSNFRAMASDFNLPPVVAKEIVASCDKCQLKGEAMHGQVD""".replace('\n', '') # DOI: 10.1073/pnas.1322611111
    # Loading data into dataframe
    sequences.loc['PEG400']  = dict(temp=295,pH=7.0,fasta=['J']*9,  ionic=0.11,N_res=9) #   400/44.05 = 9
    sequences.loc['PEG2050'] = dict(temp=295,pH=7.0,fasta=['J']*47, ionic=0.11,N_res=47) #  2050/44.05 = 47
    sequences.loc['PEG4600'] = dict(temp=295,pH=7.0,fasta=['J']*105,ionic=0.11,N_res=105) #  4600/44.05 = 105
    sequences.loc['PEG6000'] = dict(temp=295,pH=7.0,fasta=['J']*135,ionic=0.11,N_res=135) #  6000/44.05 = 135
    sequences.loc['PEG8000'] = dict(temp=295,pH=7.0,fasta=['J']*181,ionic=0.11,N_res=181) #  8000/44.05 = 181
    sequences.loc['PEG35000']= dict(temp=295,pH=7.0,fasta=['J']*794,ionic=0.11,N_res=794) # 35000/44.05 = 794
    sequences.loc['ACTR']    = dict(temp=295,pH=7.0,fasta=list(fasta_ACTR),ionic=0.11,N_res=len(fasta_ACTR))
    sequences.loc['aSyn']    = dict(temp=277,pH=7.4,fasta=list(fasta_aSyn),ionic=0.25,N_res=len(fasta_aSyn))
    sequences.loc['aSynM5DP5A']  = dict(temp=277,pH=7.4,fasta=list(fasta_aSynM5DP5A),ionic=0.25,N_res=len(fasta_aSynM5DP5A))
    sequences.loc['A1']          = dict(temp=277,pH=7.0,fasta=list(fasta_A1),ionic=0.15,N_res=len(fasta_A1))
    sequences.loc['ProTaC']      = dict(temp=295,pH=7.0,fasta=list(fasta_protac), ionic=0.11,N_res=len(fasta_protac))
    sequences.loc['ProTaN']      = dict(temp=295,pH=7.0,fasta=list(fasta_protan), ionic=0.11,N_res=len(fasta_protan))
    sequences.loc['ProTaCfull']  = dict(temp=295,pH=7.0,fasta=list(fasta_protacF),ionic=0.11,N_res=len(fasta_protacF))
    sequences.loc['ProTaNfull']  = dict(temp=295,pH=7.0,fasta=list(fasta_protanF),ionic=0.11,N_res=len(fasta_protanF))
    sequences.loc['IN']          = dict(temp=295,pH=7.0,fasta=list(fasta_in),     ionic=0.11,N_res=len(fasta_in))
    #
    sequences.index.name = 'seq_name'
    return sequences


def genParams(df,proteins,composition,temp,ionic):
    kT = 8.3145*temp*1e-3
    fepsw = lambda T : 5321/T+233.76-0.9297*T+0.1417*1e-2*T*T-0.8292*1e-6*T**3
    epsw = fepsw(temp)
    lB = 1.6021766**2/(4*np.pi*8.854188*epsw)*6.022*1000/kT
    pH = proteins.loc[composition.index[0]].pH
    lj_eps = 0.2*4.184
    # Calculate the inverse of the Debye length
    yukawa_kappa = np.sqrt(8*np.pi*lB*ionic*6.022/10)
    charge = []
    fasta_termini = []
    for name in composition.index:
        fasta = composition.loc[name].fasta.copy()
        r = df.copy()
        # Set the charge on HIS based on the pH of the protein solution
        r.loc['H','q'] = 1. / ( 1 + 10**(pH-6) )
        if name[:3] != 'PEG':
            r.loc['X'] = r.loc[fasta[0]]
            r.loc['Z'] = r.loc[fasta[-1]]
            r.loc['X','q'] = r.loc[fasta[0],'q'] + 1.
            r.loc['Z','q'] = r.loc[fasta[-1],'q'] - 1.
            fasta[0] = 'X'
            fasta[-1] = 'Z'
        # Calculate the prefactor for the Yukawa potential
        charge.append([r.loc[a].q*np.sqrt(lB*kT) for a in fasta])
        fasta_termini.append(fasta)
    composition['charge'] = charge
    composition['fasta_termini'] = fasta_termini
    return composition, yukawa_kappa, lj_eps


def xy_spiral_array(n, delta=0, arc=.38, separation=.7):
    """
    create points on an Archimedes' spiral
    with `arc` giving the length of arc between two points
    and `separation` giving the distance between consecutive
    turnings
    """
    def p2c(r, phi):
        """
        polar to cartesian
        """
        return (r * np.cos(phi), r * np.sin(phi))
    r = arc
    b = separation / (2 * np.pi)
    phi = float(r) / b
    coords = []
    for i in range(n):
        coords.append(list(p2c(r, phi))+[0])
        phi += float(arc) / r
        r = b * phi
    return np.array(coords)+delta
