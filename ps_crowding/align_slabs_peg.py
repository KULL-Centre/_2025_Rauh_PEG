#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 14:17:24 2023

@author: Arrien Symon Rauh

Based on S. von Buelow's align_slab.py
"""
import os
import shutil
import time
import numpy as np
import pandas as pd
import MDAnalysis as MDAnalysis
from MDAnalysis import transformations
from argparse import ArgumentParser


def add_topology_bonds(u, n_chains, l_chains):
    offset=0
    for n,l in zip(n_chains, l_chains):
        if n == 0: continue
        bonds = np.array([[],[]]).T

        if type(l) != int: l = int(l)

        bonds_temp = np.concatenate([np.arange(0,l-1),
                                     np.arange(1,l)],
                                    ).reshape(2,l-1).T + offset
        for i in np.arange(0,n):
            bonds = np.concatenate([bonds,bonds_temp+(i*l)],axis=0)
        u.add_bonds(bonds.astype(int))
        offset = bonds[-1,-1]
    return u


def calc_zpatch(z,h):
    cutoff = 0
    ct = 0.
    ct_max = 0.
    zwindow = []
    hwindow = []
    zpatch = []
    hpatch = []
    for ix, x in enumerate(h):
        if x > cutoff:
            ct += x
            zwindow.append(z[ix])
            hwindow.append(x)
        else:
            if ct > ct_max:
                ct_max = ct
                zpatch = zwindow
                hpatch = hwindow
            ct = 0.
            zwindow = []
            hwindow = []
    zpatch = np.array(zpatch)
    hpatch = np.array(hpatch)
    return zpatch, hpatch


def center_slab(name,number,components,temp,input_pdb,start=None,end=None,step=1,input_dcd="t.dcd"):
    # Set path to directory
    path = F'{name}/{temp}/{number}'
    try:
        # Load simulation into Universe
        print(F'{path}/{input_pdb}',F'{path}/{input_dcd}')
        u = MDAnalysis.Universe(F'{path}/{input_pdb}',F'{path}/{input_dcd}',in_memory=True)
    except RuntimeError:
        # Modify create new topology pdb without conect information.
        with open(F'{path}/{input_pdb}', 'r') as top_in:
            with open(F'{path}/{input_pdb[:-4]}_noconect.pdb', 'w') as top_out:
                for line in top_in:
                    if not line.strip("\n").startswith('CONECT'):
                        top_out.write(line)
        # Load simulation into Universe
        u = MDAnalysis.Universe(F'{path}/{input_pdb[:-4]}_noconect.pdb',
                                F'{path}/{input_dcd}',in_memory=True)

    # Generate topology
    u = add_topology_bonds(u, list(components.N), list(components.N_res))


    n_frames = len(u.trajectory[start:end:step])
    ag = u.select_atoms("not name J")
    n_atoms = ag.n_atoms
    print('n_atoms',n_atoms)

    print(u.dimensions)
    L = u.dimensions[0]/10
    lz = u.dimensions[2]
    edges = np.arange(0,lz+1,1)
    dz = (edges[1] - edges[0]) / 2.
    z = edges[:-1] + dz
    n_bins = len(z)
    hs = np.zeros((n_frames,n_bins))


    try:
        with MDAnalysis.Writer(path+'/traj_protein.dcd',n_atoms) as W:
            for t,ts in enumerate(u.trajectory[start:end:step]):
                # shift max density to center
                zpos = ag.positions.T[2]
                h, e = np.histogram(zpos,bins=edges)
                zmax = z[np.argmax(h)]
                ag.translate(np.array([0,0,-zmax+0.5*lz]))
                ts = transformations.wrap(ag)(ts)
                # Round 2
                zpos = ag.positions.T[2]
                h, e = np.histogram(zpos, bins=edges)
                zpatch, hpatch = calc_zpatch(z,h)
                zmid = np.average(zpatch,weights=hpatch)
                ag.translate(np.array([0,0,-zmid+0.5*lz]))
                ts = transformations.wrap(ag)(ts)
                # Round 3
                zpos = ag.positions.T[2]
                h, e = np.histogram(zpos,bins=edges)
                hs[t] = h
                W.write(ag)
    except ZeroDivisionError:
        with MDAnalysis.Writer(path+'/traj_protein.dcd',n_atoms) as W:
            for t,ts in enumerate(u.trajectory[start:end:step]):
                # Skip alignment and just extract histograms
                # And write protein coordinates to file.
                zpos = ag.positions.T[2]
                h, e = np.histogram(zpos,bins=edges)
                hs[t] = h
                W.write(ag)

    np.save(f'{path}/{name:s}_{temp:d}_{number}_protein.npy',hs,allow_pickle=False)
    print(hs.shape)
    hm = np.mean(hs[200:],axis=0)
    conv = 1e8/6.022/(n_atoms/100)/L/L
    print(z.min(),z.max(),lz,lz/4,3*lz/4)
    bool_dil = np.logical_or(z<lz/4,z>3*lz/4)
    print(bool_dil.sum(),bool_dil.shape,hm.shape)
    dil = hm[bool_dil].mean()*conv
    print(dil,'uM, chain length:',(n_atoms/100),', L:',L,'nm')


def align_peg(name,number,components,temp,start=None,end=None,step=1,input_pdb='top.pdb',input_dcd="t.dcd"):
    # Set path to directory
    path = F'{name}/{temp}/{number}'
    # Load simulation into Universe
    try:
        # Load simulation into Universe
        print(F'{path}/{input_pdb}',F'{path}/{input_dcd}')
        u = MDAnalysis.Universe(F'{path}/{input_pdb}',F'{path}/{input_dcd}',in_memory=True)
    except RuntimeError:
        # Modify create new topology pdb without conect information.
        with open(F'{path}/{input_pdb}', 'r') as top_in:
            with open(F'{path}/{input_pdb[:-4]}_noconect.pdb', 'w') as top_out:
                for line in top_in:
                    if not line.strip("\n").startswith('CONECT'):
                        top_out.write(line)
        # Load simulation into Universe
        u = MDAnalysis.Universe(F'{path}/{input_pdb[:-4]}_noconect.pdb',
                                F'{path}/{input_dcd}',in_memory=True)
    # Generate topology
    u = add_topology_bonds(u, list(components.N), list(components.N_res))

    shutil.copy(F'./{name}_top_stripped.pdb', F'{path}/top_stripped.pdb')
    ref = MDAnalysis.Universe(F'{path}/top_stripped.pdb',F'{path}/traj_protein.dcd',in_memory=True)
    n_frames = len(u.trajectory[start:end:step])
    # Atom selection protein in unaligned trajectory.
    ag = u.select_atoms("not name J")
    n_atoms = ag.n_atoms
    # Atom selection protein in aligned trajectory.
    ag_ref  = ref.select_atoms("not name J")
    n_atoms_ref = ag_ref.n_atoms
    if n_atoms != n_atoms_ref:
        print("Trajectories don't match: Unequal amount of proteins.")
        return None
    elif n_frames != len(ref.trajectory[start:end:step]):
        print("Trajectories don't match: Unequal amount of frames.")
        return None
    # Atom selection protein in unaligned trajectory.
    ap   = u.select_atoms("name J")
    full = u.select_atoms("all")
    n_atoms_peg = ap.n_atoms
    print('n_atoms: ',n_atoms)
    print(u.dimensions)
    # L = u.dimensions[0]/10
    lz = u.dimensions[2]
    edges = np.arange(0,lz+1,1)
    dz = (edges[1] - edges[0]) / 2.
    z = edges[:-1] + dz
    n_bins = len(z)
    # hs = np.zeros((n_frames,n_bins))
    hsp = np.zeros((n_frames,n_bins))
    translation_factors = np.zeros((n_frames,1))

    # Get Translation factors.
    for t in range(n_frames):
        # Determine translation factor
        zpos_translations = np.unique(np.round(ag_ref.positions.T[2] - ag.positions.T[2], 2))
        #
        f_wrap = lambda x: x-lz if x>0.5*lz else (x+lz if x<-0.5*lz else x)
        for i,zpos in enumerate(zpos_translations):
            zpos_translations[i] = np.round(f_wrap(zpos),2)
        if len(np.unique(zpos_translations))==1:
            translation_factors[t] = np.round(np.unique(zpos_translations),2)
        elif len(np.unique(zpos_translations))>=2:
            translation_factors[t] = np.round(np.mean(np.unique(zpos_translations)),2)

        if translation_factors[t] == 0:
            print(np.unique(zpos_translations))

        if t<=n_frames-2:
            # Move to the next frame.
            u.trajectory.next()
            ref.trajectory.next()
    translation_factors = translation_factors.flatten()
    print(translation_factors)
    np.save(f'{path}/{name:s}_{temp:d}_{number}_translation_factors.npy',translation_factors,allow_pickle=False)
    os.remove(F'{path}/traj_protein.dcd')

    u.trajectory.rewind()
    # Align the all components and write to file.
    # Extract the PEG profile.
    with MDAnalysis.Writer(path+'/traj_aligned.dcd',full.n_atoms) as W:
        for t,ts in enumerate(u.trajectory[start:end:step]):
            full.translate(np.array([0,0,translation_factors[t]]))#(ts)
            ts = transformations.wrap(full)(ts)
            W.write(full)

            # PEG z-positions.
            zposp = ap.positions.T[2]
            hp, ep = np.histogram(zposp,bins=edges)
            hsp[t] = hp
    np.save(f'{path}/{name:s}_{temp:d}_{number}_peg.npy',hsp,allow_pickle=False)


if __name__=="__main__":
    parser = ArgumentParser()
    parser.add_argument('--protein',type=str,required=True)
    parser.add_argument('--peg',type=str,required=True)
    parser.add_argument('--number',type=int,required=True)
    parser.add_argument('--temp',type=int,required=True)
    args = parser.parse_args()

    temp = args.temp

    residues = pd.read_csv('residues.csv').set_index('one',drop=False)
    proteins = pd.read_pickle('proteins.pkl')

    t0 = time.time()
    system = F"{args.protein} {args.peg}"
    for numbers in [[100,args.number]]:#numbers_dict[system]:
        print(system, numbers)
        if os.path.isfile(F"{args.protein}-{args.peg}/{temp}/{numbers[-1]}/t.dcd"):
            composition = pd.DataFrame(index=system.split(' '),
                                       columns=['N','fasta','N_res'])
            composition.N = [int(N) for N in numbers]
            composition.fasta = [proteins.loc[name].fasta for name in composition.index]
            composition.N_res = [proteins.loc[name].N_res for name in composition.index]

            print("Aligning the protein slab:")
            center_slab(name=system.replace(' ','-'),
                        number=numbers[-1],
                        components=composition,
                        temp=temp,
                        start=None,end=None,
                        step=1,
                        input_pdb='top.pdb')

            print("Aligning PEG to the protein slab:")
            if numbers[-1]>0:
                align_peg(name=system.replace(' ','-'),
                          number=numbers[-1],
                          components=composition,
                          temp=temp,
                          start=None,end=None,
                          step=1,
                          input_pdb='top.pdb')
        else: print(F"{system} was not simulated with {numbers[-1]} chains of PEG.")
    print(F'Timing {time.time()-t0:5.3f}')
