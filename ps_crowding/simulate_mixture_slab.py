from simtk import openmm, unit
from simtk.openmm import app
from simtk.openmm import XmlSerializer
from analyse_mixture import *
import time
import os
import sys
from argparse import ArgumentParser
import random
import numpy as np
import mdtraj as md



def simulate(residues,dirname,proteins,composition,temp,ionic,cutoff,boxlength):
    residues = residues.set_index('one')

    composition, yukawa_kappa, lj_eps = genParams(residues,proteins,composition,temp,ionic*1e-3)

    # Set parameters
    L = boxlength 
    margin = 4
    N_chains = composition.N.sum()
    path = F"{dirname:s}/{temp:d}/{composition.iloc[1].N:d}"
    print(path)
    
    # As taken from the single chain simulations
    N_res = len(composition.fasta.iloc[0])
    N_save = int(np.ceil(3e-4*N_res**2)*1000)

    system = openmm.System()

#    N = len(fasta)

#     # set parameters
#     L = 15.
#    margin = 2
#    if N > 350:
#        L = 25.
#        Lz = 300.
#        margin = 8
#        Nsteps = int(2e7)
#    elif N > 200:
#        L = 17.
#        Lz = 300.
#        margin = 4
#        Nsteps = int(6e7)
#    else:
#    Lz = 10*L

    # Set box vectors
    a = unit.Quantity(np.zeros([3]), unit.nanometers)
    a[0] = L * unit.nanometers
    b = unit.Quantity(np.zeros([3]), unit.nanometers)
    b[1] = L * unit.nanometers
    c = unit.Quantity(np.zeros([3]), unit.nanometers)
    Lz = 10*L
    c[2] = Lz * unit.nanometers
    system.setDefaultPeriodicBoxVectors(a, b, c)
    
    # Initial configuration
    z = np.empty(0)
    z = np.append(z,np.random.rand(1)*(L-margin)-(L-margin)/2)
    for z_i in np.random.rand(N_chains)*(L-margin)-(L-margin)/2:
        z_j = z_i-L if z_i>0 else z_i+L
        if np.all(np.abs(z_i-z_j)>.7):
            z = np.append(z,z_i)
        if z.size == N_chains:
            break
    ##### INSERTION TO SEPARATE MIXTURE COMPONENTS AT START #####
    z_off = np.zeros(z.size)
    z_off[composition.loc[composition.index[0]].N:]+=Lz/2
    z += z_off
    #############################################################

    xy = np.random.rand(N_chains,2)*(L-margin)-(L-margin)/2

    print('Number of chains',z.size)
    # Set up the topology of the system.
    top = md.Topology()
    N_beads = (composition.fasta.apply(lambda x : len(x))*composition.N).values.sum()
    pos = np.empty((N_beads,3))

    start = 0
    begin = 0
    for k,name in enumerate(composition.index):
        N = composition.loc[name].N
        Naa = len(composition.loc[name].fasta)
        for z_i,(x_i,y_i) in zip(z[start:start+N],xy[start:start+N]):
            chain = top.add_chain()
            pos[begin:begin+Naa,:] = xy_spiral_array(Naa,L/2.) + np.array([x_i,y_i,z_i])
            for resname in composition.loc[name].fasta_termini:
                residue = top.add_residue(resname, chain)
                top.add_atom(resname, element=md.element.carbon, residue=residue)
            for i in range(chain.n_atoms-1):
                top.add_bond(chain.atom(i),chain.atom(i+1))
            begin += Naa
        start += N
    # Write the topology to a PDB file.
    md.Trajectory(np.array(pos).reshape(N_beads,3), top, 0,
                  [L,L,Lz], [90,90,90]).save_pdb(path+'/top.pdb')

    pdb = app.pdbfile.PDBFile(path+'/top.pdb')
    print(pdb)

    for name in composition.index: 
        fasta = composition.loc[name].fasta
        for _ in range(composition.loc[name].N):
            system.addParticle((residues.loc[fasta[0]].MW+2)*unit.amu)
            for a in fasta[1:-1]:
                system.addParticle(residues.loc[a].MW*unit.amu) 
            system.addParticle((residues.loc[fasta[-1]].MW+16)*unit.amu)

    hb = openmm.openmm.HarmonicBondForce()
    energy_expression = 'select(step(r-2^(1/6)*s),4*eps*l*((s/r)^12-(s/r)^6-shift),4*eps*((s/r)^12-(s/r)^6-l*shift)+eps*(1-l))'
    ah = openmm.openmm.CustomNonbondedForce(energy_expression+'; s=0.5*(s1+s2); l=select(step(m1*m2),0.5*(l1+l2),lambda_cross); shift=(0.5*(s1+s2)/rc)^12-(0.5*(s1+s2)/rc)^6')

    ah.addGlobalParameter('eps',lj_eps*unit.kilojoules_per_mole)
    ah.addGlobalParameter('rc',cutoff*unit.nanometer)
    ah.addGlobalParameter('lambda_cross',residues.loc['J'].lambdas*unit.dimensionless)
    ah.addPerParticleParameter('s')
    ah.addPerParticleParameter('l')
    ah.addPerParticleParameter('m')
    
    print('rc',cutoff*unit.nanometer)
 
    #yu = openmm.openmm.CustomNonbondedForce('q*(exp(-kappa*r)/r-shift1-(r-4)*shift2); q=q1*q2')
    yu = openmm.openmm.CustomNonbondedForce('q*(exp(-kappa*r)/r-shift1); q=q1*q2')
    yu.addGlobalParameter('kappa',yukawa_kappa/unit.nanometer)
    yu.addGlobalParameter('shift1',np.exp(-yukawa_kappa*4.0)/4.0/unit.nanometer)
    #yu.addGlobalParameter('shift2',-(yukawa_kappa*4.0+1)*np.exp(-yukawa_kappa*4.0)/(4.0*unit.nanometer)**2)
    yu.addPerParticleParameter('q')

    begin = 0
    for name in composition.index:
        fasta = composition.loc[name].fasta
        Naa = len(fasta)
        for _ in range(composition.loc[name].N):
            for a,e in zip(fasta,composition.loc[name].charge):
                yu.addParticle([e*unit.nanometer*unit.kilojoules_per_mole])
                ah.addParticle([residues.loc[a].sigmas*unit.nanometer,
                                residues.loc[a].lambdas*unit.dimensionless,
                                residues.loc[a].m*unit.dimensionless])
            for i in range(begin,begin+Naa-1):
                if name[:3] == 'PEG':
                    hb.addBond(i, i+1, 0.33*unit.nanometer, 8033*unit.kilojoules_per_mole/(unit.nanometer**2))
                else:
                    hb.addBond(i, i+1, 0.38*unit.nanometer, 8033*unit.kilojoules_per_mole/(unit.nanometer**2))
                yu.addExclusion(i, i+1)
                ah.addExclusion(i, i+1)
            begin += Naa

    print(begin,N_beads)

    yu.setForceGroup(0)
    ah.setForceGroup(1)
    yu.setNonbondedMethod(openmm.openmm.CustomNonbondedForce.CutoffPeriodic)
    ah.setNonbondedMethod(openmm.openmm.CustomNonbondedForce.CutoffPeriodic)
    hb.setUsesPeriodicBoundaryConditions(True)
    # Set Yukawa and Ashbaugh cut-offs.
    yu.setCutoffDistance(4*unit.nanometer)
    ah.setCutoffDistance(cutoff*unit.nanometer)

    #ah.setUseSwitchingFunction(True)
    #ah.setSwitchingDistance(0.9*cutoff*unit.nanometer)

    system.addForce(hb)
    system.addForce(yu)
    system.addForce(ah)
    
    # Set integrator settings.
    integrator = openmm.openmm.LangevinIntegrator(temp*unit.kelvin,
                                                  0.01/unit.picosecond,
                                                  0.01*unit.picosecond)
    # Set the hardware to run on.
    platform = openmm.Platform.getPlatformByName('CUDA')

    simulation = app.simulation.Simulation(pdb.topology, system, integrator,
                                           platform, dict(CudaPrecision='mixed'))

    check_point = path+'/restart.chk'.format(temp)

    # Check if the simulation is new or a continuation of a previous one.
    if os.path.isfile(check_point):
        print('Reading check point file')
        simulation.loadCheckpoint(check_point)
        simulation.reporters.append(app.dcdreporter.DCDReporter(path+'/t.dcd'.format(name),
                                                                int(5e4),enforcePeriodicBox=True,
                                                                append=True))
    else:
        simulation.context.setPositions(pdb.positions)
        simulation.minimizeEnergy()
        simulation.reporters.append(app.dcdreporter.DCDReporter(path+'/t.dcd'.format(name),
                                                                int(5e4),enforcePeriodicBox=True))

    simulation.reporters.append(app.checkpointreporter.CheckpointReporter(check_point,reportInterval=N_save)) 
    simulation.reporters.append(app.statedatareporter.StateDataReporter(path+'/log',100000,
             step=True,speed=True,elapsedTime=True,separator='\t'))

    print("Starting the simulation.")
    simulation.runForClockTime(71.5*unit.hour,checkpointFile=check_point,checkpointInterval=0.5*unit.hour)

    # print("No. Steps to Run: ",50000*N_save)
    # Run rounds of simulations to allow for extra checkpoints.
    # simulation.step(50000*N_save)
    


if __name__=="__main__":
    parser = ArgumentParser()
    parser.add_argument('--proteins',nargs='+',required=True)
    parser.add_argument('--numbers',nargs='+',required=True)
    parser.add_argument('--temp',nargs='?',const='',type=int,required=True)
    parser.add_argument('--dirname',nargs='?',const='',type=str,required=True)
    parser.add_argument('--ionic',nargs='?',const='',type=int,required=True)
    parser.add_argument('--cutoff',nargs='?',const='', type=float)
    parser.add_argument('--boxlength',nargs='?',const='', type=float)
    args = parser.parse_args()
    
    
    residues = pd.read_csv('residues.csv').set_index('three',drop=False)
    proteins = pd.read_pickle('proteins.pkl')
    
    composition = pd.DataFrame(index=args.proteins,columns=['N','fasta'])
    composition.N = [int(N) for N in args.numbers]
    composition.fasta = [proteins.loc[name].fasta for name in composition.index]
    
    t0 = time.time()
    simulate(residues,args.dirname,proteins,composition,args.temp,args.ionic,args.cutoff,args.boxlength)
    print('Timing {:5.3f}'.format(time.time()-t0))
