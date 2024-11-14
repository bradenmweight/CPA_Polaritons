import numpy as np
from multiprocessing import Pool
from time import time
import subprocess as sp

from molecule import Molecule
from el_structure import do_GS_calc, do_ES_calc

class Collective():
    def __init__(self, num_mol=10, num_steps=100, time_step=0.25):

        # Parallelization Information
        self.num_procs = 1

        # Dynamics Information
        self.num_steps = num_steps
        self.time_step = time_step

        self.do_Langevin     = False
        self.langevin_lambda = 10.0

        self.doRescale       = True
        self.rescale_freq    = 5
        self.temperature     = 300 # K

        # Molecule Information
        self.num_mol   = num_mol
        self.molecules = [ Molecule(mol_number=i) for i in range(num_mol) ]

        # Clean files
        sp.call("rm trajectory.xyz", shell=True)
        sp.call("rm energy_mol.dat", shell=True)
        sp.call("rm energy_polariton.dat", shell=True)
        sp.call("rm temperature.dat", shell=True)
        sp.call("rm photon_number.dat", shell=True)

    def propagate_nuclear_R(self):
        for molecule in self.molecules:
            molecule.propagate_nuclear_R(self,dt=self.time_step)

    def propagate_nuclear_V(self):
        for molecule in self.molecules:
            molecule.propagate_nuclear_V(self,dt=self.time_step)

    def do_el_structure(self):

        # Parallelize the electronic structure calculations
        T0 = time()
        if ( self.num_procs > 1 ):
            pool = Pool(processes=self.num_procs)

            for molecule in self.molecules:
                pool.apply_async( do_GS_calc(molecule) )
            pool.close()
            pool.join()

            for molecule in self.molecules:
                pool.apply_async( do_ES_calc(molecule) )
            pool.close()
            pool.join()

        else:
            for molecule in self.molecules:
                do_GS_calc( molecule )
                do_ES_calc( molecule )
                #molecule.do_el_structure()
        T1 = time()
        print("Time for Electronic Structure Calculation = %1.4f" % (T1-T0))

    def save_data(self, cavity):
        self.save_XYZ()
        self.save_energy(cavity)
        self.save_temperature()
        self.save_photon_per_eigenstate(cavity)
    
    def save_photon_per_eigenstate(self, cavity):
        # Append number of photons in each eigenstate to photons.dat
        with open("photon_number.dat", "a") as f:
            f.write("%1.4f" % (self.step * self.time_step))
            NPOL_STATES = cavity.polariton_wavefunctions.shape[1]
            for P in range(NPOL_STATES):
                cavity_start_ind = NPOL_STATES - cavity.num_modes
                f.write(" %1.4f" % np.sum( np.abs(cavity.polariton_wavefunctions[cavity_start_ind:,P])**2 ))
            f.write("\n")
    
    def save_temperature(self):
        # Append temperature to temperature.dat
        KE = 0.0
        for molecule in self.molecules:
            KE += 0.5000 * np.einsum("R,Rd,Rd->", molecule.masses, molecule.V, molecule.V )
        T = (2/3) * KE / (self.num_mol * self.molecules[0].natoms) * (300 / 0.025) # eV --> K
        with open("temperature.dat", "a") as f:
            f.write("%1.4f %1.4f\n" % (self.step * self.time_step, T))

    def save_XYZ(self):
        # Append XYZ coordinates of all molecules to XYZ trajectory.xyz
        with open("trajectory.xyz", "a") as f:
            f.write("%d\n" % sum([molecule.natoms for molecule in self.molecules]))
            f.write("Step %d\n" % (self.step))
            for molecule in self.molecules:
                for at in range(molecule.natoms):
                    f.write("%s %1.4f %1.4f %1.4f\n" % (molecule.LABELS[at], molecule.R[at,0], molecule.R[at,1], molecule.R[at,2]) )

    def save_energy(self, cavity):
        # Append energies of all molecules to energy_mol.dat
        with open("energy_mol.dat", "a") as f:
            f.write("%1.4f" % (self.step * self.time_step))
            for molecule in self.molecules:
                for state in range(molecule.n_ES_states+1):
                    f.write(" %1.4f" % (molecule.adiabatic_energy[state]))
            f.write("\n")
        
        # Append polariton energies to energy_polariton.dat
        with open("energy_polariton.dat", "a") as f:
            f.write("%1.4f" % (self.step * self.time_step))
            for energy in cavity.polariton_energy:
                f.write(" %1.4f" % energy)
            f.write("\n")
