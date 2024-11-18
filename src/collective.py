import numpy as np
from multiprocessing import Pool
from time import time
import subprocess as sp
import os

from molecule import Molecule
from el_structure import do_GS_calc, do_ES_calc

class Collective():
    def __init__(self, num_mol=10, num_steps=100, time_step=0.25):

        print("Doing %d MD steps with a time step of %1.4f fs" % (num_steps, time_step))

        # Parallelization Information
        self.num_procs = 96

        # Dynamics Information
        self.num_steps = num_steps
        self.time_step = time_step * 41.341 # fs --> a.u.

        self.do_Langevin     = False
        self.langevin_lambda = 50.0

        self.doRescale       = False
        self.rescale_freq    = 20

        self.do_MB_dist      = False

        self.temperature     = 300 # K

        # Clean output files
        self.clean_output_files()

        # Initialize Molecules (do first electronic structure calculations)
        self.num_mol   = num_mol
        self.molecules = [ Molecule(mol_number=i) for i in range(num_mol) ]


    def clean_output_files(self):
        self.output_dir = "OUTPUT"
        if ( os.path.exists(self.output_dir) ):
            sp.call(f"rm -r {self.output_dir}", shell=True)
        os.mkdir( self.output_dir )

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

            with Pool(self.num_procs) as pool:
                for molecule in self.molecules:
                    pool.apply_async( do_GS_calc(molecule) )
                pool.close()
                pool.join()

            with Pool(self.num_procs) as pool:
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
        self.save_dipole_matrix(cavity)
        self.save_polariton_wavefunctions(cavity)
        self.save_molecular_absorption_spectra()
        #self.save_cavity_molecular_absorption_spectra(cavity) # TODO
    
    def save_molecular_absorption_spectra(self):
        # Calculate absorption spectra from oscillator strengths
        # f_0j = (2/3) E_0j (mu_0j . mu_0j)
        # Do not include cavity polarization here. This is just the molecular absorption spectra (outside cavity).
        osc_str = np.zeros( (self.num_mol, self.molecules[0].n_ES_states) )            
        for moli,molecule in enumerate(self.molecules):
            E0j  = molecule.adiabatic_energy[1:] - molecule.adiabatic_energy[0]
            mu0j = molecule.dipole_matrix[0,1:,:]
            osc_str[moli,:] = (2/3) * E0j * np.einsum("sd,sd->s", mu0j, mu0j)
        np.save("%s/molecular_oscillator_strengths__step_%d.dat" % (self.output_dir, self.step), osc_str) # (NMOL, NSTATES)
        
        EGRID = np.linspace(0,1,10000) # a.u.
        SIG   = 0.001/27.2114 # a.u.
        ABS_G = np.zeros( (len(EGRID)) )
        ABS_L = np.zeros( (len(EGRID)) )
        for pt in range( len(EGRID) ):
            for moli,molecule in enumerate(self.molecules):
                E0j  = molecule.adiabatic_energy[1:] - molecule.adiabatic_energy[0]
                ABS_G[pt] += np.sum( osc_str[moli,:] * np.exp( -1 * (EGRID[pt] - E0j[:]) ** 2 / (2 * SIG**2) ) )
                ABS_L[pt] += np.sum( osc_str[moli,:] * 1 / ( 1 + ((EGRID[pt] - E0j[:])/SIG)**2 ) )
        ABS_G[:] *= 1 / np.sqrt(2*np.pi) / SIG # Normalization of the Gaussian
        ABS_L[:] *= 1 / np.pi / SIG # Normalization of the Lorentzian
        np.savetxt("%s/molecular_absorption_spectra_Gauss_Loren_step_%d.dat" % (self.output_dir, self.step), np.c_[EGRID*27.2114,ABS_G,ABS_L], header="Energy(eV) Gaussian Lorentzian")

    def save_cavity_molecular_absorption_spectra(self,cavity):
        # Calculate absorption spectra from oscillator strengths of the polariton states
        # f_0J = (2/3) E_0J (mu_0J * mu_0J)
        # Include cavity polarization here.
        POL_WFN    = cavity.polariton_wavefunctions
        NS         = self.molecules[0].n_ES_states
        NMOL       = self.num_mol
        NPOL       = len(POL_WFN)
        MOL_DIP    = np.array([ self.molecules[i].dipole_matrix for moli in range(NMOL)])
        MOL_DIP    = np.einsum("Aijd,dM->AijM", MOL_DIP, cavity.cavity_polarization) # (NMOL, NSTATES, 3)
        POL_DIPOLE = np.zeros( (NPOL,NPOL) )

        # # Write the dipole matrix in the Hamiltonian structure
        for moli,molecule in enumerate(collective.molecules):
            for state in range(NS):
                POL_DIPOLE[moli*NS + state, moli*NS + state] = 





        # for moli,molecule in enumerate(self.molecules):
        #     E0j  = molecule.adiabatic_energy[1:] - molecule.adiabatic_energy[0]
        #     mu0j = molecule.dipole_matrix[0,1:,:]
        #     osc_str[moli,:] = (2/3) * E0j * np.einsum("sd,sd->s", mu0j, mu0j)
        # np.save("%s/molecular_oscillator_strengths__step_%d.dat" % (self.output_dir, self.step), osc_str) # (NMOL, NSTATES)
        
        # EGRID = np.linspace(0,1,10000) # a.u.
        # SIG   = 0.001/27.2114 # a.u.
        # ABS_G = np.zeros( (len(EGRID)) )
        # ABS_L = np.zeros( (len(EGRID)) )
        # for pt in range( len(EGRID) ):
        #     for moli,molecule in enumerate(self.molecules):
        #         E0j  = molecule.adiabatic_energy[1:] - molecule.adiabatic_energy[0]
        #         ABS_G[pt] += np.sum( osc_str[moli,:] * np.exp( -1 * (EGRID[pt] - E0j[:]) ** 2 / (2 * SIG**2) ) )
        #         ABS_L[pt] += np.sum( osc_str[moli,:] * 1 / ( 1 + ((EGRID[pt] - E0j[:])/SIG)**2 ) )
        # ABS_G[:] *= 1 / np.sqrt(2*np.pi) / SIG # Normalization of the Gaussian
        # ABS_L[:] *= 1 / np.pi / SIG # Normalization of the Lorentzian
        # np.savetxt("%s/molecular_absorption_spectra_Gauss_Loren_step_%d.dat" % (self.output_dir, self.step), np.c_[EGRID*27.2114,ABS_G,ABS_L], header="Energy(eV) Gaussian Lorentzian")



    def save_photon_per_eigenstate(self, cavity):
        # Append number of photons in each eigenstate to photons.dat
        with open("%s/photon_number.dat" % self.output_dir, "a") as f:
            f.write("%1.4f" % (self.step * self.time_step / 41.341))
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
        T = (2/3) * KE / (self.num_mol * self.molecules[0].natoms) * (300 / 0.025 * 27.2114) # a.u. --> K
        with open("%s/temperature.dat" % self.output_dir, "a") as f:
            f.write("%1.4f %1.4f\n" % (self.step * self.time_step / 41.341, T))

    def save_XYZ(self):
        # Append XYZ coordinates of all molecules to XYZ trajectory.xyz
        with open("%s/trajectory.xyz" % self.output_dir, "a") as f:
            f.write("%d\n" % sum([molecule.natoms for molecule in self.molecules]))
            f.write("Step %d\n" % (self.step))
            for molecule in self.molecules:
                shift = molecule.mol_number * 10 
                coords = molecule.R * 0.529 # Angstrom
                coords[:,0] += shift # Put molecules in a line in cavity
                for at in range(molecule.natoms):
                    f.write("%s %1.4f %1.4f %1.4f\n" % (molecule.LABELS[at], coords[at,0], coords[at,1], coords[at,2]) )

    def save_energy(self, cavity):
        # Append energies of all molecules to energy_mol.dat
        E_GS = 0.0
        for molecule in self.molecules:
            E_GS += molecule.GS_ENERGY
        with open("%s/energy_mol.dat" % self.output_dir, "a") as f:
            f.write("%1.4f" % (self.step * self.time_step / 41.341))
            for molecule in self.molecules:
                for state in range(molecule.n_ES_states+1):
                    f.write(" %1.8f" % (molecule.adiabatic_energy[state]) )
            f.write("\n")
        
        # Append polariton energies to energy_polariton.dat
        with open("%s/energy_polariton.dat" % self.output_dir, "a") as f:
            f.write("%1.4f" % (self.step * self.time_step / 41.341))
            for energy in cavity.polariton_energy:
                f.write(" %1.8f" % energy)
            f.write("\n")

    def save_dipole_matrix(self,cavity):
        # Save the dipole matrix as a .npy file for each time-step
        dipole_matrix = np.zeros( (self.num_mol, self.molecules[0].n_ES_states+1, self.molecules[0].n_ES_states+1,3) )
        dipole_matrix_projected = np.zeros( (self.num_mol, self.molecules[0].n_ES_states+1, self.molecules[0].n_ES_states+1, cavity.num_modes) )
        for moli,molecule in enumerate( self.molecules ):
            dipole_matrix[moli] = molecule.dipole_matrix
            for mode in range(cavity.num_modes):
                dipole_matrix_projected[moli,:,:,mode] = np.einsum("abx,x->ab", molecule.dipole_matrix, cavity.cavity_polarization[mode])
        np.save("%s/dipole_matrix__step_%d.npy" % (self.output_dir,self.step), dipole_matrix)
        np.save("%s/dipole_matrix_projected__step_%d.npy" % (self.output_dir,self.step), dipole_matrix_projected)
    
    def save_polariton_wavefunctions(self, cavity):
        # Save the polariton wavefunctions as a .npy file for each time-step
        np.save("%s/polariton_wavefunctions__step_%d.npy" % (self.output_dir,self.step), cavity.polariton_wavefunctions)