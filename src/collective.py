import numpy as np
from time import time, sleep
import subprocess as sp
import os

from multiprocessing import Pool
#from multiprocessing import Process
#from pathos.multiprocessing import ProcessingPool as Pool


from molecule import Molecule
from el_structure import do_GS_calc, do_ES_calc

class Collective():
    def __init__(self, num_mol=10, num_steps=100, time_step=0.25):

        print("Doing %d MD steps with a time step of %1.4f fs" % (num_steps, time_step))

        # Parallelization Information
        self.num_procs = 36

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

        # Do the first electronic structure calculations
        self.do_el_structure()


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
                self.molecules = pool.map(do_GS_calc, self.molecules)
                self.molecules = pool.map(do_ES_calc, self.molecules)
        else:
            for molecule in self.molecules:
                molecule = do_GS_calc( molecule )
                molecule = do_ES_calc( molecule )

        T1 = time()
        print("Time for Electronic Structure Calculation = %1.4f" % (T1-T0))




