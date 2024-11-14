import numpy as np
from random import gauss

from el_structure import do_GS_calc, do_ES_calc
from read_input import get_GEOM

class Molecule():
    def __init__(self, mol_number=None):
        
        # For handling many-molecule simulations
        assert(mol_number is not None and mol_number >= 0), "mol_number must be a positive integer" # Check if mol_number is valid
        self.mol_number = mol_number
        
        # Electronic Structure Variables
        self.basis_set        = "sto3g"
        self.n_ES_states      = 2
        self.ES_type          = 'SD' # 'TDA'
        self.do_TDA_gradients = False 
        self.xc               = None # "pbe,pbe" -- None means to do HF

        # Initialize the molecule
        self.__build()

    def __build(self):
        get_GEOM(self)
        do_GS_calc(self)
        do_ES_calc(self)
    
    def propagate_nuclear_R(self, params, dt=None):
        #if ( hasattr( self, "F_NEW" ) ):
        #    self.F_OLD    = self.F_NEW * 1.0
        self.F_NEW    = -1 * self.GS_GRADIENT
        
        if ( params.do_Langevin == True ):
            self.do_Langevin_XStep(params, dt)
        else:
            self.R       += self.V * dt + 0.5 * dt**2 * self.F_NEW / self.masses[:,None]

    def propagate_nuclear_V(self, params, dt=None):
        if ( not hasattr( self, "F_OLD" ) ):
            self.F_OLD = self.F_NEW * 1.0
        self.F_NEW    = -1 * self.GS_GRADIENT

        if ( params.do_Langevin == True ):
            self.do_Langevin_VStep(params, dt)
        else:
            self.V       += 0.5 * dt * (self.F_OLD + self.F_NEW) / self.masses[:,None]
    
        if ( params.doRescale ):
            if ( params.step % params.rescale_freq == 0 or params.step == 0 or params.step == 1 ):
                self.rescale_V(params)
    
    def rescale_V(self, params):
        KE = 0.5000 * np.einsum("R,Rd,Rd->", self.masses, self.V, self.V )
        T  = (2/3) * KE / self.natoms * (300 / 0.025) # eV --> K
        print( "KE, T",  KE, T )
        if ( KE > 1e-6 ):
            self.V *= np.sqrt( params.temperature / T )
        else:
            print("KE too small to rescale V")
            # Sample Boltzmann distribution at temperature params.temperature
            self.V = np.array([gauss(0,params.temperature * (0.025 / 300 / 27.2114)) for dof in range(3*self.natoms)], dtype=float).reshape((self.natoms,3)) # Gaussian random number

    def do_el_structure(self):
        do_GS_calc(self)
        do_ES_calc(self)





    def do_Langevin_XStep(self,params,dt=None):
        """
        Mark Tuckerman -- Stat. Mech.: Theory and Mol. Simulation
        Chapter 15.5 Page 594 Eq. 15.5.18
        """
        TEMP  = params.temperature * (0.025 / 300) / 27.2114 # K -> KT (a.u.)

        self.langevin_epsilon = np.array([gauss(0,1) for dof in range(3*self.natoms)], dtype=float).reshape((self.natoms,3)) # Gaussian random number
        self.langevin_theta   = np.array([gauss(0,1) for dof in range(3*self.natoms)], dtype=float).reshape((self.natoms,3)) # Gaussian random number

        # Difference in acceleration and damped velocity
        a_ORIG = self.F_NEW / self.masses[:,None]  # Original acceleration
        a_DAMP = params.langevin_lambda * self.V # Acceleration due to damping
        SIGMA = np.sqrt(2 * TEMP * params.langevin_lambda / self.masses[:,None]) # Gaussian Width
        RANDOM_FAC = 0.5 * self.langevin_epsilon + 1/(2*np.sqrt(3)) * self.langevin_theta

        # A(t) has units of position
        # Store this for VStep
        self.langevin_A = 0.5 * dt**2 * (a_ORIG - a_DAMP) + SIGMA * dt**(3/2) * RANDOM_FAC
        self.R += dt * self.V + self.langevin_A

    def do_Langevin_VStep(self,params,dt=None):
        """
        Mark Tuckerman -- Stat. Mech.: Theory and Mol. Simulation
        Chapter 15.5 Page 594 Eq. 15.5.18
        """
        TEMP  = params.temperature * (0.025 / 300) / 27.2114 # K -> KT (a.u.) # TODO Change units in read_input.py

        SIGMA = np.sqrt(2 * TEMP * params.langevin_lambda / self.masses[:,None]) # Gaussian Width

        self.V += 0.5000000 * dt * ( self.F_OLD + self.F_NEW ) / self.masses[:,None] - \
                  dt * params.langevin_lambda  * self.V + SIGMA * np.sqrt(dt) * self.V - \
                  params.langevin_lambda * self.langevin_A
