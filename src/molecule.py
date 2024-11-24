import numpy as np
from random import gauss

from el_structure import do_GS_calc, do_ES_calc
from read_input import get_GEOM, get_VELOC

class Molecule():
    def __init__(self, mol_number=None):
        
        # For handling many-molecule simulations
        self.mol_number = mol_number
        
        # Electronic Structure Variables
        self.basis_set        = "sto5g"
        self.n_ES_states      = 1
        self.ES_type          = 'TDA' # 'TDA' -- Tamm-Dancoff Approx. or 'RPA' -- Random Phase Approx. or 'SD' -- Slater Determinant
        self.do_TDA_gradients = False 
        self.xc               = "PBE" # "" -- Do LDA, 'MINDO3' -- Do semi-empirical MINDO3, "PBE0" -- Do PBE1PBE/PBE0 "pbe,pbe" -- Do PBE, None -- do HF

        # Initialize the molecule
        self.__build()

    def __build(self):
        get_GEOM(self)
        get_VELOC(self)
        #do_GS_calc(self)
        #do_ES_calc(self)
    
    def propagate_nuclear_R(self, params, dt=None):
        self.F_NEW    = -1 * self.GS_GRADIENT
        if ( params.do_Langevin == True ):
            self.do_Langevin_XStep(params, dt)
        else:
            self.R       += self.V * dt + 0.5 * dt**2 * self.F_NEW / self.masses[:,None]

    def propagate_nuclear_V(self, params, dt=None):
        self.F_OLD = self.F_NEW * 1.0
        self.F_NEW = -1 * self.GS_GRADIENT

        # Check if F_NEW or F_OLD is zero
        if ( np.all( self.F_NEW == 0 ) or np.all( self.F_OLD == 0 ) ):
            print("F_NEW or F_OLD is zero. Exiting.")
            exit()

        if ( params.step == 1 and params.do_MB_dist == True ):
            self.MaxwellBoltzmann(params)

        if ( params.do_Langevin == True ):
            self.do_Langevin_VStep(params, dt)
        else:
            self.V += 0.5 * dt * (self.F_OLD + self.F_NEW) / self.masses[:,None]
    
        if ( params.doRescale ):
            if ( params.step % params.rescale_freq == 0 or params.step == 0 or params.step == 1 ):
                self.rescale_V(params)
    
    def rescale_V(self, params):
        KE = 0.5000 * np.einsum("R,Rd,Rd->", self.masses, self.V, self.V )
        T  = (2/3) * KE / self.natoms * (300 / 0.025 * 27.2114) # a.u. --> K
        print( "Before: T = %1.4f" %  T )
        if ( KE > 1e-6 ):
            self.V *= np.sqrt( params.temperature / T )
        else:
            print("KE too small to rescale V. Sampling MB distribution.")
            # Sample Boltzmann distribution at temperature params.temperature
            self.MaxwellBoltzmann(params)
        
        # Check final temperature
        KE = 0.5000 * np.einsum("R,Rd,Rd->", self.masses, self.V, self.V )
        T  = (2/3) * KE / self.natoms * (300 / 0.025 * 27.2114) # eV --> K
        print( "After: T = %1.4f" % T )

    def MaxwellBoltzmann(self,params):
        KbT  = params.temperature * (0.025 / 300) / 27.2114 # K -> KT (a.u.)

        for at in range(self.natoms):
            alpha = self.masses[at] / KbT
            sigma = np.sqrt(1 / alpha)
            self.V[at,:] = np.array([gauss(0,sigma) for dof in range(3)], dtype=float)
        

    def do_el_structure(self):
        print("Doing Electronic Structure Calculations for Molecule %d" % self.mol_number)
        do_GS_calc(self)
        do_ES_calc(self)





    def do_Langevin_XStep(self,params,dt=None):
        """
        Mark Tuckerman -- Stat. Mech.: Theory and Mol. Simulation
        Chapter 15.5 Page 594 Eq. 15.5.18
        """
        TEMP  = params.temperature * (0.025 / 300) / 27.2114 # K -> KT (a.u.)

        self.langevin_RAND1 = np.array([gauss(0,1) for dof in range(3*self.natoms)], dtype=float).reshape((self.natoms,3)) # Gaussian random number
        self.langevin_RAND2   = np.array([gauss(0,1) for dof in range(3*self.natoms)], dtype=float).reshape((self.natoms,3)) # Gaussian random number

        # Difference in acceleration and damped velocity
        a_ORIG = self.F_NEW / self.masses[:,None]  # Original acceleration
        a_DAMP = params.langevin_lambda/1000/27.2114 * self.V # Acceleration due to damping
        SIGMA = np.sqrt(2 * TEMP * params.langevin_lambda/1000/27.2114 / self.masses[:,None]) # Gaussian Width
        RANDOM_FAC = 0.5 * self.langevin_RAND1 + 1/(2*np.sqrt(3)) * self.langevin_RAND2

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

        SIGMA = np.sqrt(2 * TEMP * params.langevin_lambda/1000/27.2114 / self.masses[:,None]) # Gaussian Width

        self.V += 0.5000000 * dt * ( self.F_OLD + self.F_NEW ) / self.masses[:,None] \
                    - dt * params.langevin_lambda/1000/27.2114  * self.V \
                    + SIGMA * np.sqrt(dt) * self.langevin_RAND1 \
                    - params.langevin_lambda/1000/27.2114 * self.langevin_A
