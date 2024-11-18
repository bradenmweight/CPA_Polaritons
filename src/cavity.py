import numpy as np

class Cavity():
    def __init__(self,myTRAJ):

        #En0 = self.get_average_molecular_excitation_energy(myTRAJ, n=1)
        #detuning = 0.0 # a.u. -- Cavity detuning from the molecular excitation energy
        #wc = En0 - detuning # a.u. -- Fundamental frequency of the cavity
        
        # Cavity Variables
        num_mol                  = myTRAJ.num_mol
        self.num_modes           = 21 # Choose odd number of modes
        self.get_FP_cavity( wc=0.25, Emax=0.3 ) # CO2 -- TDA
        self.cavity_coupling     = np.array([5.0]*self.num_modes) / np.sqrt(myTRAJ.num_mol) / np.sqrt(self.num_modes) # a.u.


    def get_average_molecular_excitation_energy(self, myTRAJ, n=1):
        # Get average molecular excitation energy S0 --> Sn
        num_mol = myTRAJ.num_mol
        En0 = 0.0
        for molecule in myTRAJ.molecules:
            En0 += molecule.adiabatic_energy[n] - molecule.adiabatic_energy[0]
        En0 /= num_mol
        return En0

    def get_FP_cavity( self, Emax=None, wc=None ):
        Thetamax            = np.atan( (Emax/wc)**2 - 1 )
        self.cavity_freq            = np.zeros( (self.num_modes) )
        self.cavity_polarization    = np.zeros( (self.num_modes,3) )
        #wc1 = wc * np.sqrt( 1 + np.tan(np.linspace(0,Thetamax/2,self.num_modes//2+1)) )
        #wc1 = wc * np.sqrt( 1 + np.tan(np.linspace(0,Thetamax/2,self.num_modes//2+1)) )
        #wc2 = wc * np.sqrt( 1 + np.tan(np.linspace(0,Thetamax/2,self.num_modes//2+1)) )[1:][::-1]
        #self.cavity_freq[:]         = np.append(wc2, wc1)
        self.cavity_freq[:]         = wc * np.sqrt( 1 + np.tan(np.linspace(0,Thetamax,self.num_modes)) )
        self.cavity_polarization[:] = np.array([1.0, 1.0, 1.0]*self.num_modes).reshape( (-1,3) ) # a.u.
        e_norm = np.sqrt(np.einsum("md,md->m", self.cavity_polarization, self.cavity_polarization) )
        self.cavity_polarization[:] = np.einsum("md,m->md", self.cavity_polarization,  1 / e_norm)



    def build_H_cavity(self, collective):
        # Build the cavity Hamiltonian in the first excitated subspace
        
        NMOL  = collective.num_mol
        NS    = collective.molecules[0].n_ES_states # Here I assumed all molecules have the same number of states...
        NMODE = self.num_modes

        gc    = np.sqrt(self.cavity_freq / 2) * self.cavity_coupling
        
        self.H_cavity = np.zeros( (NMOL * NS + NMODE, NMOL * NS + NMODE), dtype=np.complex128 )
        
        # Molecular Energies on the Diagonal of A block
        for moli,molecule in enumerate(collective.molecules):
            for state in range(NS):
                #print( moli*NS + state, moli*NS + state )
                self.H_cavity[moli*NS + state, moli*NS + state] = molecule.adiabatic_energy[state+1] - molecule.adiabatic_energy[0]

        # Cavity Frequencies on the Diagonal of the B block
        for mode in range(self.num_modes):        
            #print( NMOL*NS + mode, mode, NMOL*NS + mode )
            self.H_cavity[NMOL*NS + mode, NMOL*NS + mode] = self.cavity_freq[mode]
        
        # Coupling between Molecular States and Cavity Modes
        for moli,molecule in enumerate(collective.molecules):
            for state in range(NS):
                for mode in range(self.num_modes):
                    #print( moli*NS + state, mode, NMOL*NS + mode )
                    polarized_dipole = np.dot( molecule.dipole_matrix[0,state,:], self.cavity_polarization[mode] )
                    phase = np.exp( 1j * molecule.mol_number * mode )
                    self.H_cavity[moli*NS + state, NMOL*NS + mode] = gc[mode] * polarized_dipole * phase
                    self.H_cavity[NMOL*NS + mode, moli*NS + state] = self.H_cavity[moli*NS + state, NMOL*NS + mode].conj()

        E, U = np.linalg.eigh(self.H_cavity)
        self.polariton_energy = E.real
        self.polariton_wavefunctions = U

        #print( np.round(self.H_cavity,2) )