import numpy as np

class Cavity():
    def __init__(self, num_modes=1):
        
        # Electronic Structure Variables
        self.num_modes           = 1
        self.cavity_freq         = np.array([0.25]) # a.u.
        self.cavity_coupling     = np.array([0.05]) # a.u.
        self.cavity_polarization = np.array([1.0, 1.0, 1.0]) # a.u.
        self.cavity_polarization = self.cavity_polarization / np.linalg.norm(self.cavity_polarization)
    
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
                    polarized_dipole = np.dot( molecule.dipole_matrix[0,state,:], self.cavity_polarization )
                    phase = np.exp( 1j * molecule.mol_number * mode )
                    self.H_cavity[moli*NS + state, NMOL*NS + mode] = gc[mode] * polarized_dipole * phase
                    self.H_cavity[NMOL*NS + mode, moli*NS + state] = self.H_cavity[moli*NS + state, NMOL*NS + mode].conj()

        E, U = np.linalg.eigh(self.H_cavity)
        self.polariton_energy = E.real
        self.polariton_wavefunctions = U

        #print( np.round(self.H_cavity,2) )