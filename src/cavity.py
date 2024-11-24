import numpy as np

class Cavity():
    def __init__(self,myTRAJ):

        #En0 = self.get_average_molecular_excitation_energy(myTRAJ, n=1)
        #detuning = 0.0 # a.u. -- Cavity detuning from the molecular excitation energy
        #wc = En0 - detuning # a.u. -- Fundamental frequency of the cavity
        
        # Cavity Variables
        self.num_modes           = 27 # 51 # Choose odd number of modes
        self.cavity_coupling     = 0.030 / np.sqrt(myTRAJ.num_mol) #/ np.sqrt(self.num_modes) # a.u.
        self.get_FP_cavity( wc=3.0/27.2114, Emax=4/27.2114 ) # STO-6G
        #self.get_FP_cavity( wc=1.5/27.2114, Emax=2.5/27.2114 ) # STO-3G

    def get_average_molecular_excitation_energy(self, myTRAJ, n=1):
        # Get average molecular excitation energy S0 --> Sn
        num_mol = myTRAJ.num_mol
        En0 = 0.0
        for molecule in myTRAJ.molecules:
            En0 += molecule.adiabatic_energy[n] - molecule.adiabatic_energy[0]
        En0 /= num_mol
        return En0

    def get_FP_cavity( self, Emax=None, wc=None ):
        if ( self.num_modes % 2 != 1 ):
            print( "Number of modes must be odd to be symmetric and get k=0" )
            exit()
        num_modes_half         = self.num_modes #// 2 + 1
        self.cavity_coupling   = np.ones( (num_modes_half) ) * self.cavity_coupling # Assume equal coupling
        try:
            Thetamax           = np.atan( (Emax/wc)**2 - 1 )
        except AttributeError: # Depends on the version of numpy...this makes me very sad
            Thetamax           = np.arctan( (Emax/wc)**2 - 1 )
        self.cavity_freq            = np.zeros( (num_modes_half) )
        self.cavity_polarization    = np.zeros( (num_modes_half,3) )
        self.Theta                  = np.linspace(0,Thetamax,num_modes_half)
        self.cavity_freq[:]         = wc * np.sqrt( 1 + np.tan(self.Theta)**2 )
        self.cavity_polarization[:] = np.array([1.0, 1.0, 1.0]*num_modes_half).reshape( (-1,3) ) # a.u.
        e_norm                      = np.sqrt(np.einsum("md,md->m", self.cavity_polarization, self.cavity_polarization) )
        self.cavity_polarization[:] = np.einsum("md,m->md", self.cavity_polarization,  1 / e_norm)
        
        #### TODO -- For \pm k modes, we need to start the phase in build_H_cavity as a negative number for mol_index
        # self.cavity_freq            = np.concatenate( ( self.cavity_freq[1:][::-1], self.cavity_freq) )
        # self.cavity_polarization    = np.concatenate( ( self.cavity_polarization[1:][::-1], self.cavity_polarization) )
        # self.Theta                  = np.concatenate( ( -self.Theta[1:][::-1], self.Theta) )
        # self.cavity_coupling        = np.concatenate( ( -self.cavity_coupling[1:][::-1], self.cavity_coupling) )
        
    def build_H_cavity(self, collective):
        # Build the cavity Hamiltonian in the first excitated subspace
        
        # TODO -- Build Hamiltonian with block structure as np.block( [ [AA, AB], [AB.conj(), BB] ] )

        NMOL  = collective.num_mol
        NEXC    = collective.molecules[0].n_ES_states # Here I assumed all molecules have the same number of states...
        NMODE = self.num_modes

        gc    = np.sqrt(self.cavity_freq / 2) * self.cavity_coupling
        
        self.H_cavity = np.zeros( (NMOL * NEXC + NMODE, NMOL * NEXC + NMODE), dtype=np.complex128 )
        
        # Molecular Energies on the Diagonal of AA block
        for moli,molecule in enumerate(collective.molecules):
            for state in range(NEXC):
                #print( moli*NEXC + state, moli*NEXC + state )
                self.H_cavity[moli*NEXC + state, moli*NEXC + state] = molecule.adiabatic_energy[state+1] - molecule.adiabatic_energy[0]

        # Cavity Frequencies on the Diagonal of the BB block
        for mode in range(self.num_modes):        
            #print( NMOL*NEXC + mode, mode, NMOL*NEXC + mode )
            self.H_cavity[NMOL*NEXC + mode, NMOL*NEXC + mode] = self.cavity_freq[mode]
        
        # Coupling between Molecular States and Cavity Modes on the AB block
        for moli,molecule in enumerate(collective.molecules):
            for state in range(NEXC):
                for modei in range(self.num_modes):
                    #print( moli*NEXC + state, modei, NMOL*NEXC + modei )
                    polarized_dipole = np.dot( molecule.dipole_matrix[0,state+1,:], self.cavity_polarization[modei] )
                    phase = np.exp( 1j * moli * modei * 2 * np.pi / NMOL )
                    self.H_cavity[moli*NEXC + state, NMOL*NEXC + modei] =  gc[modei] * polarized_dipole * phase
                    self.H_cavity[NMOL*NEXC + modei, moli*NEXC + state] = (gc[modei] * polarized_dipole * phase).conj()

        E, U = np.linalg.eigh(self.H_cavity)
        self.polariton_energy = E.real
        self.polariton_wavefunctions = U

        #print( np.round(self.H_cavity,2) )