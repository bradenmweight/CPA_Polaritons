import numpy as np

class QM_Dynamics():
    def __init__( self, cavity ):

        self.NPOL_STATES  = len(cavity.polariton_energy)
        self.build_initial_wavefunction()
        self.qm_propagation_type = "exact" # "exact" -- U @ phase @ U.T @ \\psi





    def build_initial_wavefunction( self ):
        self.wavefunction = np.zeros( (self.NPOL_STATES), dtype=np.complex128 )
        self.wavefunction[0] = 1.0

    def propagate_quantum( self, collective, cavity ):

        if ( self.qm_propagation_type == "exact" ):
            E  = cavity.polariton_energy
            U  = cavity.polariton_wavefunctions
            dt = collective.time_step
            phase = np.exp( -1j * E * dt )
            self.wavefunction = U.T.conj() @ self.wavefunction
            self.wavefunction *= phase
            self.wavefunction = U @ self.wavefunction
        else:
            print(f"Error: Propagation type '{self.qm_propagation_type}' not implemented")
            exit()
