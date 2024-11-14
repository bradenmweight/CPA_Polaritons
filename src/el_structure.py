import numpy as np

from pyscf import gto, scf, dft, tdscf





def do_GS_calc(self):
    # Initialize the molecule
    atom = ""
    for at in range(len(self.LABELS)):
        atom += "%s %1.10f %1.10f %1.10f; " % (self.LABELS[at], self.R[at,0], self.R[at,1], self.R[at,2])
    mol = gto.M(atom=atom, basis=self.basis_set)
    mol.verbose = 0
    mol.build()
    self.n_elec_alpha, self.n_elec_beta = mol.nelec
    if ( self.xc is None ):
        mf = scf.RHF(mol)
    else:
        mf    = dft.RKS(mol)
        mf.xc = self.xc
    mf = mf.newton()
    mf.kernel()
    self.dm          = mf.make_rdm1()
    self.GS_ENERGY   = mf.e_tot
    self.GS_GRADIENT = mf.Gradients().grad()
    self.MO_ENERGY   = mf.mo_energy
    self.mf          = mf
    self.mol         = mol
    self.C_HF        = mf.mo_coeff
    self.overlap_ao  = mol.intor("int1e_ovlp")

    # Get dipole matrix in AO representation
    nuc_dipole = np.einsum( "R,Rx->x", mol.atom_charges(), mol.atom_coords() ) / mol.atom_charges().sum()
    with mol.with_common_orig( nuc_dipole ):
        self.dipole_ao = -1 * mol.intor("int1e_r", comp=3)
    self.dipole_ao = np.einsum("xab->abx", self.dipole_ao) # (nao,nao,3)
    self.n_ao = len( self.dipole_ao )

    # Get quadrupole matrix in AO representation
    with mol.with_common_orig( nuc_dipole ):
        self.quadrupole_ao = mol.intor("int1e_rr", comp=9).reshape( (3,3,self.n_ao,self.n_ao) )
    self.quadrupole_ao = np.einsum("xyab->abxy", self.quadrupole_ao) # (nao,nao,3,3)

    self.GS_dipole = np.einsum("ab,abx->x", self.dm, self.dipole_ao)
    #print("Dipole (PySCF)", mf.dip_moment(unit="au", verbose=0) )
    #print("My Dipole", self.GS_dipole )


def do_TDA_calc(self):
    mytd = tdscf.TDA(self.mf)
    mytd.nstates = self.n_ES_states
    mytd.kernel()
    self.adiabatic_energy    = np.zeros( self.n_ES_states+1 )
    self.adiabatic_energy[0] = self.GS_ENERGY
    for state in range( self.n_ES_states ):
        self.adiabatic_energy[state+1] = mytd.e[state] + self.GS_ENERGY

    if ( self.do_TDA_gradients == True ):
        tdg = mytd.Gradients()
        self.ES_TDA_GRADIENT = np.zeros( (self.n_ES_states, self.natoms, 3) )
        for state in range( self.n_ES_states ):
            self.ES_TDA_GRADIENT[state] = tdg.kernel(state=state+1)

    # TODO -- self.ES_dipole between CIS states

    #self.ES_TDA_OSC_STR = mytd.oscillator_strength(gauge='length', order=1)
    #include corrections due to magnetic dipole and electric quadruple
    #mytd.oscillator_strength(gauge='velocity', order=1)
    #also include corrections due to magnetic quadruple and electric octupole
    #mytd.oscillator_strength(gauge='velocity', order=2)

def do_SD_calc(self):
    # Get excited slater determinant energies and dipoles
    # Use only the ground state occupied and virtual orbitals
    o = slice( 0, self.n_elec_alpha )
    v = slice( self.n_elec_alpha, self.n_ao )
    E_vo = self.MO_ENERGY[v][:,None] - self.MO_ENERGY[o]
    #for oi in range(self.n_elec_alpha):
    #    for vi in range(self.n_ao - self.n_elec_alpha):
    #        print( oi, vi, E_vo[vi,oi] )
    #print( E_vo.shape, self.n_elec_alpha, self.n_ao )
    ovE = np.array([ [oi, vi, E_vo[vi,oi]] for oi in range(self.n_elec_alpha) for vi in range(self.n_ao - self.n_elec_alpha) ])
    inds = np.argsort( ovE[:,-1] )
    ovE = ovE[inds]
    ovE = ovE[:self.n_ES_states] # Keep only n_ES_states lowest energy SD states
    #print("HOMO LUMO Gap:", self.MO_ENERGY[self.n_elec_alpha] - self.MO_ENERGY[self.n_elec_alpha-1])
    #print("HOMO LUMO Gap:", ovE[0,-1])

    self.adiabatic_energy    = np.zeros( self.n_ES_states+1 )
    self.adiabatic_energy[0] = self.GS_ENERGY
    for SDi, (oi,vi,Evo) in enumerate( ovE ):
        self.adiabatic_energy[SDi+1] = Evo + self.GS_ENERGY

    # TODO -- Get dipole matrix elements between SD states
    
    self.dipole_matrix = np.zeros( (self.n_ES_states+1, self.n_ES_states+1, 3) )
    self.dipole_matrix[0,0] = self.GS_dipole
    all_occ_mo = [j for j in range(self.n_elec_alpha)]
    L          = all_occ_mo.copy() # Copy is very important here...
    for SD1, (o1,v1,Evo) in enumerate( ovE ):
        R          = all_occ_mo.copy() # Copy is very important here...
        R[int(o1)] = int(v1) + self.n_elec_alpha
        rho_LR = np.einsum( "ai,bi->ab", self.C_HF[:,L], self.C_HF[:,R] )
        MO_DIP = np.einsum( "ab,abx->x", rho_LR, self.dipole_ao )
        self.dipole_matrix[0,SD1+1] = MO_DIP
        self.dipole_matrix[SD1+1,0] = MO_DIP
    
    for SD1, (o1,v1,Evo) in enumerate( ovE ):
        L          = all_occ_mo.copy() # Copy is very important here...
        L[int(o1)] = int(v1) + self.n_elec_alpha
        #print( "L", SD1,int(o1),int(v1), L )
        for SD2, (o2,v2,Evo) in enumerate( ovE ):
            R          = all_occ_mo.copy() # Copy is very important here...
            R[int(o2)] = int(v2) + self.n_elec_alpha
            rho_LR = np.einsum( "ai,bi->ab", self.C_HF[:,L], self.C_HF[:,R] )
            MO_DIP = np.einsum( "ab,abx->x", rho_LR, self.dipole_ao )
            #print( "R", SD2,int(o2),int(v2), R, np.round(MO_DIP,2) )
            self.dipole_matrix[SD1+1,SD2+1] = MO_DIP
    #print( np.round(self.dipole_matrix[:,:,0],1) )


def do_ES_calc(self):
    if ( self.ES_type == "TDA" ):
        do_TDA_calc(self)
    elif ( self.ES_type == "SD" ):
        do_SD_calc(self)