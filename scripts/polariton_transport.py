import numpy as np
from matplotlib import pyplot as plt
from numba import njit
import subprocess as sp


def get_globals():
    global data_dir, LAM, NMOL, EPOL, RMOL, OUTPUT_DIR, num_modes
    EPOL            = np.array([1,1,1])
    EPOL            = EPOL / np.linalg.norm(EPOL)
    NMOL            = 108
    num_modes       = 27
    data_dir        = "LAM_0.05_NMOL_108" # "LAM_0.05_NMOL_108", "LAM_0.0_NMOL_1000"
    OUTPUT_DIR      = "DYNAMICS_NMOL_108_NMODE_27" # "DYNAMICS_NMOL_108_NMODE_27",  "DYNAMICS_NMOL_1000_NMODE_201"
    sp.call(f"mkdir -p {OUTPUT_DIR}", shell=True)

    global dx, Lx # Cavity parameters
    dx   = 40/0.529 # Ang --> Bohr
    Lx   = (NMOL-1) * dx
    RMOL = np.arange( NMOL ) * dx

def read_data():
    global mol_energy, mol_dipole, time, nstates
    print("\n\tReading molecular data...")
    mol_energy    = np.loadtxt(f"{data_dir}/OUTPUT/energy_mol.dat") # (time, mol_1 S0, mol_1 S1, ..., mol_2 S0, mol_2 S1, ...)
    nsteps        = mol_energy.shape[0]
    time          = mol_energy[:,0] # already in fs
    mol_energy    = mol_energy[:,1:] # (nsteps,nstates) # Remove time column
    nstates       = mol_energy.shape[1] // NMOL
    tmp           = np.zeros((nsteps,NMOL,nstates))
    for moli in range(NMOL):
        for state in range(nstates):
            tmp[:,moli,state] = mol_energy[:,moli*nstates + state]
    mol_energy = tmp

    mol_dipole   = np.zeros((nsteps,NMOL,nstates,nstates,3),dtype=np.complex128)
    for step in range(nsteps):
        mol_dipole[step] = np.load(f"{data_dir}/OUTPUT/dipole_matrix__step_{step}.npy") # (NMOL,nstates,nstates,3)
    print("\tFinished reading molecular data.")

def get_FP_cavity():
    global cavity_angle, cavity_freq, cavity_k, cavity_polarization, cavity_coupling
    EMIN                   = 3.0/27.2114
    EMAX                   = 4.0/27.2114
    #num_modes              = 201 # 201, 27
    #num_modes_half         = num_modes // 2 + 1
    #cavity_coupling        = np.ones( (num_modes_half) ) * LAM # Assume equal coupling
    cavity_polarization    = np.array([EPOL]*num_modes) # Assume equal direction
    # Thetamax               = np.arctan( (EMAX/EMIN)**2 - 1 )
    # cavity_freq            = np.zeros( (num_modes_half) )
    # cavity_angle           = np.linspace(0,Thetamax,num_modes_half)
    # cavity_freq[:]         = EMIN * np.sqrt( 1 + np.tan(cavity_angle)**2 )
    # cavity_k               = EMIN * 137.03599 * np.tan( cavity_angle ) # c = 137.03599 a.u.
    
    # cavity_freq            = np.concatenate( ( cavity_freq[1:][::-1], cavity_freq) )
    #cavity_polarization    = np.concatenate( ( cavity_polarization[1:][::-1], cavity_polarization) )
    # cavity_k               = np.concatenate( ( -cavity_k[1:][::-1], cavity_k) )
    # cavity_angle           = np.concatenate( ( -cavity_angle[1:][::-1], cavity_angle) )
    #cavity_coupling        = np.concatenate( ( -cavity_coupling[1:][::-1], cavity_coupling) )


    ### FROM BEN
    c    = 137.03599
    kz   = EMIN/c
    kx   = 2 * np.pi * np.arange(-num_modes//2+1,num_modes//2+1) / Lx
    nr   = 1.0       # Check with experiments
    cavity_freq = (c/nr) * (kx**2.0 + kz**2.0)**0.5
    cavity_k    = kx

@njit
def get_H_GTC( step, LAM ):
    """
    Build the Hamiltonian in the first excited subspace
    """
    NPOL = NMOL * (nstates-1) + num_modes
    EMOL = mol_energy[step,:,1:] - mol_energy[step,:,0] # (NMOL,nstates)
    DIP  = mol_dipole[step,:,0,1:,:] # (NMOL,nstates,nstates,3)
    
    gc    = np.sqrt(cavity_freq / 2) * np.ones(num_modes) * LAM / np.sqrt(NMOL)
    
    H = np.zeros( (NPOL, NPOL), dtype=np.complex128 )
    
    # Molecular Energies on the Diagonal of AA block
    for moli in range(NMOL):
        for state in range(nstates-1):
            H[moli*(nstates-1) + state, moli*(nstates-1) + state] = EMOL[moli,state]

    # Cavity Frequencies on the Diagonal of the BB block
    for mode in range(num_modes):        
        H[NMOL*(nstates-1) + mode, NMOL*(nstates-1) + mode] = cavity_freq[mode]
    
    # Coupling between Molecular States and Cavity Modes on the AB block
    for moli in range(NMOL):
        for state in range(nstates-1):
            for modei in range(num_modes):
                polarized_dipole = np.sum(DIP[moli,state,:] * cavity_polarization[modei]) # np.einsum("d,d->", DIP[moli,state,:], cavity_polarization[modei] )
                phase            = np.exp( -1j * cavity_k[modei] * RMOL[moli] )
                H[moli*(nstates-1) + state, NMOL*(nstates-1) + modei] =              gc[modei] * polarized_dipole * phase
                H[NMOL*(nstates-1) + modei, moli*(nstates-1) + state] = np.conjugate(gc[modei] * polarized_dipole * phase)

    E, U = np.linalg.eigh( H )
    return H, E, U

def get_initial_conditions( E, U ):
    """
    Start from Gaussian function centered at certain positive cavity_k and with energy E0
    """
    # E0 = np.average( mol_energy[:,:,1:] - mol_energy[:,:,0][:,:,None] ) # Average molecular transition energy
    # gc = np.sqrt(cavity_freq / 2) * cavity_coupling
    
    # print("Starting Gaussian state centered aat the average molecular transition energy + half approx. Rabi splitting")
    # print("Average molecular transition energy: %1.4f eV"% (E0*27.2114))
    # within_window = np.where(np.abs(E - E0) < 0.005)[0]
    # print("Half approx. Rabi splitting: %1.4f eV"% ( gc[init_mode] * 27.2114 / 2))
    # E_shift  = E0 + gc[init_mode] / 2
    # SIG_E    = 0.1 / 27.2114 # LASER Width
    # psi_0    = np.exp( - ((E - E_shift))**2 / 2 / SIG_E**2 )
    # psi_0[:] = psi_0 / np.linalg.norm(psi_0)

    psi_adFock    = np.zeros( (NMOL * (nstates-1) + num_modes) )
    #psi_adFock[NMOL//2]   = 1.0 # Symmetric
    #psi_adFock[NMOL//2+1] = 1.0 # Asymmetric
    #psi_adFock[0] = 1.0
    # psi_adFock[1] = 1.0
    psi_adFock[:NMOL] = np.exp( -(RMOL - RMOL[NMOL//2])**2 / 2 / (2*dx)**2 )
    psi_adFock[NMOL:] = 1e-12 # For numerical perposes later
    psi_adFock = psi_adFock / np.linalg.norm(psi_adFock)
    psi_0      = np.einsum("aj,a->j", U, psi_adFock )
    return psi_0

@njit
def rotate_psi( U1, U0, psi ):
    return U1.conj().T @ U0 @ psi # |E(t1)> = <E(t1)|j,n> <j,n|E(t0)> |E(t0)>

@njit
def createEU( U ):
    overlap_mat = np.conj( U ).T
    for n in range(nstates):
        f = np.exp(1j*np.angle(overlap_mat[n,n]))
        U[:,n] = f * U[:,n] # phase correction
    return U

@njit
def updateEU( U0, U1):
    overlap_mat = np.conj(U1).T @ U0
    for n in range(nstates):
        f = np.exp(1j * np.angle(overlap_mat[n,n]) )
        U1[:,n] = f * U1[:,n] # phase correction
    return U1

def run_dynamics( LAM ):
    print("\n\tStarting dynamics...")
    psi_xn_list   = []
    E_AVE         = np.zeros( ( len(time) ) )
    R_AVE         = np.zeros( ( len(time) ) )
    STD_R_AVE     = np.zeros( ( len(time) ) )
    RMSD_AVE_R    = np.zeros( ( len(time) ) )
    RMSD_AVE_k    = np.zeros( ( len(time) ) )
    k_AVE         = np.zeros( ( len(time) ) )
    STD_k_AVE     = np.zeros( ( len(time) ) )
    IPR           = np.zeros( ( len(time) ) )
    
    H, E0, U0     = get_H_GTC( 0, LAM )
    U0 = createEU( U0 )
    psi           = get_initial_conditions( E0, U0 )
    psi_molFock   = np.einsum("aj,j->a", U0, psi ) # Go to non-diagonal basis
    psi_xn        = get_psi_xn( psi_molFock ) # Compute spatial wavefunction
    psi_km        = get_psi_km( psi_molFock ) # Compute momentum wavefunction
    _, _, R_AVE_0 = measure_RMSD_R2_R( psi_xn, 0 )
    _, _, k_AVE_0 = measure_RMSD_k2_k( psi_km, 0 )
    dt            = time[1] - time[0]
    psi_xn_list.append( psi_xn )
    for step in range(len(time)-1):
        #print("Working on step %d of %d" % (step,len(time)))
        E_AVE[step]                = np.sum( E0 * psi.conj() * psi ).real # Quantum energy only

        psi_molFock    = np.einsum("aj,j->a", U0, psi ) # Go to non-diagonal basis
        psi_xn         = get_psi_xn( psi_molFock ) # Compute spatial wavefunction
        psi_km         = get_psi_km( psi_molFock ) # Compute momentum wavefunction
        IPR[step]      = measure_IPR( psi_xn )
        RMSD_AVE_R[step], STD_R_AVE[step], R_AVE[step] = measure_RMSD_R2_R( psi_xn, R_AVE_0 )
        RMSD_AVE_k[step], STD_k_AVE[step], k_AVE[step] = measure_RMSD_k2_k( psi_km, k_AVE_0 )

        H, E1, U1                  = get_H_GTC( step+1, LAM )
        U1                         = updateEU( U0, U1)
        psi                        = rotate_psi( U1, U0, psi ) # E(t0) to E(t1) basis
        psi                        = np.exp( -1j * E1 * dt ) * psi

        U0 = U1*1
        E0 = E1*1
        psi_xn_list.append( psi_xn )

    # Save last step's data
    E_AVE[-1]              = np.sum( E1 * psi.conj() * psi ).real
    psi_molFock            = np.einsum("aj,j->a", U1, psi ) # Go to non-diagonal basis
    psi_xn         = get_psi_xn( psi_molFock ) # Compute spatial wavefunction
    psi_km         = get_psi_km( psi_molFock ) # Compute momentum wavefunction
    IPR[-1]        = measure_IPR( psi_xn )
    RMSD_AVE_R[-1], STD_R_AVE[-1], R_AVE[-1] = measure_RMSD_R2_R( psi_xn, R_AVE_0 )
    RMSD_AVE_k[-1], STD_k_AVE[-1], k_AVE[-1] = measure_RMSD_k2_k( psi_km, k_AVE_0 )
    psi_xn_list.append( psi_xn )

    print("\tFinished dynamics.")
    make_plots( LAM, E_AVE, R_AVE, STD_R_AVE, RMSD_AVE_R, k_AVE, STD_k_AVE, RMSD_AVE_k, IPR, psi_xn_list )

    return RMSD_AVE_R


def main():
    get_globals()
    read_data()
    get_FP_cavity()
    
    LAM_LIST = [0.0,0.025,0.05,0.075,0.1]
    MSD      = np.zeros( (len(LAM_LIST),len(time)) )
    for lami,lam in enumerate(LAM_LIST):
        print("Working on coupling strength %1.3f a.u." % (lam))
        MSD[lami,:] = run_dynamics( lam )

    #print("\tFinished simulation.")
    #print("NMOL = %d, LAM = %1.3f a.u., dx = %1.2f nm, Lx = %1.2f nm" % (NMOL, LAM, dx*0.529/10, Lx*0.529/10))

    for lami,lam in enumerate(LAM_LIST):
        plt.plot(time, MSD[lami]*0.529/10, label="$\\lambda$ = %1.3f a.u." % (lam) )
    plt.legend()
    plt.xlabel("Time (fs)", fontsize=15)
    plt.ylabel("MSD = $\\langle R^2 \\rangle (t) - \\langle R \\rangle^2 (0)$ (nm)", fontsize=15)
    plt.tight_layout()
    plt.savefig("%s/MSD_LAM_SCAN.jpg" % (OUTPUT_DIR), dpi=300)
    plt.clf()




def get_psi_xn( psi_molFock ):
    MATT   = np.sum( np.abs(psi_molFock[:NMOL])**2 )
    PHOT   = np.sum( np.abs(psi_molFock[NMOL:])**2 )
    phase  = np.exp( 1j * RMOL[:,None] * cavity_k[None,:]  )
    psi_xn =  MATT * psi_molFock[:NMOL] \
            + PHOT * np.einsum("R,Rm,m->R", psi_molFock[:NMOL], phase, psi_molFock[NMOL:]) / np.sqrt(NMOL)
    norm   = np.sum( psi_xn.conj() * psi_xn ).real
    return psi_xn / np.sqrt(norm)

def get_psi_km( psi_molFock ):
    MATT   = np.sum( np.abs(psi_molFock[:NMOL])**2 )
    PHOT   = np.sum( np.abs(psi_molFock[NMOL:])**2 )
    phase  = np.exp( -1j * RMOL[:,None] * cavity_k[None,:]  )
    psi_km =  MATT * np.einsum("R,Rm,m->m", psi_molFock[:NMOL], phase, psi_molFock[NMOL:]) / np.sqrt(num_modes) \
            + PHOT * psi_molFock[NMOL:]
    norm   = np.sum( psi_km.conj() * psi_km ).real
    return psi_km / np.sqrt(norm)

def measure_RMSD_R2_R( psi_xn, R_AVE_0 ):
    """
    psi must be in the non-diagonal basis
    """
    R1_OP  = RMOL    # TODO -- Include possibility of nstates > 1
    R2_OP  = RMOL**2 # TODO -- Include possibility of nstates > 1
    norm   = np.sum( psi_xn.conj() * psi_xn ).real
    R_AVE  = np.einsum("R,R,R->", psi_xn.conj(), R1_OP, psi_xn ).real / norm
    R2_AVE = np.einsum("R,R,R->", psi_xn.conj(), R2_OP, psi_xn ).real / norm
    STD_R  = np.abs(R2_AVE - R_AVE**2) # Sometimes negative zero at initial time, need abs to kill the python warning
    RMSD   = np.abs(R2_AVE - R_AVE_0**2) # Sometimes negative zero at initial time, need abs to kill the python warning
    return np.sqrt(RMSD), np.sqrt( STD_R ), R_AVE

def measure_RMSD_k2_k( psi_km, k_AVE_0 ):
    """
    psi must be in the non-diagonal basis
    """
    k1_OP  = cavity_k    # TODO -- Include possibility of nstates > 1
    k2_OP  = cavity_k**2 # TODO -- Include possibility of nstates > 1
    norm   = np.sum( psi_km.conj() * psi_km ).real
    k_AVE  = np.einsum("m,m,m->", psi_km.conj(), k1_OP, psi_km ).real / norm
    k2_AVE = np.einsum("m,m,m->", psi_km.conj(), k2_OP, psi_km ).real / norm
    STD_k  = np.abs(k2_AVE - k_AVE**2) # Sometimes negative zero at initial time, need abs to kill the python warning
    RMSD   = np.abs(k2_AVE - k_AVE_0**2) # Sometimes negative zero at initial time, need abs to kill the python warning
    return np.sqrt(RMSD), np.sqrt( STD_k ), k_AVE

def measure_IPR( psi_xn ):
    """
    psi must be in the non-diagonal basis
    """
    PROB   = np.abs(psi_xn)**2
    IPR    = 1 / np.sum( PROB**2 ).real
    return IPR

def make_plots( LAM, E_AVE, R_AVE, STD_R_AVE, RMSD_AVE_R, k_AVE, STD_k_AVE, RMSD_AVE_k, IPR, psi_xn_list ):
    plt.plot(time, E_AVE*27.2114)
    plt.xlabel("Time (fs)", fontsize=15)
    plt.ylabel("Average Energy (eV)", fontsize=15)
    plt.tight_layout()
    plt.savefig("%s/E_AVE_LAM_%1.4f.jpg" % (OUTPUT_DIR,LAM), dpi=300)
    plt.clf()

    plt.plot(time, R_AVE / Lx * 100)
    plt.xlabel("Time (fs)", fontsize=15)
    plt.ylabel("$\\langle R \\rangle$ (% of box)", fontsize=15)
    plt.tight_layout()
    plt.savefig("%s/R_AVE_LAM_%1.4f.jpg" % (OUTPUT_DIR,LAM), dpi=300)
    plt.clf()

    plt.plot(time, STD_R_AVE / Lx * 100)
    plt.xlabel("Time (fs)", fontsize=15)
    plt.ylabel("$\\langle (\\Delta R)^2 \\rangle$ (% of box)", fontsize=15)
    plt.tight_layout()
    plt.savefig("%s/STD_R_LAM_%1.4f.jpg" % (OUTPUT_DIR,LAM), dpi=300)
    plt.clf()

    plt.plot(time, RMSD_AVE_R * 0.529 / 10)
    plt.xlabel("Time (fs)", fontsize=15)
    plt.ylabel("$\\langle R^2 \\rangle (t) - \\langle R \\rangle^2 (0)$ (nm)", fontsize=15)
    plt.tight_layout()
    plt.savefig("%s/RMSD_R_LAM_%1.4f.jpg" % (OUTPUT_DIR,LAM), dpi=300)
    plt.clf()

    plt.plot(time, RMSD_AVE_R * 0.529 / 10 / RMSD_AVE_R[0] )
    plt.xlabel("Time (fs)", fontsize=15)
    plt.ylabel("$\\frac{\\langle R^2 \\rangle (t) - \\langle R \\rangle^2 (0)}{\\langle R^2 \\rangle (0) - \\langle R \\rangle^2 (0)}$", fontsize=15)
    plt.tight_layout()
    plt.savefig("%s/RMSD_R_NORM_LAM_%1.4f.jpg" % (OUTPUT_DIR,LAM), dpi=300)
    plt.clf()

    plt.plot(time, k_AVE / (2 * np.pi / Lx))
    plt.xlabel("Time (fs)", fontsize=15)
    plt.ylabel("Average Momentum (units of $\\frac{2\\pi}{L}$)", fontsize=15)
    plt.tight_layout()
    plt.savefig("%s/k_LAM_%1.4f.jpg" % (OUTPUT_DIR,LAM), dpi=300)
    plt.clf()    

    plt.plot( time, STD_k_AVE )
    plt.xlabel("Time (fs)", fontsize=15)
    plt.ylabel("$\\langle (\\Delta k)^2 \\rangle$", fontsize=15)
    plt.tight_layout()
    plt.savefig("%s/STD_k_LAM_%1.4f.jpg" % (OUTPUT_DIR,LAM), dpi=300)
    plt.clf()

    plt.plot(time, RMSD_AVE_k / (2 * np.pi / Lx))
    plt.xlabel("Time (fs)", fontsize=15)
    plt.ylabel("$\\langle R^2 \\rangle (t) - \\langle R \\rangle^2 (0)$ (units of $\\frac{2\\pi}{L}$)", fontsize=15)
    plt.tight_layout()
    plt.savefig("%s/RMSD_k_LAM_%1.4f.jpg" % (OUTPUT_DIR,LAM), dpi=300)
    plt.clf()

    plt.plot(time, RMSD_AVE_k / (2 * np.pi / Lx) / RMSD_AVE_k[0] )
    plt.xlabel("Time (fs)", fontsize=15)
    plt.ylabel("$\\frac{\\langle R^2 \\rangle (t) - \\langle R \\rangle^2 (0)}{\\langle R^2 \\rangle (0) - \\langle R \\rangle^2 (0)}$ (units of $\\frac{2\\pi}{L}$)", fontsize=15)
    plt.tight_layout()
    plt.savefig("%s/RMSD_k_NORM_LAM_%1.4f.jpg" % (OUTPUT_DIR,LAM), dpi=300)
    plt.clf()

    plt.plot(time, IPR)
    plt.xlabel("Time (fs)", fontsize=15)
    plt.ylabel("Inverse PR", fontsize=15)
    plt.tight_layout()
    plt.savefig("%s/IPR_LAM_%1.4f.jpg" % (OUTPUT_DIR,LAM), dpi=300)
    plt.clf()   

    fig, ax = plt.subplots()
    colormap = plt.get_cmap("brg")
    time_plot = np.arange(0, len(time), 50)
    for step, t in enumerate( time_plot ):
        plt.semilogy(RMOL*0.529/10, np.abs(psi_xn_list[step])**2, lw=1, alpha=0.25, color=colormap(step/len(time_plot)))
    mappable = plt.cm.ScalarMappable(cmap=colormap, norm=plt.Normalize(vmin=time_plot[0], vmax=time_plot[-1]))
    plt.colorbar(mappable, ax=ax, label="Time (fs)", pad=0.01)
    plt.xlabel("Position (nm)", fontsize=15)
    plt.ylabel("Probability Density", fontsize=15)
    plt.ylim(1e-4,1)
    plt.tight_layout()
    plt.savefig("%s/psi_xn_LAM_%1.4f.jpg" % (OUTPUT_DIR,LAM), dpi=300)
    plt.clf()


if ( __name__ == "__main__" ):
    main()

