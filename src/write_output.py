import numpy as np

def save_data(collective, cavity, myQM):
    save_XYZ(collective)
    save_energy(collective,cavity)
    save_temperature(collective)
    save_photon_per_eigenstate(collective,cavity)
    save_dipole_matrix(collective,cavity)
    save_polariton_wavefunctions(collective,cavity)
    save_molecular_absorption_spectra(collective,)
    #save_cavity_molecular_absorption_spectra(collective,cavity) # TODO
    save_transmission_spectrum(collective,cavity)
    save_qm_wavefunction(collective,myQM)

def save_qm_wavefunction(collective, myQM):
    # Append wavefunction to wavefunction.dat
    with open("%s/wavefunction_re.dat" % collective.output_dir, "a") as f:
        f.write("%1.4f" % (collective.step * collective.time_step / 41.341))
        for P in range(myQM.NPOL_STATES):
            f.write(" %1.4f" % np.real(myQM.wavefunction[P])**2)
        f.write("\n")
    with open("%s/wavefunction_im.dat" % collective.output_dir, "a") as f:
        f.write("%1.4f" % (collective.step * collective.time_step / 41.341))
        for P in range(myQM.NPOL_STATES):
            f.write(" %1.4f" % np.imag(myQM.wavefunction[P])**2)
        f.write("\n")
    populations = np.abs(myQM.wavefunction)**2
    with open("%s/wavefunction_populations.dat" % collective.output_dir, "a") as f:
        f.write("%1.4f" % (collective.step * collective.time_step / 41.341))
        for P in range(myQM.NPOL_STATES):
            f.write(" %1.4f" % populations[P])
        f.write("\n")


def save_transmission_spectrum(collective, cavity):
    NPOL_STATES = cavity.polariton_wavefunctions.shape[1]
    PHOT_CHAR   = np.zeros( (NPOL_STATES) )
    for P in range(NPOL_STATES):
        cavity_start_ind = NPOL_STATES - cavity.num_modes
        PHOT_CHAR[P] = np.sum( np.abs(cavity.polariton_wavefunctions[cavity_start_ind:,P])**2 )

    EGRID = np.linspace(0,1,10000) # a.u.
    SIG   = 0.001/27.2114 # a.u.
    ABS_G = np.zeros( (len(EGRID)) )
    ABS_L = np.zeros( (len(EGRID)) )
    for pt in range( len(EGRID) ):
        ABS_G[pt] = np.sum( PHOT_CHAR * np.exp( -1 * (EGRID[pt] - cavity.polariton_energy[:]) ** 2 / (2 * SIG**2) ) )
        ABS_L[pt] = np.sum( PHOT_CHAR * 1 / ( 1 + ((EGRID[pt] - cavity.polariton_energy[:])/SIG)**2 ) )
    ABS_G[:] *= 1 / np.sqrt(2*np.pi) / SIG # Normalization of the Gaussian
    ABS_L[:] *= 1 / np.pi / SIG # Normalization of the Lorentzian
    np.savetxt("%s/transmission_spectrum_Gauss_Loren__step_%d.dat" % (collective.output_dir, collective.step), np.c_[EGRID*27.2114,ABS_G,ABS_L], header="Energy(eV) Gaussian Lorentzian")


def save_molecular_absorption_spectra(collective):
    # Calculate absorption spectra from oscillator strengths
    # f_0j = (2/3) E_0j (mu_0j . mu_0j)
    # Do not include cavity polarization here. This is just the molecular absorption spectra (outside cavity).
    osc_str = np.zeros( (collective.num_mol, collective.molecules[0].n_ES_states) )            
    for moli,molecule in enumerate(collective.molecules):
        E0j  = molecule.adiabatic_energy[1:] - molecule.adiabatic_energy[0]
        mu0j = molecule.dipole_matrix[0,1:,:]
        osc_str[moli,:] = (2/3) * E0j * np.einsum("sd,sd->s", mu0j, mu0j)
    np.save("%s/molecular_oscillator_strengths__step_%d.dat" % (collective.output_dir, collective.step), osc_str) # (NMOL, NSTATES)
    
    EGRID = np.linspace(0,1,10000) # a.u.
    SIG   = 0.001/27.2114 # a.u.
    ABS_G = np.zeros( (len(EGRID)) )
    ABS_L = np.zeros( (len(EGRID)) )
    for pt in range( len(EGRID) ):
        for moli,molecule in enumerate(collective.molecules):
            E0j  = molecule.adiabatic_energy[1:] - molecule.adiabatic_energy[0]
            ABS_G[pt] += np.sum( osc_str[moli,:] * np.exp( -1 * (EGRID[pt] - E0j[:]) ** 2 / (2 * SIG**2) ) )
            ABS_L[pt] += np.sum( osc_str[moli,:] * 1 / ( 1 + ((EGRID[pt] - E0j[:])/SIG)**2 ) )
    ABS_G[:] *= 1 / np.sqrt(2*np.pi) / SIG # Normalization of the Gaussian
    ABS_L[:] *= 1 / np.pi / SIG # Normalization of the Lorentzian
    np.savetxt("%s/molecular_absorption_spectra_Gauss_Loren__step_%d.dat" % (collective.output_dir, collective.step), np.c_[EGRID*27.2114,ABS_G,ABS_L], header="Energy(eV) Gaussian Lorentzian")

def save_cavity_molecular_absorption_spectra(collective,cavity):
    """
    TODO: Implement this function
    """
    # Calculate absorption spectra from oscillator strengths of the polariton states
    # f_0J = (2/3) E_0J (mu_0J * mu_0J)
    # Include cavity polarization here.
    POL_WFN    = cavity.polariton_wavefunctions
    NS         = collective.molecules[0].n_ES_states
    NMOL       = collective.num_mol
    NPOL       = len(POL_WFN)
    MOL_DIP    = np.array([ collective.molecules[i].dipole_matrix for moli in range(NMOL)])
    MOL_DIP    = np.einsum("Aijd,dM->AijM", MOL_DIP, cavity.cavity_polarization) # (NMOL, NSTATES, 3)
    POL_DIPOLE = np.zeros( (NPOL,NPOL) )

    # # Write the dipole matrix in the Hamiltonian structure
    #for moli,molecule in enumerate(collective.molecules):
        #for state in range(NS):
            #POL_DIPOLE[moli*NS + state, moli*NS + state] = 





    # for moli,molecule in enumerate(collective.molecules):
    #     E0j  = molecule.adiabatic_energy[1:] - molecule.adiabatic_energy[0]
    #     mu0j = molecule.dipole_matrix[0,1:,:]
    #     osc_str[moli,:] = (2/3) * E0j * np.einsum("sd,sd->s", mu0j, mu0j)
    # np.save("%s/molecular_oscillator_strengths__step_%d.dat" % (collective.output_dir, collective.step), osc_str) # (NMOL, NSTATES)
    
    # EGRID = np.linspace(0,1,10000) # a.u.
    # SIG   = 0.001/27.2114 # a.u.
    # ABS_G = np.zeros( (len(EGRID)) )
    # ABS_L = np.zeros( (len(EGRID)) )
    # for pt in range( len(EGRID) ):
    #     for moli,molecule in enumerate(collective.molecules):
    #         E0j  = molecule.adiabatic_energy[1:] - molecule.adiabatic_energy[0]
    #         ABS_G[pt] += np.sum( osc_str[moli,:] * np.exp( -1 * (EGRID[pt] - E0j[:]) ** 2 / (2 * SIG**2) ) )
    #         ABS_L[pt] += np.sum( osc_str[moli,:] * 1 / ( 1 + ((EGRID[pt] - E0j[:])/SIG)**2 ) )
    # ABS_G[:] *= 1 / np.sqrt(2*np.pi) / SIG # Normalization of the Gaussian
    # ABS_L[:] *= 1 / np.pi / SIG # Normalization of the Lorentzian
    # np.savetxt("%s/molecular_absorption_spectra_Gauss_Loren_step_%d.dat" % (collective.output_dir, collective.step), np.c_[EGRID*27.2114,ABS_G,ABS_L], header="Energy(eV) Gaussian Lorentzian")



def save_photon_per_eigenstate(collective, cavity):
    # Append number of photons in each eigenstate to photons.dat
    with open("%s/photon_number.dat" % collective.output_dir, "a") as f:
        f.write("%1.4f" % (collective.step * collective.time_step / 41.341))
        NPOL_STATES = cavity.polariton_wavefunctions.shape[1]
        for P in range(NPOL_STATES):
            cavity_start_ind = NPOL_STATES - cavity.num_modes
            f.write(" %1.4f" % np.sum( np.abs(cavity.polariton_wavefunctions[cavity_start_ind:,P])**2 ))
        f.write("\n")

def save_temperature(collective):
    # Append temperature to temperature.dat
    KE = 0.0
    for molecule in collective.molecules:
        KE += 0.5000 * np.einsum("R,Rd,Rd->", molecule.masses, molecule.V, molecule.V )
    T = (2/3) * KE / (collective.num_mol * collective.molecules[0].natoms) * (300 / 0.025 * 27.2114) # a.u. --> K
    with open("%s/temperature.dat" % collective.output_dir, "a") as f:
        f.write("%1.4f %1.4f\n" % (collective.step * collective.time_step / 41.341, T))

def save_XYZ(collective):
    # Append XYZ coordinates of all molecules to XYZ trajectory.xyz
    with open("%s/trajectory.xyz" % collective.output_dir, "a") as f:
        f.write("%d\n" % sum([molecule.natoms for molecule in collective.molecules]))
        f.write("Step %d\n" % (collective.step))
        for molecule in collective.molecules:
            shift = molecule.mol_number * 10 
            coords = molecule.R * 0.529 # Angstrom
            coords[:,0] += shift # Put molecules in a line in cavity
            for at in range(molecule.natoms):
                f.write("%s %1.4f %1.4f %1.4f\n" % (molecule.LABELS[at], coords[at,0], coords[at,1], coords[at,2]) )

def save_energy(collective, cavity):
    # Append energies of all molecules to energy_mol.dat
    E_GS = 0.0
    for molecule in collective.molecules:
        E_GS += molecule.GS_ENERGY
    with open("%s/energy_mol.dat" % collective.output_dir, "a") as f:
        f.write("%1.4f" % (collective.step * collective.time_step / 41.341))
        for molecule in collective.molecules:
            for state in range(molecule.n_ES_states+1):
                f.write(" %1.8f" % (molecule.adiabatic_energy[state]) )
        f.write("\n")
    
    # Append polariton energies to energy_polariton.dat
    with open("%s/energy_polariton.dat" % collective.output_dir, "a") as f:
        f.write("%1.4f" % (collective.step * collective.time_step / 41.341))
        for energy in cavity.polariton_energy:
            f.write(" %1.8f" % energy)
        f.write("\n")

def save_dipole_matrix(collective,cavity):
    # Save the dipole matrix as a .npy file for each time-step
    dipole_matrix = np.zeros( (collective.num_mol, collective.molecules[0].n_ES_states+1, collective.molecules[0].n_ES_states+1,3) )
    dipole_matrix_projected = np.zeros( (collective.num_mol, collective.molecules[0].n_ES_states+1, collective.molecules[0].n_ES_states+1, cavity.num_modes) )
    for moli,molecule in enumerate( collective.molecules ):
        dipole_matrix[moli] = molecule.dipole_matrix
        for mode in range(cavity.num_modes):
            dipole_matrix_projected[moli,:,:,mode] = np.einsum("abx,x->ab", molecule.dipole_matrix, cavity.cavity_polarization[mode])
    np.save("%s/dipole_matrix__step_%d.npy" % (collective.output_dir,collective.step), dipole_matrix)
    np.save("%s/dipole_matrix_projected__step_%d.npy" % (collective.output_dir,collective.step), dipole_matrix_projected)

def save_polariton_wavefunctions(collective, cavity):
    # Save the polariton wavefunctions as a .npy file for each time-step
    np.save("%s/polariton_wavefunctions__step_%d.npy" % (collective.output_dir,collective.step), cavity.polariton_wavefunctions)