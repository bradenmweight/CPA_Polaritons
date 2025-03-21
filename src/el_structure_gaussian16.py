import numpy as np
from time import time, sleep
import subprocess as sp
import os

def do_GS_calc(self):
    sp.call("mkdir -p scratch/%d_GS" % (self.mol_number), shell=True)
    os.chdir("scratch/%d_GS" % (self.mol_number))
    
    COM_FILE = open("geometry.com", "w")
    string = \
"""%chk=geometry.chk
%mem=1GB
%nprocshared=1

#P Force AM1/STO-3G NoSymm

Title Card Required

0 1
"""
    for at in range(len(self.LABELS)):
        # G16 expects Angstrom
        string += "%s %1.10f %1.10f %1.10f\n" % (self.LABELS[at], self.R[at,0]*0.529, self.R[at,1]*0.529, self.R[at,2]*0.529)
    string += "\n\n\n\n\n\n\n\n\n\n\n"
    COM_FILE.write(string)
    COM_FILE.close()

    # # Run Gaussian16
    sp.call("g16 < geometry.com > geometry.out", shell=True)
    sleep(5)

    # # Read the ground state energy
    E_GS = float( sp.check_output("grep 'SCF Done' geometry.out | tail -1 | awk '{print $5}'", shell=True).decode() )

    # # Read the ground state forces
    FORCE      = np.zeros( (len(self.LABELS),3) )
    sp.call("grep 'Forces (Hartrees/Bohr)' geometry.out -A %d | tail -n %d | awk '{print $3, $4, $5}' > FORCE.dat" % (self.natoms+2, self.natoms), shell=True)
    FORCE[:,:] = np.loadtxt("FORCE.dat")

    self.GS_ENERGY   = E_GS
    self.GS_GRADIENT = -1 * FORCE

    os.chdir("../../")

    return self


def do_TDA_calc(self):

    sp.call("mkdir -p scratch/%d_ES" % (self.mol_number), shell=True)
    sp.call("mv scratch/%d_GS/geometry.chk scratch/%d_ES/" % (self.mol_number, self.mol_number), shell=True)
    os.chdir("scratch/%d_ES" % (self.mol_number))
    
    COM_FILE = open("geometry.com", "w")
    string = \
f"""%chk=geometry.chk
%mem=1GB
%nprocshared=1

#P AM1/STO-3G NoSymm guess=read
#P TDA=(singlets,nstates={self.n_ES_states+1})

Title Card Required

0 1
"""
    for at in range(len(self.LABELS)):
        # G16 expects Angstrom
        string += "%s %1.10f %1.10f %1.10f\n" % (self.LABELS[at], self.R[at,0]*0.529, self.R[at,1]*0.529, self.R[at,2]*0.529)
    string += "\n\n\n\n\n\n\n\n\n\n\n"
    COM_FILE.write(string)
    COM_FILE.close()

    # # Run Gaussian16
    sp.call("g16 < geometry.com > geometry.out", shell=True)

    # # Read the excited state energies
    sp.call("grep 'Excited State' geometry.out | tail -n %d | awk '{print $5}' > EXC_ENERGY.dat" % (self.n_ES_states), shell=True)
    E_ES  = np.loadtxt("EXC_ENERGY.dat")
    sp.call("rm EXC_ENERGY.dat", shell=True)

    # # Read the transition dipoles
    DIP      = np.zeros( (self.n_ES_states,3) )
    sp.call("grep 'Ground to excited state transition electric dipole moments' geometry.out -A %d | tail -n %d | awk '{print $2, $3, $4}' > DIP.dat" % (self.n_ES_states+1, self.n_ES_states), shell=True)
    DIP[:,:] = np.loadtxt("DIP.dat")

    self.adiabatic_energy     = np.zeros( (self.n_ES_states+1) )
    self.adiabatic_energy[0]  = self.GS_ENERGY
    self.adiabatic_energy[1:] = self.GS_ENERGY + E_ES / 27.2114 # Convert to a.u.
    self.dipole_matrix        = np.zeros( (self.n_ES_states+1, self.n_ES_states+1, 3) )
    self.dipole_matrix[1:,0]  = DIP # Already in a.u.
    self.dipole_matrix[0,1:]  = DIP # Already in a.u.

    os.chdir("../../")
    sp.call("rm -r scratch/%d_GS" % (self.mol_number), shell=True)
    sp.call("rm -r scratch/%d_ES" % (self.mol_number), shell=True)

    return self

def do_SD_calc(self):
    # Get excited slater determinant energies and dipoles
    # Use only the ground state occupied and virtual orbitals
    raise NotImplementedError("Not implemented for Gaussian16 yet. Use PySCF instead -- is very slow though.")

def do_ES_calc(self):
    if ( self.ES_type == "TDA" ):
        return do_TDA_calc(self)
    elif ( self.ES_type == "SD" ):
        return do_SD_calc(self)