import numpy as np

def get_GEOM(self):
    lines   = open("input_geometry.xyz", "r").readlines()
    natoms  = int(lines[0])
    LABELS  = []
    R    = np.zeros((natoms, 3))
    start_line = 2 + self.mol_number*(natoms+2)
    if ( len(lines) < start_line + natoms ):
        print("\tWarning: Not enough lines in input_geometry.xyz")
        print("\t\tUsing first molecule for the remaining number of molecules")
        start_line = 2
    for at in range(natoms):
        line = lines[start_line + at].split()
        LABELS.append( line[0] )
        R[at,:] = np.array( line[1:] )
    
    self.LABELS = LABELS
    self.R   = R / 0.529 # Angstroms --> a.u.
    self.natoms = natoms

    self.masses = set_masses( self.LABELS )

def get_VELOC(self):
    self.V   = np.zeros_like(self.R)
    try:
        lines   = open("input_velocity.xyz", "r").readlines()
    except:
        print("Doing Maxwell-Boltzmann Velocity Distribution")
        MaxwellBoltzmann(self)
    start_line = 2 + self.mol_number*(self.natoms+2)
    if ( len(lines) < start_line + self.natoms ):
        start_line = 2
    for at in range(self.natoms):
        line = lines[start_line + at].split()
        self.V[at,:] = np.array( line[1:] )
    self.V /= 0.529 * 41.341 # Ang/fs --> a.u.

    def MaxwellBoltzmann(self):
        KbT  = 300 * (0.025 / 300) / 27.2114 # K -> KT (a.u.)
        for at in range(self.natoms):
            alpha = self.masses[at] / KbT
            sigma = np.sqrt(1 / alpha)
            self.V[at,:] = np.array([gauss(0,sigma) for dof in range(3)], dtype=float)

def set_masses(LABELS):
    mass_amu_to_au = 1837/1.007 # au / amu
    masses_amu = \
{"H":   1.00797,
"He":	4.00260,
"Li":	6.941,
"Be":	9.01218,
"B":    10.81,
"C":    12.011,
"N":    14.0067,
"O":    15.9994,
"F":    18.998403,
"Ne":	20.179,
"Na":	22.98977,
"Mg":	24.305,
"Al":	26.98154,
"Si":	28.0855,
"P":    30.97376,
"S":    32.06,
"Cl":	35.453,
"K":    39.0983,
"Ar":	39.948,
"Ca":	40.08,
"Sc":	44.9559,
"Ti":	47.90,
"V":    50.9415,
"Cr":	51.996,
"Mn":	54.9380,
"Fe":	55.847,
"Ni":	58.70,
"Co":	58.9332,
"Cu":	63.546,
"Zn":	65.38,
"Ga":	69.72,
"Ge":	72.59,
"As":	74.9216,
"Se":	78.96,
"Br":	79.904,
"Kr":	83.80,
"Rb":	85.4678,
"Sr":	87.62,
"Y":    88.9059,
"Zr":	91.22,
"Nb":	92.9064,
"Mo":	95.94,
"Ru":	101.07,
"Rh":	102.9055,
"Pd":	106.4,
"Ag":	107.868,
"Cd":	112.41,
"In":	114.82,
"Sn":	18.69,
"Sb":	121.75,
"I":    126.9045,
"Te":	127.60,
"Xe":	131.30,
"Cs":	132.9054,
"Ba":	137.33,
"La":	138.9055,
"Ce":	140.12,
"Pr":	140.9077,
"Nd":	144.24,
"Sm":	150.4,
"Eu":	151.96,
"Gd":	157.25,
"Tb":	158.9254,
"Dy":	162.50,
"Ho":	164.9304,
"Er":	167.26,
"Tm":	168.9342,
"Yb":	173.04,
"Lu":	174.967,
"Hf":	178.49,
"Ta":	180.9479,
"W":    183.85,
"Re":	186.207,
"Os":	190.2,
"Ir":	192.22,
"Pt":	195.09,
"Au":	196.9665,
"Hg":	200.59,
"Tl":	204.37,
"Pb":	207.2,
"Bi":	208.9804,
"Ra":	226.0254}


    masses = []
    for at in LABELS:
        masses.append( masses_amu[at] )
    return np.array(masses) * mass_amu_to_au