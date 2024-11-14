import numpy as np

from collective import Collective
from cavity import Cavity

def main():

    print("\n\tStarting Program.\n")

    mycollective = Collective(num_mol=2, num_steps=100, time_step=1.0)
    for molecule in mycollective.molecules:
        print("GS Energy of molecule    %d = %1.4f" % (molecule.mol_number, molecule.GS_ENERGY) ) 

    mycavity = Cavity(num_modes=0)
    for modei in range(mycavity.num_modes):
        print("Frequency of Cavity Mode %d = %1.4f" % (modei, mycavity.cavity_freq[modei]) )
        print("Coupling  of Cavity Mode %d = %1.4f" % (modei, mycavity.cavity_coupling[modei]) )

    print("Step 0")

    mycollective.do_el_structure()
    mycavity.build_H_cavity(mycollective)
    mycollective.step = 0
    mycollective.save_data(mycavity)
    for mycollective.step in range( 1, mycollective.num_steps ):
        print("Step %d" % mycollective.step)
        mycollective.propagate_nuclear_R()
        mycollective.do_el_structure()
        mycollective.propagate_nuclear_V()
        mycavity.build_H_cavity(mycollective)
        mycollective.save_data(mycavity)





if ( __name__ == "__main__" ):
    main()