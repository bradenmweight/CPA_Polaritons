import numpy as np
from time import time
import os
import subprocess as sp

from collective import Collective
from cavity import Cavity
from dynamics import QM_Dynamics
from write_output import save_data

def main():

    print("\n\tStarting Program.\n")
    if ( os.path.exists("scratch") ): # TODO - Push this to compute node scratch space
        sp.call("rm -r scratch", shell=True) # TODO - Push this to compute node scratch space
    sp.call("mkdir -p scratch", shell=True) # TODO - Push this to compute node scratch space


    mycollective = Collective(num_mol=108, num_steps=1000, time_step=1.0)
    for molecule in mycollective.molecules:
        print("GS Energy of molecule               %d = %1.4f" % (molecule.mol_number, molecule.GS_ENERGY) ) 

    mycavity = Cavity(mycollective)
    mycavity.build_H_cavity(mycollective)
    for modei in range(mycavity.num_modes):
        print("Frequency (Coupling) of Cavity Mode %d = %1.4f (%1.4f)" % (modei, mycavity.cavity_freq[modei], mycavity.cavity_coupling[modei]) )

    print("Step 0")
    #mycollective.do_el_structure() # Already done in Collective.__init__
    myQM = QM_Dynamics( mycavity )
    mycollective.step = 0
    save_data(mycollective, mycavity, myQM)
    for mycollective.step in range( 1, mycollective.num_steps ):
        T0 = time()
        mycollective.propagate_nuclear_R()
        mycollective.do_el_structure()
        mycollective.propagate_nuclear_V()
        mycavity.build_H_cavity(mycollective)
        myQM.propagate_quantum(mycollective, mycavity)
        save_data(mycollective, mycavity, myQM)
        print("Step %d took %1.2f seconds." % (mycollective.step, time()-T0) )
    
    sp.call("rm -r scratch", shell=True) # TODO - Push this to compute node scratch space



if ( __name__ == "__main__" ):
    main()