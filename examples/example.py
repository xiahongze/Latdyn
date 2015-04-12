###########
# Example #
###########
import numpy as np
from latdyn.ewald import Ewald
lvec = np.array([[0.5,0.5,0.],
                 [0.5,0.,0.5],
                 [0.,0.5,0.5]])
charge = np.array([1,-1])
# charge = np.array([2,2,-1,-1,-1,-1])
basis = np.array([[0.,0.,0.],[0.25,0.25,0.25]])
# bc = np.array([[1,1,1],[-1,-1,1],[-1,1,-1],[1,-1,-1]])*0.25*0.5
# bc = np.array([[1,1,1],[1,1,-1],[1,-1,1],[-1,1,1]])*0.25*0.5
# basis = np.vstack((basis,bc))
calc = Ewald(lvec=lvec, basis=basis, charge=charge, \
            rgrid=[3,3,3], kgrid=[3,3,3])
calc.get_etot()
print 'The self-energy is', calc.es
print 'The real space energy is', calc.er
print 'The reciprocal space energy is', calc.ek
print 'The total energy is', calc.etot
print 'The Ewald parameter used is', calc.alp
r = basis[1] - basis[0]
r0 = np.sqrt(np.sum(r**2))
print 'The Madelung constant is', calc.etot*r0/charge[0]/charge[1]
calc.get_force()
