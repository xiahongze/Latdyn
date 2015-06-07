#!/usr/bin/env python

from Latdyn import ABCM,QD,Reload
import numpy as np
np.set_printoptions(precision=5,linewidth=200,suppress=True)

# calc = Reload("siqd-0.5.pickle")
# print calc.get_ph_disp()
a0 = 0.543 # in nm
# r0 = 0.5 # in nm
# a,basis,symbol = QD('diamond',r0,a0,withBC=True,r12=1.)
# symbol[symbol=="A0"] = "Si"
# symbol[symbol=="A1"] = "Si"
# # export structure with ASE
# # from ase.atoms import Atoms
# # atoms = Atoms(symbol,positions=b*a0*10.,cell=a*a0*10.,pbc=(1,1,1))
# # atoms.write("siqd.traj")
#
# mask = (symbol == 'Si')
# mass = np.ones(mask.sum())*28.085
# charge = np.ones(len(symbol))*-1
# charge[mask] = 2
# # print a
# calc = ABCM(lvec=a,basis=basis,mass=mass,symbol=symbol)
# # eps0 = 0.001
# # calc.set_ewald(charge=charge,eps=eps0,rgrid=[1,1,1],kgrid=[1,1,1])
# calc.set_nn(dist2=0.5)
# # calc.get_nn_label() # inspect the n.n
# # print calc.nn
# # calc.ecalc.get_force()
# #### set FC using set_fc() method #######
# fc_dict = {'alpha': {'Si-Si': 49.206425529087873, 'Si-BC': 19.57188469141693}, 'beta': {'BC-Si': 19.442024302561467}, 'sigma': {'Si-Si': 85.758293173499396, 'BC-Si-BC': -21.187939239442802}}
fc_dict = {'alpha': {'Si-Si': 8.4773136777830098, 'Si-BC-Si': 38.038288359413592}, 'beta': {'Si-BC': 9.752487363110907}}
# calc.set_fc(fc_dict)
# # print np.array(calc.fc)
# # calc.fix_interface()
# ################TEST A KPOINT###########
# # kpts=[[0,0,0],[0,0,0.25]]
kpts=[[0,0,0]]
# calc.set_kpts(kpts,crys=True)
# # print np.sort(calc.get_ph_disp())
# # print calc.dyn
# calc.save("siqd-0.5.pickle")

r = [0.5,1.0,1.5,2.0,2.5,3.0]
for r0 in r:
    print "r0 = %5.2f" % r0
    name = "si-qd-%.2f.pickle" % r0
    a,basis,symbol = QD('diamond',r0,a0,withBC=True,r12=1.)
    symbol[symbol=="A0"] = "Si"
    symbol[symbol=="A1"] = "Si"
    mask = (symbol == 'Si')
    mass = np.ones(mask.sum())*28.085
    calc = ABCM(lvec=a,basis=basis,mass=mass,symbol=symbol)
    calc.set_nn(dist2=0.5)
    calc.set_fc(fc_dict)
    calc.set_kpts(kpts,crys=True)
    print calc.get_ph_disp()
    calc.save(name)