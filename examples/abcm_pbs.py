#!/usr/bin/env python

from Latdyn import ABCM,BulkBuilder,default_k_path
import numpy as np
np.set_printoptions(precision=3,linewidth=200,suppress=True)

r12 = 1.5 # r_cation/r_anion
cc = 6*r12*r12/(1+r12*r12) # charge of cation, for tetrahedral bonding
ca = 6 - cc # charge of anion, for tetrahedral bonding

a,b,bc,symion,symbc = BulkBuilder("rocksalt",withBC=True,r12=r12)
basis = np.vstack((b,bc))

mass = [207.2,32.066]
symbol = ['Pb','S','BC','BC','BC','BC','BC','BC']
calc = ABCM(lvec=a,basis=basis,mass=mass,symbol=symbol)
calc.set_ewald(charge=[cc,ca,-1,-1,-1,-1,-1,-1],eps=0.66832)
calc.set_nn(dist2=0.5)
calc.get_nn_label() # insperct the n.n

#### set FC using set_fc() method #######
# a_dict = {"Pb-S":40.,"Pb-BC":10.,"BC-S": 10.}
# b_dict = {"BC-S": 20.,"Pb-BC":10.}
# fc_dict = {"alpha":a_dict,"beta":b_dict}
# FITTED ONE
fc_dict = {'alpha': {'BC-S': -0.65272215878396056, 'Pb-S': 2.6121752429292417, 'Pb-BC': -0.0037192306306280116}, 'beta': {'BC-S': 22.88292483831167, 'Pb-BC': -1.1627118875739446}}
calc.set_fc(fc_dict)
calc.fix_interface()

################TEST A KPOINT###########
# kpts=[[0,0,0],[0.5,0.5,0]]
# kpts=[[0.5,0.5,0.]]
# kpts=[[0.,0.,0]]
# kpts = [ [0.,0.,0.],[0.5,0.5,0.],[0.5,0.5,0.5] ]
# calc.set_kpts(kpts,crys=True)
# print calc.get_ph_disp()
# print calc.dyn.real
# print calc.dyn.imag
############## FITTING ##############
# you could fit the model to more than one k-point
# src_freq = [[0.,0.,0.,15.4,15.4,15.4],[4.5,4.5,12.3,12.3,13.9,13.9],[3.45,3.45,11.3,12.6,14.7,14.7]]
# kpts = [ [0.,0.,0.],[0.5,0.5,0.],[0.5,0.5,0.5] ]
# calc.fit_freq(src_freq,kpts,fc_dict,method='Powell')
# from qeplotter import ReadBand
# b0 = ReadBand("pbs1.freq")
# calc.fit_freq(b0.band/33.3333,b0.kvec,fc_dict,eps0=7.,method='L-BFGS-B',crys=False,maxiter=300) # L-BFGS-B,Nelder-Mead,Powell

####### define kpath #############
kpts, point_names, x, X = default_k_path("fcc",a,num=300)
calc.set_kpts(kpts,crys=True)

calc.get_ph_disp()
calc.plot(mytup=(point_names,x,X),filename="ph.pdf")
# calc.save_ph_csv(phunit="cm-1")

#### DOS calc. ##############
# calc.get_dos(kgrid=(8,8,8))
# calc.plot_dos()

####### Thermal conductivity and specific heat #########
# temp = np.linspace(1.,1000,200)
# calc.get_debye(temp,alat=5.43,kgrid=(4,4,4))
# calc.get_therm_cond(temp,alat=5.43,tau=50.,x="x",y="y",kgrid=(4,4,4),koff=(1,1,1))
