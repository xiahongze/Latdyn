#!/usr/bin/env python

pi                            = 3.14159265358979323846
TPI                          = 2.0*pi
FPI                          = 4.0*pi
H_PLANCK_SI          = 6.62606896E-34   # J s
HARTREE_SI            = 4.35974394E-18   # J
C_SI                       = 2.99792458E+8    # m sec^-1
AU_SEC                  = H_PLANCK_SI/TPI/HARTREE_SI
AU_PS                    = AU_SEC * 1.0E+12
AU_TERAHERTZ       = AU_PS
RY_TO_THZ             = 1.0 / AU_TERAHERTZ / FPI
RY_TO_CMM1          = 1.E+10 * RY_TO_THZ / C_SI
M_PROTON              = 1.67262178E-27   # kg
THZ                        = 1.0E+12          # s^-1
M_THZ                    = 1.0/M_PROTON/THZ/THZ
KB                          = 1.3806488e-23    # J K^-1 Boltzmann constant
THZ_TO_J               = H_PLANCK_SI*THZ
THZ_TO_CM            = 1e10/C_SI
THZ_TO_MEV          = THZ_TO_CM/8.06573