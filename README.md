# Latdyn
Lattice dynamics with various semi-classical models

This project aims to create a versatile Python library that is capable of producing phonon band-structures using semi-classical vibration models, as well as other related calculations like specific heat, debye temperature, group velocity and etc. Semi-classical models are more computation efficient and suitable for large-systems like quantum dots and quantum wells, which initiated the creation of this project.

The author have created the Valence Force Field Model and the Adiabatic Bond-Charge Model for this Python library. These two models are capable of dealing with regular structures like simple cubic, face centre cube and hexagonal close pack. Ideal/Same length bonding is essential for these models to work. I had to admit this is a shortcoming. Nevertheless, the capability of these two models should not be underestimated as many superlattices are based on regular bonding bulk materials.

The Coulombic interaction is done via Ewald summation technique and due to the intensity of this calculation, fortran codes are used to ease the computational burden. To compile this fortran module, one needs to go to "Latdyn/ewald/" and run "./f2py -c -m dyn_ewald dyn_ewald.pyf dyn_ewald.f90" provided that "f2py" is made executable and you have gfortran compiler installed in your system.

To be continued.
