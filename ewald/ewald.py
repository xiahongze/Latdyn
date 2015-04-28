#!/usr/bin/env python

'''
This script contains an Ewald summation class that calculates the total energy 
of the system. Inputs to create this object include the reduced lattice vector
, array of charges, real and reciprocal mesh grid and the Ewald parameter. Of
course, the basis within the unit cell should be given as well. Force calculat
-ion is added as well.
P.S. 1/(4*pi*epsilon) is taken as 1. And so is the electron charge.
                                                --------Hongze Xia, 3 Jul 2013
Deserno, M., & Holm, C. (1998). How to mesh up Ewald sums. I. A theoretical and
numerical comparison of various particle mesh routines. The Journal of Chemical
Physics, 109(18), 7678. doi:10.1063/1.477414
2014-08-15: Add Fortran extension that calculates the dynamical matrix and a few
            modifications.
Mon Sep  1 17:07:57 2014: Replace loops with mesh grid
'''

import numpy as np
from numpy.linalg import inv
from scipy.special import erfc
from dyn_ewald import dyn_ewald,ewald_abcm
M_PROTON         = 1.67262178E-27   # kg
THZ              = 1.0E+12          # s^-1
M_THZ            = 1.0/M_PROTON/THZ/THZ

class Ewald(object):
    '''
    Class which calculates the total electric-static energy using the classic
    Ewald method.
    Paramters
    ---------
    lvec : lattice vector, 3 by 3 matrix, in unit of alat
    basis : basis in one unit cell
    charge : array of charges that correspond to basis
    rgrid : real space grid, integer
    kgrid : reciprocal space grid, integer
    alpha : Ewald parameter. Has a default value.
    '''
    def __init__(self, lvec, basis, charge, rgrid, kgrid, alpha=None):
        # Validation of array inputs
        self.lvec, self.bas, self.cha, self.alp = \
            map(np.array, (lvec, basis, charge, alpha))
        assert self.lvec.shape == (3,3)
        assert len(self.bas) == len(self.cha)
        assert np.sum(self.cha) == 0
        rgrid, kgrid = map(np.array, (rgrid, kgrid))
        assert rgrid.shape == (3,)
        assert kgrid.shape == (3,)
        # calculate the unit cell volume and reciprocal lattice vector
        self.v = abs(np.inner(np.cross(self.lvec[0], self.lvec[1]), self.lvec[2]))
        self.kvec = 2*np.pi*inv(lvec)
        # initiate the Ewald parameter
        if alpha == None:
            self.alp = 1.3 / np.power(self.v,1.0/3.0)
        assert self.alp >= 0.0
        # initialize the r and k mesh
        self.set_kmesh(kgrid)
        self.set_rmesh(rgrid)
        
    def _set_es(self):
        '''
        compute the self energy
        '''
        self.es = -self.alp/np.sqrt(np.pi)*np.sum(self.cha**2)
    
    def _set_ek(self):
        '''
        compute the energy contribution from reciprocal space
        '''
        ekmesh = np.zeros(len(self.kmesh))
        rho = np.zeros(len(self.kmesh),dtype=complex)
        k2 = (self.kmesh**2).sum(axis=1)
        # workaround the zero limbo
        mask = (k2<1e-4); k2 += 1.e4*mask
        for i in range(len(self.bas)):
            kr = np.dot(self.kmesh,self.bas[i])
            rho += self.cha[i]*np.exp(-1j*kr)
        ekmesh = np.abs(rho)**2/k2*np.exp(-k2/4.0/self.alp**2)
        self.ek = 2.*np.pi/self.v*ekmesh.sum()

    def _set_er(self):
        '''
        compute the energy contribution from real space
        '''
        ermesh = np.zeros(len(self.rmesh))
        for i in range(len(self.bas)):
            for j in range(len(self.bas)):
                q1 = self.cha[i]; q2 = self.cha[j]
                r = self.bas[i] - self.bas[j] + self.rmesh
                r2 = np.sum(r**2,axis=1)
                # workaround the zero limbo
                mask = (r2<1e-4); r2 += 1.e4*mask
                ermesh += q1*q2*erfc(self.alp*np.sqrt(r2))/np.sqrt(r2)
        self.er = 0.5*ermesh.sum()
        
    def get_etot(self):
        '''
        compute the total energy
        '''
        self._set_es()
        self._set_ek()
        self._set_er()
        self.etot = self.ek + self.er + self.es
        return self.etot
    
    def _gen_grid(self, grid, choice):
        n = 0
        X,Y,Z = grid
        x,y,z = np.mgrid[-X:X+1, -Y:Y+1, -Z:Z+1]
        x = x.reshape(-1); y = y.reshape(-1); z = z.reshape(-1)
        if choice == 1:
            self.rmesh = np.dot(self.lvec,(x,y,z)).T
        else:
            self.kmesh = np.dot(self.kvec,(x,y,z)).T

    def set_kmesh(self,kgrid):
        self._gen_grid(kgrid,2)
    def set_rmesh(self,rgrid):
        self._gen_grid(rgrid,1)

    def get_force(self):
        '''
        compute the forces on each ion and store them in self.force;
        forces are only due to energy from real and reciprocal space;
        no force is due to the self-energy.
        '''
        # initiate forces storage
        self.force = np.zeros(shape=(len(self.bas),3))

        for i in range(len(self.bas)):
            for j in range(len(self.bas)):
                # compute the contribution from the k-space first
                sumk = 0

                k2 = (self.kmesh**2).sum(axis=1)
                # workaround the zero limbo
                mask = (k2<1e-4); k2 += 1.e4*mask
                kr = np.dot(self.kmesh,self.bas[i]-self.bas[j])
                fk = 4*np.pi/k2*np.exp(-k2/4./self.alp**2)*np.sin(kr)
                kx,ky,kz = self.kmesh.T
                tmp = np.asarray((fk*kx,fk*ky,fk*kz))/self.v
                sumk = tmp.T.sum(axis=0)
                # print 'k contribute', sumk

                # compute the contribution from the r-space then
                sumr = 0
                r = self.bas[i] - self.bas[j] + self.rmesh
                r2 = np.sum(r**2,axis=1)
                # workaround the zero limbo
                mask = (r2<1e-4); r2 += 1.e4*mask
                fr = 2.*self.alp/np.sqrt(np.pi)*np.exp(-self.alp**2*r2)
                fr += erfc(self.alp*np.sqrt(r2))/np.sqrt(r2)
                fr /= r2
                rx,ry,rz = r.T
                tmp = np.asarray((fr*rx,fr*ry,fr*rz))
                sumr = tmp.T.sum(axis=0)
                # print 'r contribute', sumr

                # scale the force with q_j
                self.force[i] += (sumk+sumr)*self.cha[j]
            # scale the force with q_i
            self.force[i] *= self.cha[i]
        # Kindly print out the forces in good format
        for i in range(len(self.bas)):
            print "Force acts on ion %d is : %8.4f %8.4f %8.4f" % \
                (i,self.force[i][0],self.force[i][1],self.force[i][2])

        return self.force

    def get_dyn(self,mass,qvec,crys=True,mode="vffm"):
        """
        Calculate the equation of motion under harmonic approximation.
        Return the dynamical matrix at a specific q point.
        mass: python list or numpy array
              in atomic unit, with the same length as the basis
        qvec: python list or numpy array
              either in shape(3,) or shape(nks,3)
        crys: boolean (default:True)
              if true, qvec is in reciprocal lattice unit;
              otherwise, in unit of 2pi/alat
        mode: str
              either "vffm" or "abcm"
        """
        self.mass, qvec = map(np.array, (mass, qvec))
        if mode == "vffm":
            if self.mass.shape != self.cha.shape:
                raise ValueError('shape(mass) does not match!')
        elif mode == "abcm":
            pass
        else:
            raise ValueError("Wrong mode! Need to be either abcm or vffm")
        if qvec.shape == (3,): qvec = np.array([qvec])
        nks = len(qvec); N = len(self.mass)
        # self.qvec = np.array([np.dot(item,self.kvec) for item in qvec]) \
        self.qvec = np.dot(self.kvec,qvec.T).T \
            if crys else qvec*2.*np.pi
        # self.dyn = np.zeros((nks,N*3,N*3),dtype=complex)
        # for q in range(nks):
        #     if mode == "vffm":
        #         self.dyn[q] = dyn_ewald(self.bas,self.mass,self.cha,self.rmesh,\
        #                 self.kmesh,self.alp,self.v,self.qvec[q])*self.v*M_THZ
        #     elif mode == "abcm":
        #         self.dyn[q] = ewald_abcm(self.bas,self.mass,self.cha,self.rmesh,\
        #                 self.kmesh,self.alp,self.v,self.qvec[q])*self.v*M_THZ
        if mode == "vffm":
            self.dyn = dyn_ewald(self.bas,self.mass,self.cha,self.rmesh,\
                    self.kmesh,self.alp,self.v,self.qvec)*self.v*M_THZ
        elif mode == "abcm":
            self.dyn = ewald_abcm(self.bas,self.mass,self.cha,self.rmesh,\
                    self.kmesh,self.alp,self.v,self.qvec)*self.v*M_THZ
        return self.dyn
