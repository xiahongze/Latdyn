#!/usr/bin/env python
import numpy as np
from numpy.linalg import inv,eigh,eig,norm
from constants import M_THZ,TPI,KB,THZ_TO_J

def EigenSolver(m,fldata="freq_mode_dyn.pckl"):
    """
    This function returns phonon frequencies in THz
    and dump the results if fldata != None
    """
    nks = len(m)
    nval = len(m[0,0,:])
    w2 = np.zeros((nks,nval),dtype=complex)
    evec = np.zeros(m.shape,dtype=complex)
    for q in range(nks):
        tmp,evec[q] = eigh(m[q]); w2[q] = tmp.real
        if (w2[q]<-1e-4).sum() > 0:
            print "Warning: imaginary frequency occurs at k[%d]" % q
    freq = np.sqrt(np.abs(w2))/TPI
    if fldata:
        pickle.dump((freq,evec,m),open(fldata,"wb"))
    return freq,evec

def MonkhorstPack(kgrid=(4,4,4),koff=(0,0,0),withBoundary=False):
    """
    Gamma-centred reciprocal space generator.
    with BZ boundary [the +1 feature]
    tau: ndarray, (3,3)
        reciprocal lattice vectors
    withBoundary: boolean
        Set it to True if one needs interpolations
    return:
        q-points in crystal coordinates
    """
    ki,kj,kk = kgrid; kii,kjj,kkk = koff
    if withBoundary:
        kx,ky,kz = np.mgrid[0:ki+1,0:kj+1,0:kk+1] # Double boudaries
    else:
        kx,ky,kz = np.mgrid[0:ki,0:kj,0:kk] # This is the normal BZ
    kx = kx.reshape(-1)/float(ki) + 0.5*kii/ki
    ky = ky.reshape(-1)/float(ki) + 0.5*kjj/kj
    kz = kz.reshape(-1)/float(ki) + 0.5*kkk/kk
    k_xyz = np.array((kx,ky,kz)).T
    return k_xyz
