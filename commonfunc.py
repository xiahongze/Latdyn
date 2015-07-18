#!/usr/bin/env python
import numpy as np
from numpy.linalg import inv,eigh,eig,norm
from constants import M_THZ,TPI,KB,THZ_TO_J
import pickle

def EigenSolver(m,fldata="freq_mode_dyn.pckl",herm=True):
    """
    This function returns phonon frequencies in THz
    and dump the results if fldata != None
    """
    nks = len(m)
    nval = len(m[0,0,:])
    w2 = np.zeros((nks,nval),dtype=complex)
    evec = np.zeros(m.shape,dtype=complex)
    for q in range(nks):
        tmp,evec[q] = eigh(m[q]) if herm else eig(m[q])
        w2[q] = tmp.real
        mask = (w2[q]<-1e-4); pm = mask*-1; pm[mask==False] = 1
        if mask.sum() > 0:
            print "Warning: imaginary frequency occurs at k[%d]" % q
    mask = (w2<-1e-4); pm = mask*-1; pm[mask==False] = 1
    freq = np.sqrt(np.abs(w2))/TPI*pm
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

def tetra_dos(freq,kgrid,grid,N,nstep):
    """
    Using the tetrahedron method to get DOS
    Ref:
    Blochl PRB 1994 Improved tetrahedron method for Brillouin-zone integrations
    freq: numpy ndarray
        shape: nkpt,N*3
    grid:
        MonkhorstPack grid with boundary, correspond to freq
    N: int
        number of atoms in the unit cell
    nstep: int
        total points in DOS
    kgrid: tuple of 3 integers
        Gamma centred k grid
    """

    grid0 = MonkhorstPack(kgrid)
    dos = np.zeros(nstep)
    fall = np.linspace(freq.min(),freq.max(),nstep)
    tetra = np.array([
            [0.,0.,0.],[0.,0.,1.],[0.,1.,0.],[1.,0.,1.], # 1,5,3,6
            [0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[1.,0.,1.], # 1,2,3,6
            [1.,1.,0.],[1.,0.,0.],[0.,1.,0.],[1.,0.,1.], # 4,2,3,6
            [1.,1.,0.],[1.,1.,1.],[0.,1.,0.],[1.,0.,1.], # 4,8,3,6
            [0.,1.,1.],[1.,1.,1.],[0.,1.,0.],[1.,0.,1.], # 7,8,3,6
            [0.,1.,1.],[0.,0.,1.],[0.,1.,0.],[1.,0.,1.]  # 7,5,3,6
    ])
    tetra /= kgrid
    tetra = tetra.reshape(24,1,3)
    tetra_ends = grid0+tetra # shape = 24,len(grid0),3
    tetra_ends = tetra_ends.reshape(24,-1,1,3)
    tmp = tetra_ends - grid # shape = 24,len(grid0),len(grid),3
    tmp = np.sum((tmp)**2,axis=-1) # shape = 24,len(grid0),len(grid)
    # find the indices of each tetrahedral ends
    indices = tmp.argsort()[:,:,0] # shape = 24,len(grid0)
    freq6 = freq[indices] # shape = 24,len(grid0),len(freq[0])
    freq6 = freq6.reshape((6,4,len(grid0),-1))
    freq6 = np.sort(freq6,axis=1)
    f1 = freq6[:,0,:,:]; f2 = freq6[:,1,:,:]
    f3 = freq6[:,2,:,:]; f4 = freq6[:,3,:,:]
    f21 = f2 - f1; f31 = f3 - f1; f41 = f4 - f1
    f32 = f3 - f2; f42 = f4 - f2; f43 = f4 - f3
    # get rid of dividing zeros
    mask = (f21<1e-3); f21 += mask*1e-4
    mask = (f31<1e-3); f31 += mask*1e-4
    mask = (f41<1e-3); f41 += mask*1e-4
    mask = (f32<1e-3); f32 += mask*1e-4
    mask = (f42<1e-3); f42 += mask*1e-4
    mask = (f43<1e-3); f43 += mask*1e-4
    for i in range(nstep):
        f = fall[i]
        c2 = (f<f2)*(f>=f1) # in Appendix C, Blochl PRB 1994
        c3 = (f<f3)*(f>=f2)
        c4 = (f<=f4)*(f>=f3)
        d2 = 3.*(f-f1)**2/f21/f31/f41 * c2
        d3 = (3.*f21+6*(f-f2)-3.*(f31+f42)*(f-f2)**2/f32/f42) \
                / f31/f41 * c3
        d4 = 3.*(f4-f)**2/f41/f42/f43 * c4
        dos[i] = (d2+d3+d4).sum()

    integral = np.trapz(y=dos,x=fall)
    dos *= N*3.0/integral
    return np.asarray((fall,dos))

def RemoveDuplicateRow(a):
    b = np.ascontiguousarray(a).view(np.dtype((np.void, a.dtype.itemsize * a.shape[1])))
    _, idx = np.unique(b, return_index=True)
    return a[idx]

def Reload(filename="mycalc.pickle"):
    """Reload previous calculation"""
    return pickle.load(open(filename))