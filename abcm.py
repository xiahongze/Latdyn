#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This is the Adiabatic Bond-Charge Model which involves three major
interactions, i.e, ion-ion bond-stretching, ion-BC bond-stretching
and BC-BC bond-bending forces.

Hongze Xia
Mon 13 Apr 2015 21:05:44 AEST
Ref:
1. Weber, W. Adiabatic bond charge model for the phonons in diamond, Si, 
    Ge, and alpha-Sn. Phys. Rev. B (1977)
2. Rajput, B. D. & Browne, D. A. Lattice dynamics of II-VI materials using
    the adiabatic bond-charge model. Phys. Rev. B 53, 9052–9058 (1996).
}
"""
import pickle
import numpy as np
from numpy.linalg import inv,eigh,eig,norm,pinv
from constants import M_THZ,TPI,KB,THZ_TO_J,THZ_TO_CM,THZ_TO_MEV
from ewald import Ewald
from commonfunc import EigenSolver,MonkhorstPack,tetra_dos
from itertools import permutations
from sys import exit
np.set_printoptions(precision=3,linewidth=200,suppress=True)

def DynBuild(basis,bvec,fc,nn,label,kpts,Ni,Mass,crys=True):
    """
    Build the dynamical matrix from the short range force constant tensors.
    basis: ndarray of shape (N,3)
    bvec: ndarray of shape (3,3)
        reciprocal lattice vectors
    fc: list of array N*[fc_i], fc_i.shape = (len(nn_i),3,3)
        force constant tensors, e.g., output from ConstructFC
        Unit ==> N/m
    nn: list of array, N*[nn_i], nn_i.shape = (len(nn_i),3)
        positions of nearest neighbours
    label: list of list, len(label) = N, len(label_i) = len(nn_i)
        label of nearest neighbours in terms of no. of basis
    kpts: ndarray
        if crys: coordinates of reciprocal lattice vectors
        else: in terms of 2pi/alat
    """
    kpts = np.array(kpts)
    if kpts.shape == (3,): kpts = np.array([kpts])
    N = len(basis); nks = len(kpts)
    dyn = np.zeros((nks,Ni*3,Ni*3),dtype=complex)
    M_1 = inv(Mass)
    # convert kpts to Cartesian coordinates if needed
    kpts = kpts.dot(bvec)*2.*np.pi if crys else kpts*2.*np.pi
    for q in range(nks):
        dyn1 = np.zeros((N*3,N*3),dtype=complex) # temperoary
        for i in range(N):
            for j in range(len(nn[i])):
                # With which atom?
                x = basis[i]-nn[i][j]; ka = label[i][j]
                # ON-diagonal
                dyn1[i*3:i*3+3,i*3:i*3+3] -= fc[i][j]
                # OFF-diagonal
                dyn1[i*3:i*3+3,ka*3:ka*3+3] += fc[i][j]*np.exp(-1j*np.dot(kpts[q],x))
        # ABCM matrices
        R = dyn1[0:Ni*3,0:Ni*3]; S = dyn1[Ni*3:,Ni*3:]
        T = dyn1[0:Ni*3,Ni*3:]; Ts = dyn1[Ni*3:,0:Ni*3]
        # ABCM operation
        tmp = np.dot(inv(S),Ts); dyn[q] = np.dot(M_1, R - np.dot(T,tmp))
    # scale it to SI unit, omega^2 not frequency^2 after diagonalisation
    return dyn*M_THZ

class ABCM(object):
    """
    Adiabatic Bond-Charge takes inputs:
    lvec: ndarray, (3,3)
        lattice vectors in unit of alat
    basis: ndarray, (N,3)
        base atoms
    mass: array, (N_ion,)
        in atomic unit
    symbol: string array or list, (N,)
        Element names of the basis, e.g., ["Si","Si","BC","BC","BC","BC"]
    """
    def __init__(self,lvec,basis,mass,symbol):
        self.lvec,self.bas,self.mass,self.symbol =\
                    map(np.array,(lvec,basis,mass,symbol))
        assert self.lvec.shape == (3,3)
        assert len(self.bas) == len(symbol)

        # initialise the force constant tensors
        self.fc = []
        # initialise the k-points
        self.kpts = []
        # initialise the dynamical matrix
        self.dyn = []
        # initialise the dispersions
        self.freq = []
        # initialise dos
        self.dos = []
        # set the reciprocal lattice, transpose is necessary
        self.bvec = inv(self.lvec).T
        # number of basis
        self.N = len(self.bas)
        # work out the number of ions and BCs
        self.isBC = (self.symbol=="BC")
        self.N_bc = np.sum(self.isBC)
        self.N_ion = self.N - self.N_bc
        # number of nn for ion and BC, 8 for tetrahedral
        self.nn_cut = self.N_bc/self.N_ion*4
        assert len(mass) == self.N_ion
        a = []
        for i in range(self.N_ion):
            a += [self.mass[i]]*3
        self.Mass = np.diag(a) # mass matrix
        # number of bands
        self.nbnd = 3*self.N_ion
        # unit cell volume in terms of alat
        self.v = abs(np.inner(np.cross(self.lvec[0], self.lvec[1]), self.lvec[2]))
        # find and set the n.n
        self.set_nn()
        # the total number of first n.n.
        self.n1 = [len(self.nn[i]) for i in range(self.N) ]
        # initiate Ewald object
        self.set_ewald()

    def set_ewald(self,charge=[],eps=None,rgrid=[3,3,3],kgrid=[3,3,3]):
        """
        ecalc: Ewald object
            Needed when one wants to have this long range interaction
        eps: float
            Prefactor for this interaction. It is a constant in unit
            of e^2/(4PI*epsilon*alat**3)
        rgrid,kgrid: list
            Meshes for Ewald calculation.
        """
        if (charge != [] and eps == None) or (charge == [] and eps != None):
            raise ValueError("Both charge and eps should be given!")
        if charge == []:
            self.ecalc = None; self.eps = None
        else:
            self.ecalc = Ewald(lvec=self.lvec, basis=self.bas, charge=charge, \
                    rgrid=rgrid, kgrid=kgrid)
            self.eps = eps

    def set_nn(self,scope=[1,1,1],nmax=20,dist2=None,showDist=False):
        """
        Set the n.n. for the system. No need to call manually if one only
        need the first n.n. for the calculation. Used when one like to inspect
        and incorporate the second n.n.
        scope: list or tuple of 3
            The search scope for n.n.. Default is generally good enough. it could
            be increased to [2,2,2].
        nmax: integer
            The number of n.n. to search for/compare with. Increase it when
            one wants to inspect higher n.n.
        dist2: float
            the bond length of 2nd n.n.. It is recommended to set a value that
            is slightly larger than the bond length. Use this option to enable
            2nd n.n. interactions.
        showDist: boolean
            To inspect all bond-lengths of all n.n. upto nmax. Recommended
            for the first run.
        """
        X,Y,Z = scope
        x,y,z = np.mgrid[-X:X+1, -Y:Y+1, -Z:Z+1]
        x = x.reshape(-1); y = y.reshape(-1); z = z.reshape(-1)
        xyz = np.asarray((x,y,z)).T; rgrid = xyz.dot(self.lvec) # fix this hidden bug
        # allatoms = []
        # for item in self.bas:
        #     allatoms = item+rgrid if allatoms == [] \
        #         else np.vstack((allatoms,item+rgrid))
        # the most efficient way to get allatoms
        allatoms = (self.bas.reshape(-1,1,3)+rgrid).reshape(-1,3)
        self.label = []; self.nn = []; self.nndist = []
        for i in range(self.N):
            dist = norm(self.bas[i]-allatoms,axis=1)
            nn_ind = dist.argsort()[1:nmax] # consider the first nmax n.n.
            if showDist: print dist[nn_ind]
            nn_cut0 = self.nn_cut
            if self.nn_cut>nmax-1:
                # you probably have interface problems
                print "You might need to inspect the nearest neighbours \
                and need to set a reasonable dist2."
                nn_cut0 = nmax-1

            if dist2 == None:
                first_nn_ind = nn_ind[0:nn_cut0]
            else:
                # if dist2 exisits, count till that distance
                first_nn_ind = nn_ind[dist[nn_ind]<=dist2]

            label = np.array(first_nn_ind)/len(rgrid)
            nn = allatoms[first_nn_ind]
            nndist = dist[first_nn_ind]
            self.label.append(label)
            self.nn.append(nn)
            self.nndist.append(nndist)
        self.nnsymb = []
        for i in range(self.N):
            self.nnsymb.append(self.symbol[self.label[i]])

    def set_fc(self,fc_dict):
        '''
        fc_dict involves alpha and beta for different interactions between atoms/BCs.
        Make sure one now exactly how many interactions are involved by looking
        at the n.n for each atom in the basis. i.e., calc.get_nn_label()
        fc_dict must have this kind of format:
        fc_dict = {
        "alpha":a_dict,"beta":b_dict,"sigma":s_dict
        }
        a_dict and b_dict look like:
        a_dict = {
        "Ga-As": 10.0, "Al-As": 11.0, "Ga-BC":1.0
        }
        b_dict = {
        "BC-Ga":10., "BC-As": 20.
        }
        s_dict = {
        "BC-Ga":10., "BC-As": 20.
        }
        The last one with a trailing 2 indicates a second n.n. interactions.
        '''
        self.fc = []
        # initialise the FCs
        try:
        # if fc_dict.has_key("alpha"):
            a_dict = fc_dict["alpha"]
            self.na = len(a_dict)
            self.akeys = a_dict.keys()
            self.avalues = a_dict.values()
        # if fc_dict.has_key("beta"):
            b_dict = fc_dict["beta"]
            self.nb = len(b_dict)
            self.bkeys = b_dict.keys()
            self.bvalues = b_dict.values()
        # if fc_dict.has_key("sigma"):
            s_dict = fc_dict["sigma"]
            self.ns = len(s_dict)
            self.skeys = s_dict.keys()
            self.svalues = s_dict.values()
        except KeyError:
            print "Warning: some keys have been skipped."

        self.fc_dict = fc_dict

        for i in range(self.N):
            onsite = self.symbol[i]
            offsite = self.nnsymb[i]
            nos = len(offsite)
            # initialise FC tensors
            FC = np.zeros((nos,3,3))
            for j in range(nos):
                sig = 0.0
                key1 = onsite+"-"+offsite[j]
                key2 = offsite[j]+"-"+onsite
                if a_dict.has_key(key1):
                    alp = a_dict[key1]
                    if 's_dict' in locals():
                        if s_dict.has_key(key1): sig = s_dict[key1]
                    FC[j] += self.get_centre_fc(alp,i,j,sig)
                elif a_dict.has_key(key2):
                    alp = a_dict[key2]
                    if 's_dict' in locals():
                        if s_dict.has_key(key2): sig = s_dict[key2]
                    FC[j] += self.get_centre_fc(alp,i,j,sig)

                for k in range(nos):
                    if k != j:
                        inline = (norm(np.cross(self.bas[i]-self.nn[i][j], self.bas[i]-self.nn[i][k]))<1e-6)
                        if onsite != "BC" and offsite[j]!="BC" and offsite[k]=="BC" and inline: # cross-stretching
                            key1 = onsite+"-"+"BC"+"-"+offsite[j]
                            key2 = offsite[j]+"-"+"BC"+"-"+onsite
                            if a_dict.has_key(key1):
                                alp = a_dict[key1]; FC[j] += self.get_noncentre_fc(alp,i,j,k,1)
                            elif a_dict.has_key(key2):
                                alp = a_dict[key2]; FC[j] += self.get_noncentre_fc(alp,i,j,k,1)
                            # else:
                            #     raise ValueError("Missing interaction: "+key1)
                        elif onsite != "BC" and offsite[j]=="BC" and offsite[k]!="BC" and inline: # cross-stretching
                            key1 = onsite+"-"+"BC"+"-"+offsite[k]
                            key2 = offsite[k]+"-"+"BC"+"-"+onsite
                            if a_dict.has_key(key1):
                                alp = a_dict[key1]; FC[j] += self.get_noncentre_fc(alp,i,j,k,2)
                            elif a_dict.has_key(key2):
                                alp = a_dict[key2]; FC[j] += self.get_noncentre_fc(alp,i,j,k,2)
                            # else:
                            #     raise ValueError("Missing interaction: "+key1)
                        elif onsite == "BC" and offsite[j]!="BC" and offsite[k]!="BC" and inline: #cross-stretching
                            key1 = offsite[j]+"-"+"BC"+"-"+offsite[k]
                            key2 = offsite[k]+"-"+"BC"+"-"+offsite[j]
                            if a_dict.has_key(key1):
                                alp = a_dict[key1]; FC[j] += self.get_noncentre_fc(alp,i,j,k,3)
                            elif a_dict.has_key(key2):
                                alp = a_dict[key2]; FC[j] += self.get_noncentre_fc(alp,i,j,k,3)
                            # else:
                            #     raise ValueError("Missing interaction: "+key1)
                        elif onsite != "BC" and offsite[j]=="BC" and offsite[k]=="BC": # BC-bond-bending
                            r1 = self.bas[i]-self.nn[i][j]; r2 = self.bas[i]-self.nn[i][k]
                            if abs(norm(r1)-norm(r2))<1e-4: # same bond length
                                sig = 0.0
                                key1 = "BC"+"-"+onsite
                                key2 = onsite+"-"+"BC"
                                if b_dict.has_key(key1):
                                    bet = b_dict[key1]; key1 += "-BC"
                                    if 's_dict' in locals():
                                        if s_dict.has_key(key1): sig = s_dict[key1]
                                    FC[j] += self.get_bending_fc(bet,i,j,k,1,sig)
                                elif b_dict.has_key(key2):
                                    bet = b_dict[key2]; key2 = "BC-"+key2
                                    if 's_dict' in locals():
                                        if s_dict.has_key(key2): sig = s_dict[key2]
                                    FC[j] += self.get_bending_fc(bet,i,j,k,1,sig)
                                # else:
                                #     raise ValueError("Missing interaction: "+key1)
                        elif onsite == "BC" and offsite[j]!="BC" and offsite[k]=="BC": # BC-bond-bending
                            r1 = self.nn[i][j]-self.bas[i]; r2 = self.nn[i][j]-self.nn[i][k]
                            if abs(norm(r1)-norm(r2))<1e-4: # same bond length
                                sig = 0.0
                                key1 = "BC"+"-"+offsite[j]
                                key2 = offsite[j]+"-"+"BC"
                                if b_dict.has_key(key1):
                                    bet = b_dict[key1]; key1 += "-BC"
                                    if 's_dict' in locals():
                                        if s_dict.has_key(key1): sig = s_dict[key1]
                                    FC[j] += self.get_bending_fc(bet,i,j,k,2,sig)
                                elif b_dict.has_key(key2):
                                    bet = b_dict[key2]; key2 = "BC-"+key2
                                    if 's_dict' in locals():
                                        if s_dict.has_key(key2): sig = s_dict[key2]
                                    FC[j] += self.get_bending_fc(bet,i,j,k,2,sig)
                                # else:
                                #     raise ValueError("Missing interaction: "+key1)
                        elif onsite == "BC" and offsite[j]=="BC" and offsite[k]!="BC": # BC-bond-bending
                            r1 = self.nn[i][k]-self.bas[i]; r2 = self.nn[i][k]-self.nn[i][j]
                            if abs(norm(r1)-norm(r2))<1e-4: # same bond length
                                sig = 0.0
                                key1 = "BC"+"-"+offsite[k]
                                key2 = offsite[k]+"-"+"BC"
                                if b_dict.has_key(key1):
                                    bet = b_dict[key1]; key1 += "-BC"
                                    if 's_dict' in locals():
                                        if s_dict.has_key(key1): sig = s_dict[key1]
                                    FC[j] += self.get_bending_fc(bet,i,j,k,3,sig)
                                elif b_dict.has_key(key2):
                                    bet = b_dict[key2]; key2 = "BC-"+key2
                                    if 's_dict' in locals():
                                        if s_dict.has_key(key2): sig = s_dict[key2]
                                    FC[j] += self.get_bending_fc(bet,i,j,k,3,sig)
                                # else:
                                #     raise ValueError("Missing interaction: "+key1)
                        elif onsite!="BC" and offsite[j]!="BC" and offsite[k]!="BC": # ion-bod-bending
                            r1 = self.bas[i]-self.nn[i][j]; r2 = self.bas[i]-self.nn[i][k]
                            if abs(norm(r1)-norm(r2))<1e-4: # same bond length
                                keys = list(permutations([onsite,offsite[j],offsite[k]]))
                                for item in keys:
                                    key = item[0]+"-"+item[1]+"-"+item[2]
                                    if b_dict.has_key(key):
                                        bet = b_dict[key]; FC[j] += self.get_bending_fc(bet,i,j,k,1)
                                        break
                        else:
                            pass
                #
            #
            self.fc.append(FC)
        # self.fix_interface()

    def get_centre_fc(self,a,i,j,s=0):
        """
        generate ion-ion centre force constant tensor
        a: float, force constant
        i: integer, onsite ion index
        j: integer, offsite ion index
        """
        dvec = (self.bas[i]-self.nn[i][j]); d = norm(dvec)
        one = np.ones((3,3)); tmp = dvec*one
        return -np.multiply(tmp,tmp.T)*a/d/d*8 + s*np.identity(3)

    def get_bending_fc(self,b,i,j,k,ionsite,s=0):
        one = np.ones((3,3))
        if ionsite == 1: # i sits an ion
            dvec1 = (self.bas[i]-self.nn[i][j]); d1 = norm(dvec1)
            dvec2 = (self.bas[i]-self.nn[i][k]); d2 = norm(dvec2)
            tmp1 = (dvec1+dvec2)*one; tmp2 = -dvec2*one
            return 2*np.multiply(tmp1.T,tmp2)*b/d1/d2 - s*np.identity(3)
        elif ionsite == 2: # j sits an ion
            dvec1 = self.nn[i][j]-self.bas[i]; d1 = norm(dvec1)
            dvec2 = self.nn[i][j]-self.nn[i][k]; d2 = norm(dvec2)
            tmp1 = (dvec1+dvec2)*one; tmp2 = -dvec2*one
            return 2*np.multiply(tmp1,tmp2.T)*b/d1/d2 - s*np.identity(3)
        elif ionsite == 3: # k sits an ion
            dvec1 = self.bas[i]-self.nn[i][k]; d1 = norm(dvec1) # BC-ION
            dvec2 = self.nn[i][j]-self.nn[i][k]; d2 = norm(dvec2) # BC-ION
            tmp1 = one*dvec1; tmp2 = (one*dvec2)
            return 2*np.multiply(tmp1.T,tmp2)*b/d1/d2 + s*np.identity(3)
        else:
            return 0

    def get_noncentre_fc(self,a,i,j,k,bcsite):
        one = np.ones((3,3))
        if bcsite == 1: # k sits a BC
            dvec1 = (self.bas[i]-self.nn[i][k]); d1 = norm(dvec1)
            dvec2 = (self.nn[i][j]-self.nn[i][k]); d2 = norm(dvec2)
            tmp1 = one*dvec1; tmp2 = (one*dvec2)
            return 4*tmp1*tmp2.T*a/d1/d2
        if bcsite == 2: # j sits a BC
            dvec1 = (self.bas[i]-self.nn[i][j]); d1 = norm(dvec1)
            dvec2 = (self.nn[i][k]-self.nn[i][j]); d2 = norm(dvec2)
            tmp1 = one*dvec1; tmp2 = (one*dvec2)
            return -4*tmp1*tmp2.T*a/d1/d2
        if bcsite == 3: # i sits a BC
            dvec1 = (self.bas[i]-self.nn[i][j]); d1 = norm(dvec1)
            dvec2 = (self.bas[i]-self.nn[i][k]); d2 = norm(dvec2)
            tmp1 = one*dvec1; tmp2 = (one*dvec2)
            return -4*tmp1.T*tmp2*a/d1/d2
        else:
            return 0

    def fix_interface(self):
        # Average the interface connection for set_fc()
        for i in range(self.N):
            print "Checking atom ",i
            for j in range(len(self.nn[i])):
                bond0 = self.bas[i] - self.nn[i][j]
                offsiteLabel = self.label[i][j]
                for k in range(len(self.nn[offsiteLabel])):
                    bond1 = self.bas[offsiteLabel] - self.nn[offsiteLabel][k]
                    if np.allclose(bond0,-bond1): # the same bond
                        tmp = self.fc[i][j] + self.fc[offsiteLabel][k].T
                        if np.allclose(self.fc[i][j],self.fc[offsiteLabel][k].T):
                            print self.symbol[i]," with ", self.nnsymb[i][j], " is fine"
                        else:
                            print self.symbol[i]," with ", self.nnsymb[i][j], " is not fine"
                            print self.fc[i][j]-self.fc[offsiteLabel][k].T
                        tmp *= 0.5
                        self.fc[i][j] = tmp
                        self.fc[offsiteLabel][k] = tmp.T

    def set_kpts(self,kpts,crys=True):
        """
        set k-points for the calculation.
        kpts: ndarray, or list
            for single kpt: kpt = [kx,ky,kz]
            for more than one kpts: [kpt1,kpt2]
        crys: boolean
            True, for crystal coordinates [default]
            False, in unit of 2pi/alat
        """
        kpts = np.array(kpts)
        if kpts.shape == (3,):
            self.kpts = np.array([kpts])
        elif kpts.ndim == 2:
            if len(kpts[0,:]) == 3:
                self.kpts = kpts
        else:
            raise ValueError("Wrong kpts dimension!")
        self.iskcrys = crys
        self.nkpt = len(self.kpts)

    def k_conv(self,towhat=None):
        """
        towhat: string
            either "crys" or "cart"
        """
        if self.kpts == []: raise ValueError("You have not set kpoints yet!")
        if towhat == "crys" or towhat == "CRYS":
            if self.iskcrys: pass
            iskcrys = True; convec = self.lvec
        elif towhat == 'cart' or towhat == 'CART':
            if self.iskcrys == False: pass
            iskcrys = False; convec = self.bvec
        else:
            print "Wrong input for k-conversion!"
            pass
        self.kpts = self.kpts.dot(convec)
        # for i in range(self.nkpt):
        #     self.kpts[i] = np.dot(self.kpts[i],convec)
        self.iskcrys = iskcrys
        pass

    def __set_dyn(self):
        """
        set the short range dynamical matrix
        """
        if self.fc == []:
            raise ValueError("Force constants not set yet!")
        elif self.kpts == []:
            raise ValueError("Kpts not set yet!")
        self.dyn = DynBuild(self.bas,self.bvec,self.fc,\
                self.nn,self.label,self.kpts,self.N_ion,self.Mass,crys=self.iskcrys)
        if self.ecalc != None:
            self.dyn += self.eps*self.ecalc.get_dyn(self.mass,self.kpts,crys=self.iskcrys,mode="abcm")

    def get_dyn(self):
        """
        get the dynamical matrix if you have called:
        calc.get_ph_disp() before
        return m[nkpt] of shape (nkpt,3*N,3*N)
        """
        if self.dyn != []:
            return self.dyn
        else:
            raise ValueError("No dynamical matrix!")

    def get_ph_disp(self):
        """
        get the phonon band structure of the system provided all
        force constans and kpts are specified. Return phonon frequencies.
        """
        self.__set_dyn()
        if self.freq == []:
            self.freq,self.evec = EigenSolver(self.dyn,fldata=None,herm=False)
        return self.freq

    def get_nn_label(self):
        """
        Neatly print out all nearest neighbours for each base atom.
        Recommended to use when one is not sure about the interactions.
        """
        print "The nearest neighbours for all basis atoms:"
        for i in range(self.N):
            atom = self.bas[i]
            n = len(self.nn[i])
            print "Atom %d: %s at [%8.4f %8.4f %8.4f ] has %d n.n." \
                % (i,self.symbol[i],atom[0],atom[1],atom[2],n)
            nn = self.nn[i]
            symbol = self.nnsymb[i]
            nndist = self.nndist[i]
            label = self.label[i]
            for j in range(n):
                print "     %d. %s [%8.4f %8.4f %8.4f ] bond:%8.4f label %d" \
                % (j,symbol[j],nn[j,0],nn[j,1],nn[j,2],nndist[j],label[j])

    def get_dos(self,nstep=251,kgrid=(4,4,4)):
        """
        Get the Density of States out of VFFM. Autosave to "dos.csv".
        nstep: int
            total points in DOS
        kgrid: tuple of 3 integers
            Gamma centred k grid
        return: tuple
            freq,DOS
        Ref:
        Blochl PRB 1994 Improved tetrahedron method for Brillouin-zone integrations
        """
        if self.fc == []:
            raise ValueError("Force constants not set yet!")

        grid = MonkhorstPack(kgrid,withBoundary=True)
        self.set_kpts(grid,crys=True)
        freq = np.sort(self.get_ph_disp())

        self.dos = tetra_dos(freq,kgrid,grid,self.N,nstep)
        np.savetxt("dos.txt", self.dos.T, delimiter="\t",fmt="%10.5f")
        return self.dos

    def plot(self,mytup=None,filename=None):
        """
        Plot the band structure. Make sure one choose a k-path.
        Use this only for the purpose of checking.
        mytup: tuple
            (point_names, x, X) means name of the special points,
            all reciprocal points, special points.
            if given, auto-save to "ph.pdf"
        """
        if self.kpts == [] or self.dyn == []:
            print "You have nothing to plot!"
            pass
        import matplotlib.pyplot as plt
        if mytup != None:
            point_names, x, X = mytup
        freq = np.sort(self.freq)
        plt.figure(figsize=(10, 6))
        q = range(self.nkpt) if mytup == None else x
        for n in range(self.nbnd):
            plt.plot(q,freq[:,n],'k-')
        ymin = freq.min(); ymax = freq.max()+1.0
        plt.xlim(q[0], q[-1])
        plt.ylim(ymin, ymax)
        plt.xlabel("Reduced wave number", fontsize=18)
        plt.ylabel("Frequency (THz)", fontsize=18)
        plt.tick_params(axis='x', labeltop='off',labelbottom='off')
        if mytup != None:
            plt.tick_params(axis='x', labeltop='on',labelsize=15,labelbottom='off')
            # plot vertical lines at special points
            for p in X:
                plt.plot([p, p], [ymin, ymax], 'k--')
            plt.xticks(X, point_names)
        # plt.grid('on')
        if filename == None and mytup != None:
            filename = "ph.pdf"
        if filename != None:
            plt.savefig(filename,dpi=300)
        plt.show()
        del plt

    def plot_dos(self,filename="dos.pdf"):
        """Plot DOS and save it to file."""
        if self.dos!=[]:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(8,5))
            w,d = self.dos
            plt.plot(w,d,'k-',lw=1)
            plt.fill(w,d,color='lightgrey')
            plt.xlim(w.min(), w.max())
            plt.xlabel("Frequency (THz)",fontsize=18)
            plt.ylabel("DOS",fontsize=18)
            if filename != None:
                plt.savefig(filename,dpi=300)
            plt.show()
            del plt
        else:
            print "You have not done a DOS calc."

    def save(self,filename="myabcm.pickle"):
        """
        Save the whole object to Python pickle.
        filename: string
        """
        pickle.dump(self,open(filename,"wb"))

    def save_ph_csv(self,phunit="THz",kpt="cart"):
        """
        Save the phonon dispersions and k-points to two CSV files.
        phunit: frequency unit, "THz", "cm-1", "thz", "mev", "cm", "meV"
        kpt: kpts unit, "cart" or "crys"
        """
        if self.kpts == [] or self.dyn == []:
            exit("You have nothing to save!")
        if phunit.lower() == "thz":
            freq = np.sort(self.freq)
        elif phunit.lower() == "mev":
            freq = np.sort(self.freq)*THZ_TO_MEV
        elif phunit.lower() == "cm-1" or phunit.lower() == "cm":
            freq = np.sort(self.freq)*THZ_TO_CM
        else:
            exit(phunit+" is not supported!")
        kp = self.gen_kpath()
        dt = np.hstack((kp.reshape(-1,1),freq))
        np.savetxt("phfreq.txt", dt, delimiter="\t",fmt="%10.5f",
        header="Phonon frequency in unit of "+phunit)
        self.k_conv(kpt)
        np.savetxt("phkpath.txt", self.kpts, delimiter="\t",fmt="%8.3f",
        header="k-path in unit of "+kpt)

    def gen_kpath(self):
        self.k_conv("cart")
        tmp1 = self.kpts[1:]-self.kpts[:-1]
        tmp2 = norm(tmp1,axis=1)
        tmp3 = np.zeros(len(tmp2)+1)
        for i in range(len(tmp2)):
            tmp3[i+1] =tmp3[i]+tmp2[i]
        return tmp3

    def get_debye(self,temp,alat=None,kgrid=(4,4,4),koff=(1,1,1)):
        """
            Calculate the Debye specific heat.
            This method auto saves a plot "debye.pdf" and a CSV file "debye.csv"
            temp: array
                In unit Kelvin.
            alat: float
                lattice constant in angstrom
            kgrid,koff: tuple
                MonkhorstPack grid
            return: array
                the specific heat in unit J K^-1 cm^-3
        """
        if alat == None: raise ValueError("What is your lattice constant in angstrom?")
        temp = np.asarray(temp)
        # do a phonon calculation on the mesh
        grid = MonkhorstPack(kgrid,koff)
        self.set_kpts(grid,crys=True)
        self.get_ph_disp()
        ph_e = self.freq.flatten()
        mask = ph_e<0.01 # truncated at 0.01THz
        ph_e += mask.astype(int)*0.01
        ph_e *= THZ_TO_J
        kbt = KB*temp
        debye = np.zeros(len(kbt))
        for i in range(len(kbt)):
            exp_ph_kbt = np.exp(-ph_e/kbt[i])
            tmp = (ph_e/kbt[i])**2 * exp_ph_kbt / (1.0-exp_ph_kbt)**2
            debye[i] = tmp.sum()
        debye *= KB/self.nkpt/self.v/alat**3 * 1.0e24
        np.savetxt("debye.csv",(temp,debye),delimiter=",",fmt="%10.5f")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8,5))
        plt.plot(temp,debye,'k-',lw=1)
        plt.xlim(temp.min(), temp.max())
        plt.grid('on')
        plt.xlabel("Temperature (K)",fontsize=18)
        plt.ylabel('Specific heat ($J K^{-1} cm^{-3}$)',fontsize=18)
        plt.savefig("debye.pdf",dpi=300)
        del plt

    def get_group_v(self,kpts,direction='x',crys=True,dk=None):
        """
        return the phonon group velocity at q point.
        kpts: ndarray
        direction: character
            "x","y","z"
        crys: boolean
        dk: list of 3 float
            the offset in reciprocal space in unit of 2pi/alat
            the modulo of dk should not be too small
        return: vg[nkpt,nbnd] = dw/dk
            in unit of alat*THz
        """
        if dk == None:
            if direction == 'x' or direction == 'X':
                dk = [1e-2,0.0,0.0]
            elif direction == 'y' or direction == 'Y':
                dk = [0.0,1e-2,0.0]
            elif direction == 'z' or direction == 'Z':
                dk = [0.0,0.0,1e-2]
            else:
                raise ValueError("Wrong direction!")
        else:
            dk = np.asarray(dk)
            assert len(dk) == 3
        # convert to 2pi/alat kpoints
        self.set_kpts(kpts,crys=crys) # necessary
        self.k_conv(towhat='cart')
        kpts = np.copy(self.kpts)
        self.set_kpts(kpts+dk,crys=False)
        self.get_ph_disp(); freq1 = np.sort(self.freq)
        self.set_kpts(kpts,crys=False)
        self.get_ph_disp(); freq0 = np.sort(self.freq)
        df = (freq1-freq0)
        return df/norm(dk)

    def get_therm_cond(self,temp,alat=None,x='x',y='x',tau=1.,kgrid=(4,4,4),koff=(1,1,1)):
        """
            Calculate the Debye specific heat.
            This method auto saves a plot "thermal_cond_K.pdf" and a CSV file "kappa.csv"
            temp: array
                In unit Kelvin.
            alat: float
                lattice constant in angstrom
            x,y: character
                first and second direction in Boltzmann transport function
            tau: float
                averaged phonon relaxation time in ps
            kgrid,koff: tuple
                MonkhorstPack grid
            return: array
                the specific heat in unit J K^-1 m^-3
        """
        if alat == None: raise ValueError("What is your lattice constant in angstrom?")
        # do a phonon calculation on the mesh
        grid = MonkhorstPack(kgrid,koff)
        v0 = self.get_group_v(grid,direction=x,crys=True)
        v1 = self.get_group_v(grid,direction=y,crys=True)
        ph_e = np.sort(self.freq*THZ_TO_J) # in J
        kbt = KB*temp
        kappa = np.zeros(len(kbt))
        # d(Bose-Einstein)/dT
        for i in range(len(temp)):
            exp_ph_kbt = np.exp(-ph_e/kbt[i])
            tmp = kbt[i]*temp[i]*(1.-exp_ph_kbt)**2
            dnb = ph_e * exp_ph_kbt/tmp
            tmp = v0*v1*ph_e*dnb
            kappa[i] = tmp.sum()
        kappa *= tau/self.nkpt/self.v/alat*1e22 # in J/s/m/K
        np.savetxt("kappa_"+x+y+".csv",(temp,kappa),delimiter=",",fmt="%10.5f")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8,5))
        plt.plot(temp,kappa,'k-',lw=1)
        plt.xlim(temp.min(), temp.max())
        plt.grid('on')
        plt.xlabel("Temperature (K)",fontsize=18)
        plt.ylabel('Thermal conductivity $\kappa_{' +x+y+ '}$ W/(K m)',fontsize=18)
        plt.savefig("therm_cond_K_"+x+y+".pdf",dpi=300)
        del plt

    def fit_freq(self,src_freq,kpts,fc_dict,eps0=1.,crys=True,method='Powell',maxiter=100):
        """
        Fit the model to a given dispersion based on frequencies.
        src_freq: ndarray
            Please make sure ALL frequencies are included, e.g., degenerated ones and zeros
        kpts: ndarray
            kpts for source frequencies.
        fc_dict: dictionary
            initial guess of the short ranged force constants
        crys: boolean
            whether the kpts are in crystal coordinates or 2pi/alat
        method: string
            minimisation algorithm. Options could be any below:
            'Nelder-Mead','Powell','BFGS','Newton-CG'
        """
        from scipy.optimize import minimize
        np.set_printoptions(precision=3)
        np.set_printoptions(suppress=True)

        # initialise x0
        if fc_dict.has_key('sigma'):
            x0 = np.array(fc_dict['alpha'].values()+fc_dict['beta'].values()+fc_dict['sigma'].values())
        else:
            x0 = np.array(fc_dict['alpha'].values()+fc_dict['beta'].values())
        # add eps if needed
        if self.ecalc != None:
            x0 = np.hstack((x0,eps0))
            self.m_ewald = self.ecalc.get_dyn(self.mass,kpts,crys=crys,mode="abcm")
        
        self.set_fc(fc_dict)

        self.set_kpts(kpts,crys=crys)
        self.src_freq = np.sort(src_freq)
        assert len(self.src_freq) == self.nkpt
        if self.ecalc != None:
            res = minimize(self.__fit_ewald,x0=x0,method=method,options={"maxiter":maxiter})
            # res = minimize(self.__fit_ewald,x0=x0,method=method,options={"maxiter":maxiter},
            # bounds=((0.01,100),(0.01,100),(0.01,100),(0.01,100),(0,10)))
        else:
            res = minimize(self.__fit_no_ewald,x0=x0,method=method)

        self.__log_fit(res,filename="log_fit.txt")
        del minimize

    def __fit_no_ewald(self,fc):
        fc = np.abs(fc)
        afc = fc[:self.na]; bfc = fc[self.na:]
        for i in range(self.na):
            self.fc_dict['alpha'][self.akeys[i]] = afc[i]
        print "alpha: ", self.fc_dict['alpha']
        for i in range(self.nb):
            self.fc_dict['beta'][self.bkeys[i]] = bfc[i]
        print "beta: ", self.fc_dict['beta']
        self.set_fc(self.fc_dict)
        freq = np.sort(self.get_ph_disp())
        return ((freq-self.src_freq)**2).sum()/len(freq)

    def __fit_ewald(self,x0):
        # x0 = abs(x0)
        fc = x0[:-1]; self.eps = x0[-1]
        afc = fc[:self.na]; bfc = fc[self.na:self.na+self.nb]; sfc = fc[self.na+self.nb:]
        if self.fc_dict.has_key('alpha'):
            for i in range(self.na):
                self.fc_dict['alpha'][self.akeys[i]] = afc[i]
            print "alpha: ", self.fc_dict['alpha']
        if self.fc_dict.has_key('beta'):
            for i in range(self.nb):
                self.fc_dict['beta'][self.bkeys[i]] = bfc[i]
            print "beta: ", self.fc_dict['beta']
        if self.fc_dict.has_key('sigma'):
            for i in range(self.ns):
                self.fc_dict['sigma'][self.skeys[i]] = sfc[i]
            print "sigma: ", self.fc_dict['sigma']
        print "eps = %10.5f" % self.eps
        self.set_fc(self.fc_dict)
        dyn0 = DynBuild(self.bas,self.bvec,self.fc,\
                self.nn,self.label,self.kpts,self.N_ion,self.Mass,crys=self.iskcrys)
        dyn = self.eps*self.m_ewald + dyn0
        freq,evec = EigenSolver(dyn,fldata=None,herm=False)
        self.freq = np.sort(freq)
        return ((self.freq-self.src_freq)**2).sum()/len(freq)

    def __log_fit(self,res,filename="logfit.txt"):
        # Log the fitting results
        import time
        logfile = open(filename,"w")
        self.freq = np.sort(self.freq)
        print >>logfile,"Log: ",time.strftime("%Y-%m-%d %H:%M")
        print >>logfile,"Current system:"
        for i in range(self.N):
            print >>logfile,"%s %8.3f %8.3f %8.3f" % ((self.symbol[i],)+tuple(self.bas[i]))
        print >>logfile,"Unit cell dimension:"
        for vec in self.lvec: print >>logfile,"%8.4f"*3 % tuple(vec)
        print >>logfile,"Fitting routine returns: state (%s)" % res.success
        ###### Print out FCs ###########
        print >>logfile,"fc_dict =", self.fc_dict
        if  self.ecalc != None:
            print >>logfile,"eps = %10.5f" % self.eps
        ###### End of Print out FCs #####
        print >>logfile,"With message: %s" % res.message
        print >>logfile,"The system is fitted to the frequencies below: crys = %s" % self.iskcrys
        for i in range(self.nkpt):
            tup = (tuple(self.kpts[i])+tuple(self.src_freq[i]))
            fmt = "SRC k = [" + "%8.3f "*3 + "], [" + "%8.4f "*self.nbnd + "]"
            print >>logfile, fmt % tup
            tup = (tuple(self.kpts[i])+tuple(self.freq[i]))
            fmt = "FIT k = [" + "%8.3f "*3 + "], [" + "%8.4f "*self.nbnd + "]"
            print >>logfile, fmt % tup
        sqrt_k = ((self.freq-self.src_freq)**2).sum()/len(self.freq)
        print >>logfile,"Fitting error %8.4f per kpt and %7.4f per state [THz]^2" % (sqrt_k,sqrt_k/self.nbnd)
        del time
        logfile.close()
