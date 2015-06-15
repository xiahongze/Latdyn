#!/usr/bin/env python
"""
This is a simplified Valence Force Field Model which involves two
major interactions, i.e, bond-stretching and bond-bending forces. The
author found that the VFFM described in one PRB paper contains
unnecessary interactions that don't contribute to flexibility nor
help with fitting process. The so-called co-planar interaction does
not make many physical sense to the author's knowledge. As a
consequence, the author made this model (re-built Keating model) to
test this known fact.

Hongze Xia
Mon Aug 18 17:36:02 2014
Ref: http://journals.aps.org/prb/abstract/10.1103/PhysRevB.84.155204
}
"""
import pickle
import numpy as np
from numpy.linalg import inv,eigh,eig,norm
from constants import M_THZ,TPI,KB,THZ_TO_J,THZ_TO_CM,THZ_TO_MEV
from .ewald import Ewald
from commonfunc import EigenSolver,MonkhorstPack,tetra_dos
from sys import exit

def ConstructFC(alpha,beta,nn,atom):
    """
    Construct the force constant tensors according to nearest neighbours.
    It assumes single type of bonding for one atom. But one could manually
    tune them by setting either alpha or beta to zero.
    alpha: array len(nn)
        bond-stretching force constant
    beta: array beta.shape=(len(nn),len(nn))
        bond-bending force constant
    nn: array of shape (N,3)
        all nearest neighbours in unit of lattice constant
    return: ndarray of shape (N,3,3)
        tensors for all nearest neighbours
    """
    N = len(nn)
    assert len(alpha) == N
    assert len(beta) == N
    assert len(beta[0]) == N
    alpha,beta = map(np.array,(alpha,beta))

    Alpha = np.zeros((N,3,3))
    Beta = np.zeros((N,3,3))
    d2 = np.dot(atom-nn[0],atom-nn[0])
    one = np.ones((3,3))
    
    for n1 in range(N):
        dvec = (atom-nn[n1]); tmp = dvec*one
        Alpha[n1] = -np.multiply(tmp,tmp.T)*alpha[n1]
        for n2 in range(N):
            if n1 != n2:
                dvec1 = (atom-nn[n2])
                tmp1 = (dvec1+dvec)*one; tmp2 = -dvec1*one
                Beta[n1] += np.multiply(tmp1.T,tmp2)*beta[n1,n2]
    Alpha *= 8/d2
    Beta *= 2/d2

    return Alpha+Beta

def DynBuild(basis,mass,bvec,fc,nn,label,kpts,crys=True):
    """
    Build the dynamical matrix from the short range force constant tensors.
    basis: ndarray of shape (N,3)
    mass: ndarray of shape (N)
        atomic unit, e.g, m(Si) = 28
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
    assert len(basis) == len(mass)
    N = len(mass); nks = len(kpts)
    dyn = np.zeros((nks,N*3,N*3),dtype=complex)
    # convert kpts to Cartesian coordinates if needed
    kpts = kpts.dot(bvec)*2.*np.pi if crys else kpts*2.*np.pi
    for q in range(nks):
        for i in range(N):
            for j in range(len(nn[i])):
                # With which atom?
                x = basis[i]-nn[i][j]; ka = label[i][j]
                mass_root = np.sqrt(mass[i]*mass[ka])
                # ON-diagonal
                dyn[q,i*3:i*3+3,i*3:i*3+3] -= fc[i][j]/mass[i]
                # OFF-diagonal
                dyn[q,i*3:i*3+3,ka*3:ka*3+3] += fc[i][j]/mass_root*np.exp(-1j*np.dot(kpts[q],x))
    # scale it to SI unit, omega^2 not frequency^2 after diagonalisation
    return dyn*M_THZ

class VFFM(object):
    """
    Valence Force Field Model takes inputs:
    lvec: ndarray, (3,3)
        lattice vectors in unit of alat
    basis: ndarray, (N,3)
        base atoms
    mass: array, (N,)
        in atomic unit
    symbol: string array or list, (N,)
        Element names of the basis, e.g., ["Si","Si"]
    """
    def __init__(self,lvec,basis,mass,symbol):
        self.lvec,self.bas,self.mass,self.symbol =\
                    map(np.array,(lvec,basis,mass,symbol))
        assert self.lvec.shape == (3,3)
        assert len(self.bas) == len(self.mass)
        assert len(symbol) == len(self.mass)

        # initialise the force constant tensors
        self.fc = []
        # initialise the k-points
        self.kpts = []
        # initialise the dynamical matrix
        self.dyn = []
        # initialise dos
        self.dos = []
        # set the reciprocal lattice, transpose is necessary
        self.bvec = inv(self.lvec).T
        # number of basis
        self.N = len(self.bas)
        # number of bands
        self.nbnd = 3*self.N
        # unit cell volume in terms of alat
        self.v = abs(np.inner(np.cross(self.lvec[0], self.lvec[1]), self.lvec[2]))
        # find and set the n.n
        self.set_nn()
        # the total number of first n.n.
        self.n1 = [len(self.nn[i]) for i in range(self.N) ]
        # initialise second n.n.
        self.n2 = None
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
            when one wants to use 2nd n.n.
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
            if dist2 == None:
                # first_nn_ind = [item for item in nn_ind if \
                #         abs(dist[item] - dist[nn_ind[0]])<1e-4]
                mask = np.abs(dist[nn_ind]-dist[nn_ind[0]])<1e-4
                first_nn_ind = nn_ind[mask]
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
        # fill in 2nd n.n. information
        if dist2:
            self.n2 = [len(self.nn[i])-self.n1[i] for i in range(self.N)]

    def set_bulk_fc(self,alpha,beta):
        '''
        Set the force constant tensors. Only two force constants needed:
        alpha: float
            bond-stretching
        beta: float
            bond-bending
        '''
        self.fc = []
        for i in range(self.N):
            N = len(self.nn[i])
            b = [beta]*N; bb = [b]*N
            self.fc.append(ConstructFC \
                ([alpha]*N,bb,self.nn[i],self.bas[i]))

    def set_bulk_fc2(self,alpha,beta):
        '''
        Set the force constant tensors. upto 2nd n.n.
        alpha: 2*[float]
            bond-stretching
        beta: 2*[float]
            bond-bending
        '''
        self.fc = []
        a1,a2 = alpha; b1,b2 = beta
        for i in range(self.N):
            n1 = self.n1[i]
            n2 = self.n2[i]
            b = [b1]*n1; bb = [b]*n1
            self.fc.append(ConstructFC \
                ([a1]*n1,bb,self.nn[i][0:n1],self.bas[i]))
            b = [b2]*n2; bb = [b]*n2
            tmp = ConstructFC \
                ([a2]*n2,bb,self.nn[i][n1::],self.bas[i])
            self.fc[i] = np.vstack((self.fc[i],tmp))

    def set_fc(self,fc_dict):
        '''
        fc_dict involves alpha and beta for different interactions between atoms.
        Make sure one now exactly how many interactions are involved by looking
        at the n.n for each atom in the basis. i.e., calc.get_nn_label()
        fc_dict must have this kind of format:
        fc_dict = {
        "alpha":a_dict,"beta":b_dict
        }
        a_dict and b_dict look like:
        a_dict = {
        "Ga-As": 10.0, "Al-As": 11.0, "Ga-Ga2":1.0
        }
        The last one with a trailing 2 indicates a second n.n. interactions.
        '''
        self.fc = []
        a_dict = fc_dict["alpha"]; b_dict = fc_dict["beta"]

        for i in range(self.N):
            onsite = self.symbol[i]
            offsite = self.nnsymb[i]
            nos = len(offsite)
            alp = np.zeros(nos)
            bet = np.zeros((nos,nos))
            for j in range(nos):
                key1 = onsite+"-"+offsite[j]
                key2 = offsite[j]+"-"+onsite
                if j >= self.n1[i]:
                    key1 += "2"
                    key2 += "2"
                if a_dict.has_key(key1):
                    key = key1
                elif a_dict.has_key(key2):
                    key = key2
                else:
                    raise ValueError("Missing interaction: "+key1)
                alp[j] = a_dict[key]
                tmp = []
                for k in range(nos):
                    if k != j:
                        key3 = onsite+"-"+offsite[k]
                        key4 = offsite[k]+"-"+onsite
                        # print keys
                        if j >= self.n1[i]:
                            key3 += "2"
                            key4 += "2"
                        if b_dict.has_key(key3):
                            key_b = key3
                        elif b_dict.has_key(key4):
                            key_b = key4
                        else:
                            # raise ValueError("Missing interaction: "+key3)
                            pass
                        if b_dict.has_key(key1):
                            key_a = key1
                        elif b_dict.has_key(key2):
                            key_a = key2
                        else:
                            # raise ValueError("Missing interaction: "+key1)
                            pass
                        if (j < self.n1[i] and j < self.n1[i]) or (j >= self.n1[i] and j >= self.n1[i]):
                            b_avg = (b_dict[key_b]+b_dict[key_a])*0.5
                            tmp.append(b_avg)
                        else:
                            tmp.append(0.0)
                    elif k==j:
                        tmp.append(0.0)
                #
                bet[j] = tmp
            #
            self.fc.append(ConstructFC \
                (alp,bet,self.nn[i],self.bas[i]))
        # self.fix_interface()

    def fix_interface(self):
        # Average the interface connection for set_fc()
        for i in range(self.N):
            # print "Checking atom ",i
            for j in range(len(self.nn[i])):
                bond0 = self.bas[i] - self.nn[i][j]
                offsiteLabel = self.label[i][j]
                for k in range(len(self.nn[offsiteLabel])):
                    bond1 = self.bas[offsiteLabel] - self.nn[offsiteLabel][k]
                    if np.allclose(bond0,-bond1): # the same bond
                        tmp = self.fc[i][j] + self.fc[offsiteLabel][k].T
                        # if np.allclose(self.fc[i][j],self.fc[offsiteLabel][k].T):
                        #     print self.symbol[i]," with ", self.nnsymb[i][j], " is fine"
                        # else:
                        #     print self.symbol[i]," with ", self.nnsymb[i][j], " is not fine"
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
        # self.kpts = np.dot(convec,self.kpts.T).T # this is potentially wrong
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
        self.dyn = DynBuild(self.bas,self.mass,self.bvec,self.fc,\
                self.nn,self.label,self.kpts,crys=self.iskcrys)

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
        if self.ecalc != None:
            self.dyn += self.eps*self.ecalc.get_dyn(self.mass,self.kpts,crys=self.iskcrys)
        self.freq,self.evec = EigenSolver(self.dyn,fldata=None)
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
        np.savetxt("dos.csv", self.dos, delimiter=",",fmt="%10.5f")
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
        if self.dos != []:
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

    def save(self,filename="myvffm.pickle"):
        """
        Save the whole object to Python pickle.
        filename: string
        """
        pickle.dump(self,open(filename,"wb"))

    def reload(filename="myvffm.pickle"):
        """Reload previous calculation"""
        return pickle.load(open(filename))

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
        np.savetxt("phfreq.csv", dt, delimiter=",",fmt="%10.5f",
        header="Phonon frequency in unit of "+phunit)
        self.k_conv(kpt)
        np.savetxt("phkpath.csv", self.kpts, delimiter=",",fmt="%8.3f",
        header="k-path in unit of "+kpt)

    def gen_kpath(self):
        self.k_conv("cart")
        tmp1 = self.kpts[0:-1]-self.kpts[1::]
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

    def fit_freq(self,src_freq,kpts,a0=80.,b0=10.,eps0=1.,crys=True,method='Powell'):
        """
        Fit the model to a given dispersion based on frequencies.
        src_freq: ndarray
            Please make sure ALL frequencies are included, e.g., degenerated ones and zeros
        kpts: ndarray
            kpts for source frequencies.
        a0,b0,eps0: float
            initial guess of alpha,beta for short range, epsilon for long range
        crys: boolean
            whether the kpts are in crystal coordinates or 2pi/alat
        method: string
            minimisation algorithm. Options could be any below:
            'Nelder-Mead','Powell','BFGS','Newton-CG'
        """
        from scipy.optimize import minimize
        np.set_printoptions(precision=3)
        np.set_printoptions(suppress=True)

        self.set_kpts(kpts,crys=crys)
        self.src_freq = np.sort(src_freq)
        assert len(self.src_freq) == self.nkpt
        if self.ecalc != None:
            self.m_ewald = self.ecalc.get_dyn(self.mass,kpts,crys=crys)
            res = minimize(self.__fit_ewald,x0=np.array([a0,b0,eps0]),method=method)
            a,b,eps = abs(res.x)
            print "Fitting routine returns: state (%s)" % res.success
            print "alpha = %10.6f; beta = %10.6f; eps = %10.6f" % (a,b,eps)
            self.set_bulk_fc(a,b); self.__set_dyn(); self.eps = eps
            print "Fitted frequencies:"
            print np.sort(self.get_ph_disp())
        else:
            res = minimize(self.__fit_no_ewald,x0=np.array([a0,b0]),method=method)
            a,b = abs(res.x)
            print "Fitting routine returns: state (%s)" % res.success
            print "alpha = %10.6f; beta = %10.6f" % (a,b)
            self.set_bulk_fc(a,b)
            print "Fitted frequencies:"
            print np.sort(self.get_ph_disp())
        if not res.success: print res.message
        self.__log_fit(res,filename="log_fit.txt")
        del minimize

    def __fit_no_ewald(self,ab0):
        a0,b0 = abs(ab0)
        print "alpha = %10.6f; beta = %10.6f" % (a0,b0)
        self.set_bulk_fc(a0,b0)
        freq = np.sort(self.get_ph_disp())
        return ((freq-self.src_freq)**2).sum()/len(freq)

    def __fit_ewald(self,abe0):
        a0,b0,eps0 = abs(abe0)
        print "alpha = %10.6f; beta = %10.6f; eps = %10.6f" % (a0,b0,eps0)
        self.set_bulk_fc(a0,b0); self.__set_dyn()
        dyn = eps0*self.m_ewald + self.get_dyn()
        freq,evec = EigenSolver(dyn,fldata=None)
        freq = np.sort(freq)
        return ((freq-self.src_freq)**2).sum()/len(freq)

    def fit_freq2(self,src_freq,kpts,a0=80.,b0=10.,a1=8.,b1=1.,eps0=1.0,crys=True,method='Powell'):
        """
        To use this function, one must have set all 2nd n.n already.
        See fit_freq above.
        a1,b1: float
            initial guess of second n.n. force constants
        """
        from scipy.optimize import minimize
        np.set_printoptions(precision=3)
        np.set_printoptions(suppress=True)
        self.set_kpts(kpts,crys=crys)
        self.src_freq = np.sort(src_freq)
        assert len(self.src_freq) == self.nkpt
        if self.ecalc != None:
            self.m_ewald = self.ecalc.get_dyn(self.mass,kpts,crys=crys)
            res = minimize(self.__fit_ewald2,x0=np.array([a0,b0,a1,b1,eps0]),method=method)
            a,b,a1,b1,eps = abs(res.x)
            print "Fitting routine returns: state (%s)" % res.success
            print "alpha = %10.6f; beta = %10.6f; alpha1 = %10.6f; beta1 = %10.6f;eps = %10.6f" % (a,b,a1,b1,eps)
            self.set_bulk_fc2([a,a1],[b,b1]); self.__set_dyn(); self.eps = eps
            print "Fitted frequencies:"
            print np.sort(self.get_ph_disp())
        else:
            res = minimize(self.__fit_no_ewald2,x0=np.array([a0,b0,a1,b1]),method=method)
            a,b,a1,b1 = abs(res.x)
            print "Fitting routine returns: state (%s)" % res.success
            print "alpha = %10.6f; beta = %10.6f; alpha1 = %10.6f; beta1 = %10.6f" % (a,b,a1,b1)
            self.set_bulk_fc2([a,a1],[b,b1])
            print "Fitted frequencies:"
            print np.sort(self.get_ph_disp())
        if not res.success: print res.message
        self.__log_fit(res,filename="log_fit2.txt")
        del minimize

    def __fit_no_ewald2(self,x0):
        a0,b0,a1,b1 = abs(x0)
        print "alpha = %10.6f; beta = %10.6f; alpha1 = %10.6f; beta1 = %10.6f" % (a0,b0,a1,b1)
        self.set_bulk_fc2([a0,a1],[b0,b1])
        freq = np.sort(self.get_ph_disp())
        return ((freq-self.src_freq)**2).sum()/len(freq)

    def __fit_ewald2(self,x0):
        a0,b0,a1,b1,eps0 = abs(x0)
        print "alpha = %10.6f; beta = %10.6f; alpha1 = %10.6f; beta1 = %10.6f;eps = %10.6f" % (a0,b0,a1,b1,eps0)
        self.set_bulk_fc2([a0,a1],[b0,b1]); self.__set_dyn()
        dyn = eps0*self.m_ewald + self.get_dyn()
        freq,evec = EigenSolver(dyn,fldata=None)
        freq = np.sort(freq)
        return ((freq-self.src_freq)**2).sum()/len(freq)

    def __log_fit(self,res,filename="logfit.txt"):
        # Log the fitting results
        import time
        logfile = open(filename,"w")
        print >>logfile,"Log: ",time.strftime("%Y-%m-%d %H:%M")
        print >>logfile,"Current system:"
        for i in range(self.N):
            print >>logfile,"%s %8.3f %8.3f %8.3f" % ((self.symbol[i],)+tuple(self.bas[i]))
        print >>logfile,"Unit cell dimension:"
        for vec in self.lvec: print >>logfile,"%8.4f"*3 % tuple(vec)
        print >>logfile,"Fitting routine returns: state (%s)" % res.success
        print >>logfile,"%10.5f"*len(res.x) % tuple(abs(res.x))
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
