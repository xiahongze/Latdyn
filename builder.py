#!/usr/bin/env python
import numpy as np

def BulkBuilder(crystype,withBC=False,r12=None):
    """
    crystype: str
        Crystal type: fcc,hcp,sc,diamond,wurtzite,rocksalt
    return: 
        3 by 3 lattice vectors, N by 3 basis vectors if withBC==False
    """
    if crystype.lower() == 'fcc':
        a = np.array([[0.5,0.5,0.],[0.5,0.,0.5],[0.,0.5,0.5]]) # fcc lattice vectors
        b = np.array([[0.,0.,0.]])
    elif crystype.lower() == 'sc':
        a = np.array([[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])
        b = np.array([[0.,0.,0.]])
        if withBC and r12 != None:
            rbc1 = r12/(r12+1.)
            bc = Bonding('sc')*rbc1
        elif withBC and r12 == None:
            raise ValueError("The BC-ion length ratio r12 needs to be set!")
    elif crystype.lower() == 'diamond':
        a = np.array([[0.5,0.5,0.],[0.5,0.,0.5],[0.,0.5,0.5]])
        b = np.array([[0.,0.,0.],[0.25,0.25,0.25]])
        if withBC and r12 != None:
            rbc1 = r12/(r12+1.)
            bc = Bonding('diamond')*rbc1
        elif withBC and r12 == None:
            raise ValueError("The BC-ion length ratio r12 needs to be set!")
    elif crystype.lower() == 'wurtzite':
        c = np.sqrt(8./3.) # default for Wurtzite
        u = 3.0/8.0 # internal parameter
        a = np.array([[1.0,0.,0.],[-0.5,np.sqrt(3)/2,0.],[0.,0.,c]])
        b = np.array([[1./3.,2./3.,0.],[2./3.,1./3.,0.5],[1./3.,2./3.,u],[2./3.,1./3.,0.5+u]]).dot(a) # =>>Cartesian coordinates
        symion = np.array(["A0","A0","A1","A1"])
        if withBC and r12 != None:
            rbc1 = r12/(r12+1.)
            bc0,bc1 = Bonding('wurtzite')
            bc = np.vstack((bc0*r12+b[0],bc1*r12+b[1]))
        elif withBC and r12 == None:
            raise ValueError("The BC-ion length ratio r12 needs to be set!")
    elif crystype.lower() == 'hcp':
        a = np.array([[1.0,0.,0.],[-0.5,np.sqrt(3)/2,0.],[0.,0.,1.]])
        b = np.array([[0.,0.,0.],[2./3.,1./3.,0.0]]).dot(a)
        symion = np.array(["A0","A0"])
    elif crystype.lower() == 'rocksalt':
        a = np.array([[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])
        b = np.array([[0.,0.,0.],[0.5,0.5,0.5]])
        if withBC and r12 != None:
            rbc1 = r12/(r12+1.)
            bc = Bonding('rocksalt')*rbc1
        elif withBC and r12 == None:
            raise ValueError("The BC-ion length ratio r12 needs to be set!")
    else:
        raise ValueError(crystype+" is not yet supported!")
    
    if crystype.lower() in ['fcc','sc']:
        symion = np.array(["A0"])
    elif crystype.lower() in ['rocksalt','diamond']:
        symion = np.array(["A0","A1"])

    if not withBC:
        return a,b,symion
    else:
        symbc = np.array(["BC"]*len(bc))
        return a,b,bc,symion,symbc

def Bonding(crystype):
    """
    crystype: str
        Crystal type: fcc,hcp,sc,diamond,wurtzite
    return: N by 3 bond-charge vectors
    """
    if crystype.lower() == 'diamond':
        bc = np.array([[1,1,1],[-1,-1,1],[-1,1,-1],[1,-1,-1]])*0.25
    elif crystype.lower() == 'wurtzite':
        a,ion = Builder('wurtzite')
        bc0 = np.array([[ 0.     ,  0.57735,  0.61237],
                                [-0.5    ,  0.28868, -0.20412],
                                [ 0.5    ,  0.28868, -0.20412],
                                [ 0.     ,  1.1547 , -0.20412]])
        bc1 = np.array([[ 0.5    ,  0.28868,  1.42887],
                                [ 1.     ,  0.57735,  0.61237],
                                [ 0.5    , -0.28868,  0.61237],
                                [ 0.     ,  0.57735,  0.61237]])
        bc0 = bc0-ion[0]
        bc1 = bc1-ion[1]
        bc = (bc0,bc1)
    elif crystype.lower() == 'rocksalt':
        bc = np.array([[1,0,0],[0,1,0],[0,0,1],[-1,0,0],[0,-1,0],[0,0,-1]])*0.5
    elif crystype.lower() == 'sc':
        bc = np.array([[1,0,0],[0,1,0],[0,0,1],[-1,0,0],[0,-1,0],[0,0,-1]])
    else:
        raise ValueError(crystype+" is not yet supported!")
    return bc

def MQW(crystype,N=2,withBC=False,r12=None):
    """docstring for MQW"""
    if crystype.lower() == 'diamond100':
        a = np.array([[0.5,0.5,0.],[0.5,-0.5,0.],[0.,0.,N]])
        smallbasis = np.array([[0.,0.,0.],[0.25,0.25,0.25],[0.,0.5,0.5],[0.25,0.75,0.75]])
        ion = smallbasis
        tmp = np.array([0.,0.,1.])
        for i in range(1,N):
            ion = np.vstack((ion,smallbasis+tmp*i))
        symion = np.asarray(["A0","A1","A0","A1"]*N)
        if withBC and r12 != None:
            rbc1 = r12/(r12+1.)
            bonds = Bonding('diamond')*rbc1
            mask = (symion=="A0")
            bc = []
            for item in ion[mask]:
                bc = item+bonds if bc == [] else np.vstack((bc,item+bonds))
        elif withBC and r12 == None:
            raise ValueError("The BC-ion length ratio r12 needs to be set!")
    elif crystype.lower() == 'wurtzite':
        a,ion0 = Builder('wurtzite')
        c = a[2,2]
        a[2] *= N # scale on the c-axis
        # ALL ion positions
        ion = ion0
        for i in range(1,N):
            tmp = ion0+[0.,0.,c*i]
            ion = np.vstack((ion,tmp))
        symion = np.asarray(["A0","A0","A1","A1"]*N)
        if withBC and r12 != None:
            rbc1 = r12/(r12+1.)
            bc0,bc1 = Bonding('wurtzite')
            bc0 = bc0*rbc1+ion[0] # add to the ion
            bc1 = bc1*rbc1+ion[1]
            bc0all = bc0; bc1all = bc1
            for i in range(1,N):
                tmp = bc0+[0.,0.,c*i]
                bc0all = np.vstack((bc0all,tmp))
                tmp = bc1+[0.,0.,c*i]
                bc1all = np.vstack((bc1all,tmp))
            bc = np.vstack((bc0all,bc1all))
        elif withBC and r12 == None:
            raise ValueError("The BC-ion length ratio r12 needs to be set!")
    else:
        raise ValueError(crystype+" is not yet supported!")

    if crystype.lower() in ['diamond100','wurtzite']:
        symbc = ['BC']*4*2*N

    if withBC:
        return a,ion,bc,symion,symbc
    if not withBC:
        return a,ion,symion