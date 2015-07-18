#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
import numpy as np

# kpoint_convert, ibz_points and get_bandpath were taken from the Python package ASE
# Acknowledge their contribution here.

def kpoint_convert(cell_cv, skpts_kc=None, ckpts_kv=None):
    """Convert k-points between scaled and cartesian coordinates.

    Given the atomic unit cell, and either the scaled or cartesian k-point
    coordinates, the other is determined.

    The k-point arrays can be either a single point, or a list of points,
    i.e. the dimension k can be empty or multidimensional.
    """
    if ckpts_kv is None:
        icell_cv = 2 * np.pi * np.linalg.inv(cell_cv).T
        return np.dot(skpts_kc, icell_cv)
    elif skpts_kc is None:
        return np.dot(ckpts_kv, cell_cv.T) / (2 * np.pi)
    else:
        raise KeyError('Either scaled or cartesian coordinates must be given.')


def get_bandpath(points, cell, npoints=50):
    """Make a list of kpoints defining the path between the given points.

    points: list
        List of special IBZ point pairs, e.g. ``points =
        [W, L, Gamma, X, W, K]``.  These should be given in
        scaled coordinates.
    cell: 3x3 ndarray
        Unit cell of the atoms.
    npoints: int
        Length of the output kpts list.

    Return list of k-points, list of x-coordinates and list of
    x-coordinates of special points."""

    points = np.asarray(points)
    dists = points[1:] - points[:-1]
    lengths = [np.linalg.norm(d) for d in kpoint_convert(cell, skpts_kc=dists)]
    length = sum(lengths)
    kpts = []
    x0 = 0
    x = []
    X = [0]
    for P, d, L in zip(points[:-1], dists, lengths):
        n = int(round(L * (npoints - 1 - len(x)) / (length - x0)))
        for t in np.linspace(0, 1, n, endpoint=False):
            kpts.append(P + t * d)
            x.append(x0 + t * L)
        x0 += L
        X.append(x0)
    kpts.append(points[-1])
    x.append(x0)
    return np.array(kpts), np.array(x), np.array(X)

# The following is a list of the critical points in the 1. Brillouin zone
# for some typical crystal structures.
# (In units of the reciprocal basis vectors)
# See http://en.wikipedia.org/wiki/Brillouin_zone
ibz_points = {'cubic': {'Gamma': [0,     0,     0    ],
                        'X':     [0,     0 / 2, 1 / 2],
                        'R':     [1 / 2, 1 / 2, 1 / 2],
                        'M':     [0 / 2, 1 / 2, 1 / 2]},

              'fcc':   {'Gamma': [0,     0,     0    ],
                        'X':     [1 / 2, 0,     1 / 2],
                        'W':     [1 / 2, 1 / 4, 3 / 4],
                        'K':     [3 / 8, 3 / 8, 3 / 4],
                        'U':     [5 / 8, 1 / 4, 5 / 8],
                        'L':     [1 / 2, 1 / 2, 1 / 2],
                        'X1':    [0.5,   0.5,     1.0],
                        'W1':    [0.5,   0.75,   1.25]},

              'bcc':   {'Gamma': [0,      0,     0    ],
                        'H':     [1 / 2, -1 / 2, 1 / 2],
                        'N':     [0,      0,     1 / 2],
                        'P':     [1 / 4,  1 / 4, 1 / 4]},
              'hexagonal':
                       {'Gamma': [0,      0,       0   ],
                        'M':     [0,      1 / 2,   0   ],
                        'K':     [-1 / 3, 1 / 3,   0   ],
                        'A':     [0,      0,     1 / 2 ],
                        'L':     [0,     1 / 2,  1 / 2 ],
                        'H':     [-1 / 3, 1 / 3, 1 / 2 ]},
              'tetragonal':
                       {'Gamma': [0,      0,       0   ],
                        'X':     [1 / 2,  0,       0   ],
                        'M':     [1 / 2,  1 / 2,   0   ],
                        'Z':     [0,      0,     1 / 2 ],
                        'R':     [1 / 2,  0,     1 / 2 ],
                        'A':     [1 / 2,  1 / 2, 1 / 2 ]},
              'orthorhombic':
                       {'Gamma': [0,      0,       0   ],
                        'R':     [1 / 2,  1 / 2, 1 / 2 ],
                        'S':     [1 / 2,  1 / 2,   0   ],
                        'T':     [0,      1 / 2, 1 / 2 ],
                        'U':     [1 / 2,  0,     1 / 2 ],
                        'X':     [1 / 2,  0,       0   ],
                        'Y':     [0,      1 / 2,   0   ],
                        'Z':     [0,      0,     1 / 2 ]},              
}

# Below is my original contribution

ibz_path = {
    'fcc': ['Gamma', 'X', 'U', 'L', 'Gamma', 'K', 'X1', 'W1'],
    'hexagonal': ['Gamma','K','M','Gamma','A','H','L','A'],
    'tetragonal': ['Gamma', 'X', 'M', 'Gamma', 'Z', 'R', 'A'],
    'bcc': ['Gamma','H','P','Gamma','N'],
    'cubic': ['X','R','M','Gamma','R'],
    'orthorhombic': ['Z','Gamma','Y','S','R','U','X','Gamma']
}

ibz_point_names = {
    'fcc': ['$\Gamma$', 'X', 'U', 'L', '$\Gamma$', 'K', 'X', 'W'],
    'hexagonal': ['$\Gamma$', 'K', 'M', '$\Gamma$', 'A', 'H','L','A'],
    'tetragonal': ['$\Gamma$', 'X', 'M', '$\Gamma$', 'Z', 'R', 'A'],
    'bcc': ['$\Gamma$','H','P','$\Gamma$','N'],
    'cubic': ['X','R','M','$\Gamma$','R'],
    'orthorhombic': ['Z','$\Gamma$','Y','S','R','U','X','$\Gamma$']
}

def default_k_path(crys,lvec,num=300):
    """
    crys: str, "fcc", "cubic", "hexagonal", "bcc", "tetragonal","orthorhombic"
    lvec: 3 by 3 np.array
    """
    lvec = np.array(lvec)
    assert lvec.shape == (3,3), "Wrong input for lattice vectors!"
    crys = crys.lower()
    if crys == 'hcp': crys = "hexagonal"
    try:
        points = ibz_points[crys]
        point_names = ibz_point_names[crys]
        path = [points[item] for item in ibz_path[crys]]
        print "Generating k path for %s structure:" % crys
        tmp = ""
        for i in range(len(point_names)-1):
            tmp += point_names[i]+" -> "
        tmp += point_names[-1]
        print tmp
        kpts, x, X = get_bandpath(path, lvec, num)
        return kpts, point_names, x, X
    except KeyError:
        print "Warning: %s is not supported!" % crys
        print "Currently, only {cubic,fcc,hexagonal,bcc,tetragonal,orthorhombic} are supported!"