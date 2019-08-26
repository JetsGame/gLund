# This file is part of gLund by S. Carrazza and F. A. Dreyer

import fastjet as fj
import numpy as np
import math

#======================================================================
class LundCoordinates:
    """
    LundCoordinates takes two subjets associated with a declustering,
    and store the corresponding Lund coordinates."""

    #----------------------------------------------------------------------
    def __init__(self, j1, j2):
        """Define a number of variables associated with the declustering."""
        delta = j1.delta_R(j2)
        z     = j2.pt()/(j1.pt() + j2.pt())
        self.lnKt    = math.log(j2.pt()*delta)
        self.lnDelta = math.log(delta)
        # self.lnm     = 0.5*math.log(abs((j1 + j2).m2()))
        self.lnz     = math.log(z)
        self.lnKappa = math.log(z * delta)
        # self.psi     = math.atan((j1.rap() - j2.rap())/(j1.phi() - j2.phi()))


#======================================================================
class JetTree:
    """JetTree keeps track of the tree structure of a jet declustering."""

    #----------------------------------------------------------------------
    def __init__(self, pseudojet, child=None):
        """Initialize a new node, and create its two parents if they exist."""
        self.harder = None
        self.softer = None
        self.lundCoord = None
        # first define the current node
        self.node = np.array([pseudojet.px(),pseudojet.py(),pseudojet.pz(),pseudojet.E()])
        # if it has a direct child (i.e. one level further up in the
        # tree), give a link to the corresponding tree object here
        self.child  = child
        j1 = fj.PseudoJet()
        j2 = fj.PseudoJet()
        if pseudojet and pseudojet.has_parents(j1,j2):
            # order the parents in pt
            if (j2.pt() > j1.pt()):
                j1,j2=j2,j1
            # then create two new tree nodes with j1 and j2
            self.harder = JetTree(j1, self)
            self.softer = JetTree(j2, self)
            self.lundCoord = LundCoordinates(j1, j2)
    
    #----------------------------------------------------------------------
    def jet(self, pseudojet=False):
        """Return the kinematics of the JetTree."""
        #TODO: implement pseudojet option which returns a pseudojet
        #      with the reclustered constituents (after grooming)
        if not pseudojet:
            return self.node
        else:
            raise ValueError("JetTree: jet() with pseudojet return value not implemented.")

#======================================================================
class LundImage:
    """Class to create Lund images from a jet tree."""

    xval = [0.0, 7.0]
    yval = [-3.0, 7.0]
    
    __yval_kt = [-3.0, 7.0]
    __yval_z = [-8.0, 0.0]

    #----------------------------------------------------------------------
    def __init__(self, xval = None, yval = None, npxlx = 50, npxly = None,
                 norm_to_one = True, y_axis = 'kt'):
        """Set up the LundImage instance."""
        # set up the pixel numbers
        self.npxlx = npxlx
        if not npxly:
            self.npxly = npxlx
        else:
            self.npxly = npxly
        # set a flag which determines if pixels are normalized to one or not
        self.norm_to_one = norm_to_one
        # set up the bin edge and width
        LundImage.change_ybox(y_axis)
        xv = xval if xval else LundImage.xval
        yv = yval if yval else LundImage.yval
        self.xmin = xv[0]
        self.ymin = yv[0]
        self.x_pxl_wdth = (xv[1] - xv[0])/self.npxlx
        self.y_pxl_wdth = (yv[1] - yv[0])/self.npxly
        self.y_axis = y_axis

    #----------------------------------------------------------------------
    @staticmethod
    def change_ybox(y_axis):
        if y_axis=='kt':
            LundImage.yval = LundImage.__yval_kt
        elif y_axis=='z' or y_axis=='kappa':
            LundImage.yval = LundImage.__yval_z
        else:
            raise ValueError("LundImage: invalid y_axis.")

    #----------------------------------------------------------------------
    def __call__(self, tree):
        """Process a jet tree and return an image of the primary Lund plane."""
        res = np.zeros((self.npxlx,self.npxly))
        self.fill(tree, res)
        return res

    #----------------------------------------------------------------------
    def fill(self, tree, res):
        """Fill the res array recursively with the tree declusterings of the hard branch."""
        if(tree and tree.lundCoord):
            x = -tree.lundCoord.lnDelta
            if self.y_axis=='kt':
                y = tree.lundCoord.lnKt
            if self.y_axis=='z':
                y = tree.lundCoord.lnz
            elif self.y_axis=='kappa':
                y = tree.lundCoord.lnKappa
            else:
                raise ValueError("LundImage: invalid y_axis.")
            xind = math.ceil((x - self.xmin)/self.x_pxl_wdth - 1.0)
            yind = math.ceil((y - self.ymin)/self.y_pxl_wdth - 1.0)
            if (xind < self.npxlx and yind < self.npxly and min(xind,yind) >= 0):
                if (res[xind,yind] < 1 or not self.norm_to_one):
                    res[xind,yind] += 1
            self.fill(tree.harder, res)
            #self.fill(tree.softer, res)


#======================================================================
class SoftDropMult:
    """Class to compute the soft drop multiplicity of a jet."""

    #----------------------------------------------------------------------
    def __init__(self, zcut = 0.007, beta=-1, thetacut=0, R0=1):
        """Set up the LundImage instance."""
        self.zcut=zcut
        self.beta=beta
        self.thetacut=thetacut
        self.R0=R0

    #----------------------------------------------------------------------
    def __call__(self, tree):
        """Process a jet tree and return the soft drop multiplicity."""
        nsd = 0
        return self.fill(tree, nsd)

    #----------------------------------------------------------------------
    def fill(self, tree, nsd):
        """Fill the res array recursively with the tree declusterings of the hard branch."""
        if(tree and tree.lundCoord):
            delta = math.exp(tree.lundCoord.lnDelta)
            #z = math.exp(tree.lundCoord.lnz)
            z = np.exp(tree.lundCoord.lnKappa)/delta
            if (delta > self.thetacut and z > self.zcut * ((delta/self.R0)**self.beta)):
                return self.fill(tree.harder, nsd+1)
            else:
                return self.fill(tree.harder, nsd)
        return nsd

