"""
Contains the JetTree structure along with a DeclustState kinematic
bookkeeping class and a LundImage visualization tool.
"""
import fastjet as fj
import numpy as np
import math

#======================================================================
class SplittingInfo:
    dimension = 2

    #----------------------------------------------------------------------
    def __init__(self, lnDelta, lnKt):
        self.lnDelta = lnDelta
        self.lnKt    = lnKt
        #self.lnz     = lnz
        #self.lnKappa = lnKappa
        #self.psi     = psi

    #----------------------------------------------------------------------
    @classmethod
    def fromPseudoJet(cls, j1, j2):
        delta = math.sqrt(j1.squared_distance(j2))
        lnDelta = math.log(delta)
        lnKt    = math.log(j2.pt()*delta)
        #lnz     = math.log(z)
        #lnKappa = math.log(z * delta)
        #psi     = math.atan((j1.rap() - j2.rap())/(j1.phi() - j2.phi()))
        return cls(lnDelta, lnKt)

    #----------------------------------------------------------------------
    def values(self):
        """Return kinematics of the splitting as an array."""
        return np.array([self.lnDelta, self.lnKt])

    #----------------------------------------------------------------------
    def lund(self):
        """Return a two-dimensional array with lund coordinates."""
        return np.array([-self.lnDelta, self.lnKt])

#======================================================================
class BranchInfo:
    dimension = 1

    #----------------------------------------------------------------------
    def __init__(self, lnPt):
        self.lnPt = lnPt
        #self.lnm  = lnm

    #----------------------------------------------------------------------
    @classmethod
    def fromPseudoJet(cls, j):
        lnPt = math.log(j.pt())
        #lnm  = 0.5*math.log(abs(j.m2()))
        return cls(lnPt)

    #----------------------------------------------------------------------
    def values(self):
        """Return splitting information as an array."""
        return np.array([self.lnPt])

    #----------------------------------------------------------------------
    def pt(self):
        """Return pt of the branch."""
        return math.exp(self.lnPt)
        
#======================================================================
class KinematicState:
    """DeclustState contains information about a declustering."""
    dimension = 4
    
    #----------------------------------------------------------------------
    def __init__(self, branch, splitting=None):
        self.splitting = splitting
        self.branch    = branch

    #----------------------------------------------------------------------
    def lund(self):
        """"Return a two-dimensional array with lund coordinates."""
        return np.array([-self.lnDelta, self.lnKt])
    
    #----------------------------------------------------------------------
    def values(self):
        """Return the values of the splitting and branch."""
        if not self.splitting:
            return self.branch.values()
        return np.append(self.splitting.values(), self.branch.values())
        
    #----------------------------------------------------------------------
    def __lt__(self, other_tree):
        """Comparison operator needed for a priority queue."""
        return self.lnDelta > other_state.lnDelta
        
#======================================================================
class JetTree:
    """JetTree keeps track of the tree structure of a jet declustering."""

    #----------------------------------------------------------------------
    def __init__(self, state, child=None, harder=None, softer=None):
        self.state  = state
        self.child  = child
        self.harder = harder
        self.softer = softer
        
    #----------------------------------------------------------------------
    @classmethod
    def fromPseudoJet(cls, jet, child=None, splitting=None):
        j1 = fj.PseudoJet()
        j2 = fj.PseudoJet()
        branch = BranchInfo.fromPseudoJet(jet)
        state  = KinematicState(branch, splitting)
        node   = cls(state, child)
        if jet and jet.has_parents(j1,j2):
            # order the parents in pt
            if (j2.pt() > j1.pt()):
                j1,j2=j2,j1
            newsplitting = SplittingInfo.fromPseudoJet(j1, j2)
            # should the following be cls.fromPseudoJet ?
            node.harder  = JetTree.fromPseudoJet(j1, node, newsplitting)
            node.softer  = JetTree.fromPseudoJet(j2, node, newsplitting)
        return node
        
    #----------------------------------------------------------------------
    def __lt__(self, other_tree):
        """Comparison operator needed for a priority queue."""
        if not self.state:
            return False
        return self.state > other_tree.state

    #----------------------------------------------------------------------
    def __del__(self):
        """Delete the node."""
        if self.softer:
            del self.softer
        if self.harder:
            del self.harder
        del self

#======================================================================
class LundImage:
    """Class to create Lund images from a jet tree."""

    #----------------------------------------------------------------------
    def __init__(self, xval = [0.0, 7.0], yval = [-3.0, 7.0],
                 npxlx = 50, npxly = None):
        """Set up the LundImage instance."""
        # set up the pixel numbers
        self.npxlx = npxlx
        if not npxly:
            self.npxly = npxlx
        else:
            self.npxly = npxly
        # set up the bin edge and width
        self.xmin = xval[0]
        self.ymin = yval[0]
        self.x_pxl_wdth = (xval[1] - xval[0])/self.npxlx
        self.y_pxl_wdth = (yval[1] - yval[0])/self.npxly

    #----------------------------------------------------------------------
    def __call__(self, tree):
        """Process a jet tree and return an image of the primary Lund plane."""
        res = np.zeros((self.npxlx,self.npxly))
        self.fill(tree, res)
        return res

    #----------------------------------------------------------------------
    def fill(self, tree, res):
        """Fill the res array recursively with the tree declusterings of the hard branch."""
        if (tree and tree.state and tree.state.splitting):
            x,y = tree.state.splitting.lund()
            xind = math.ceil((x - self.xmin)/self.x_pxl_wdth - 1.0)
            yind = math.ceil((y - self.ymin)/self.y_pxl_wdth - 1.0)
            if (xind < self.npxlx and yind < self.npxly and min(xind,yind) >= 0):
                res[xind,yind] += 1
        if (tree):
            self.fill(tree.harder, res)

