import fastjet as fj
import numpy as np
import math

#======================================================================
class DeclustState:
    """DeclustState contains information about a declustering."""
    dimension = 4
    
    #----------------------------------------------------------------------
    def __init__(self, lnDelta, lnKt, lnPt1, lnPt2):
        self.lnDelta = lnDelta
        self.lnKt    = lnKt
        self.lnPt1   = lnPt1
        self.lnPt2   = lnPt2

    #----------------------------------------------------------------------
    @classmethod
    def fromPseudoJet(cls, j1, j2):
        delta = math.sqrt(j1.squared_distance(j2))
        lnDelta = math.log(delta)
        lnKt    = math.log(j2.pt()*delta)
        lnPt1   = math.log(j1.pt())
        lnPt2   = math.log(j2.pt())
        #lnm     = 0.5*math.log(abs((j1 + j2).m2()))
        #lnz     = math.log(z)
        #lnKappa = math.log(z * delta)
        #psi     = math.atan((j1.rap() - j2.rap())/(j1.phi() - j2.phi()))
        return cls(lnDelta, lnKt, lnPt1, lnPt2)
        #return cls(0.0, 0.0, 0.0, 0.0)

    #----------------------------------------------------------------------
    def lund(self):
        """"Return a two-dimensional array with lund coordinates."""
        return np.array([-self.lnDelta, self.lnKt])

    #----------------------------------------------------------------------
    def values(self):
        """Return the values of the declustering."""
        return np.append(self.kin_split(), np.append(self.kin_hard(),self.kin_soft()))

    #----------------------------------------------------------------------
    def kin_hard(self):
        """Return kinematics of hard branch."""
        return np.array([self.lnPt1])
    
    #----------------------------------------------------------------------
    def kin_soft(self):
        """Return kinematics of hard branch."""
        return np.array([self.lnPt2])
    
    #----------------------------------------------------------------------
    def kin_split(self):
        """Return kinematics of splitting."""
        return np.array([self.lnDelta, lnKt])
    
    #----------------------------------------------------------------------
    def values_hard(self):
        """Return the values of the hard branch of the declustering."""
        return np.append(self.kin_split(), self.kin_hard())
    
    #----------------------------------------------------------------------
    def values_soft(self):
        """Return the values of the soft branch of the declustering."""
        return np.append(self.kin_split(), self.kin_soft())
    
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
    def fromPseudoJet(cls, pseudojet, child=None):
        j1 = fj.PseudoJet()
        j2 = fj.PseudoJet()
        state  = None
        harder = None
        softer = None
        if pseudojet and pseudojet.has_parents(j1,j2):
            # order the parents in pt
            if (j2.pt() > j1.pt()):
                j1,j2=j2,j1
            state = DeclustState.fromPseudoJet(j1, j2)
            tmp1 = fj.PseudoJet()
            tmp2 = fj.PseudoJet()
            node = cls(state, child)
            if j1.has_parents(tmp1,tmp2):
                harder = JetTree.fromPseudoJet(j1, node)
                node.harder = harder
            if j2.has_parents(tmp1,tmp2):
                softer = JetTree.fromPseudoJet(j2, node)
                node.softer = softer
            return node
        return None
    
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
        if(tree and tree.state):
            x,y = tree.state.lund()
            xind = math.ceil((x - self.xmin)/self.x_pxl_wdth - 1.0)
            yind = math.ceil((y - self.ymin)/self.y_pxl_wdth - 1.0)
            if (xind < self.npxlx and yind < self.npxly and min(xind,yind) >= 0):
                res[xind,yind] += 1
            self.fill(tree.harder, res)
            #self.fill(tree.softer, res)

