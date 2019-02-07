"""Test script that creates random jets, using no data and no training."""

from JetTree import *
import numpy as np
import random

deltacut = 0.005

def addSplitting(nodes):
    # if list is empty, no splitting can be added
    if not nodes:
        return False
    # first get a random node from the list
    ind  = random.randint(0, len(nodes)-1)
    node = nodes.pop(ind)
    # first get pt1, pt2
    pt   = node.state.branch.pt()
    z    = np.random.uniform(0,0.5)
    pt1  = (1-z)*pt
    pt2  = z*pt
    # delta sampled from uniform distribution between 0 and 1
    deltamax = 1.0 if not node.state.splitting else np.exp(node.state.splitting.lnDelta)
    delta = np.random.uniform(0, deltamax)
    # lnkt sampled from normal distribution with mean=2, var=3
    lnkt = np.random.normal(loc=2.0,scale=3.0)
    # if delta below cut, veto the splitting
    if delta < deltacut:
        return False
    # otherwise create the two new branches and add them to the queue
    split = SplittingInfo(math.log(delta), lnkt)
    branch1 = BranchInfo(math.log(pt1))
    branch2 = BranchInfo(math.log(pt2))
    node.harder  = JetTree(KinematicState(branch1, split), node)
    node.softer  = JetTree(KinematicState(branch2, split), node)
    nodes.append(node.harder)
    nodes.append(node.softer)
    return True

    
def newJet(pt = 1.0):
    root  = JetTree(KinematicState(BranchInfo(math.log(pt))))
    nodes = [root]
    while nodes:
        addSplitting(nodes)
    return root
