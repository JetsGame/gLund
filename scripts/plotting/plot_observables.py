# This file is part of gLund by S. Carrazza and F. A. Dreyer
import matplotlib.pyplot as plt
from glund.read_data import Jets 
from glund.JetTree import JetTree, LundImage, SoftDropMult
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.interpolate import interp2d
import numpy as np
import argparse,math

def n_sd(image, xbins, ybins, beta, zcut, thetacut, R0=1.0):
    """The soft drop multiplicity of an image. Assumes y-axis=kappa"""
    nsd=0
    dx=xbins[1]-xbins[0]
    dy=ybins[1]-ybins[0]
    for ix in range(image.shape[0]):
        for iy in range(image.shape[1]):
            if (image[ix,iy]>0.0):
                delta = np.exp(xbins[ix]+np.random.uniform(-dx/2,dx/2))
                z = np.exp(ybins[iy]+np.random.uniform(-dy/2,dy/2))/delta
                if (delta > thetacut and z > zcut * ((delta/R0)**beta)):
                    nsd+=1
    return nsd

#----------------------------------------------------------------------
def plot_sdmult(filedic, imgs_ref, figname, nsd_ref, npx=24, zcut=0.007, beta=-1, thetacut=0.0):
    """Plot a slice in kt of the lund image for different models and a reference sample."""
    xvals=np.linspace(LundImage.xval[0], LundImage.xval[1], npx+1)
    yvals=np.linspace(LundImage.yval[0], LundImage.yval[1], npx+1)
    xbins=np.array([0.5*(xvals[i]+xvals[i+1]) for i in range(len(xvals)-1)])
    ybins=np.array([0.5*(yvals[i]+yvals[i+1]) for i in range(len(yvals)-1)])

    yy=np.linspace(LundImage.yval[0], LundImage.yval[1], 10000)
    xx=np.linspace(LundImage.xval[0], LundImage.xval[1], 10000)
    
    fct= {}
    nsd= {}
    for lab in filedic.keys():
        ims = np.load(filedic[lab])
        im = np.average(ims, axis=0)
        f = interp2d(xbins, ybins, im, kind='linear') #linear, cubic, quintic
        fct[lab] = f
        nsd[lab]=[]
        for i in range(ims.shape[0]):
            nsd[lab].append(n_sd(ims[i], xbins, ybins, beta, zcut, thetacut))
            
    nsd_ref_im = []
    for i in range(len(imgs_ref)):
        nsd_ref_im.append(n_sd(imgs_ref[i], xbins, ybins, beta, zcut, thetacut))
    
    fig, ax = plt.subplots(figsize=(5,3.5))
    bins = np.arange(0, 25, 1)
    for lab in nsd.keys():
        plt.hist(nsd[lab], bins=bins, histtype='step', density=True, label=lab)
    plt.hist(nsd_ref_im, bins=bins, histtype='step', density=True, label='Pythia 8')
    plt.hist(nsd_ref, bins=bins, histtype='step', color='C3', ls=':', density=True)
    plt.text(0.4,0.275,'$z_\mathrm{cut}=%.3f,\, \\beta=%i,\, \\theta_\mathrm{cut}=%.1f$' % (zcut,beta,thetacut))
    ax.set_xlim((0,12))
    ax.set_ylim((0.0,0.30))
    ax.set_xlabel('$n_\mathrm{SD}$')
    plt.legend()
    ax.grid(linestyle=':')
    plt.savefig(figname, bbox_inches='tight')
    plt.close()

#----------------------------------------------------------------------
def main(args):
    zcut=0.007
    beta=-1
    thetacut=0.0009
    if args.data:
        sdmult=SoftDropMult(zcut=zcut, beta=beta, thetacut=thetacut)
        reader=Jets(args.data, args.nev)
        events=reader.values()
        imgs_ref=np.zeros((len(events), args.npx, args.npx))
        li_gen=LundImage(npxlx = args.npx, y_axis=args.yaxis)
        nsd_ref=[]
        for i, jet in enumerate(events): 
            tree = JetTree(jet)
            nsd_ref.append(sdmult(tree))
            imgs_ref[i]=li_gen(tree)
        imgref=np.average(imgs_ref,axis=0)
    else:
        imgref=None
    folder = args.output.strip('/')+'/' if args.output else ''

    assert(len(args.label_data_pairs)%2==0)
    filedic={}
    for i in range(0,len(args.label_data_pairs),2):
        lab=args.label_data_pairs[i]
        filedic[lab] = args.label_data_pairs[i+1]

    print('Plotting soft drop multiplicity')
    plot_sdmult(filedic, imgs_ref, folder+'softdropmult.pdf', nsd_ref, npx=args.npx,
                zcut=zcut, beta=beta, thetacut=thetacut)

#----------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot a kt and delta slice.')
    parser.add_argument('--data', type=str, help='The reference data file')
    parser.add_argument('--output', type=str, default='', help='Output folder')
    parser.add_argument('--npx',type=int, default=24, help='Pixel number')
    parser.add_argument('--nev',type=int, default=-1, help='Pixel number')
    parser.add_argument('label_data_pairs',  type=str, nargs='+',
                        help='List of label and generated data files.')
    parser.add_argument('--y-axis', type=str, dest='yaxis', help='Type of y axis')
    args = parser.parse_args()
    main(args)
