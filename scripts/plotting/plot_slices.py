# This file is part of gLund by S. Carrazza and F. A. Dreyer
import matplotlib.pyplot as plt
from glund.read_data import Jets 
from glund.JetTree import JetTree, LundImage
import numpy as np
import argparse,math

    
#----------------------------------------------------------------------
def plot_slice_kt(filedic, imgref, figname, npx=24):
    """Plot a slice in kt of the lund image for different models and a reference sample."""
    img = {}
    for lab in filedic.keys():
        img[lab] = np.average(np.load(filedic[lab]), axis=0)
    yvals=np.linspace(-3, 7, npx+1)
    xvals=np.linspace(0,  7, npx+1)

    if imgref is not None:
        img['Pythia 8'] = imgref

    # get kt slice. This is tuned for 24 pixel images
    kt_min = math.exp(yvals[12])
    kt_max = math.exp(yvals[15])
    xbins=np.array([0.5*(xvals[i]+xvals[i+1]) for i in range(len(xvals)-1)])
    fig = plt.figure(figsize=(5.5,3.2))
    for lab in img.keys():
        print(lab)
        kt_slice = np.sum(img[lab][:,12:15],axis=1)
        plt.semilogx(np.exp(-xbins), kt_slice, label=lab)
    plt.xlim([1.0, 0.01])
    plt.ylim([0.0, 0.15])
    #plt.yticks(ticks=[0.02, 0.04, 0.1, 0.2, 0.4], labels=['0.02','0.04','0.1','0.2','0.4'])
    plt.xticks(ticks=[1, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01],
               labels=['1', '0.5', '0.2', '0.1', '0.05','0.02', '0.01'])
    plt.xlabel('$\\Delta$')
    plt.ylabel('$\\rho(\\Delta, \\mathrm{fixed}\;k_t)$')
    plt.text(0.8,0.01,'$%.2f < k_t\, \mathrm{[GeV]} < %.2f$' % (kt_min,kt_max))
    fig.subplots_adjust(bottom=0.15)
    plt.legend()
    plt.savefig(figname)
    plt.close()
    
#----------------------------------------------------------------------
def plot_slice_delta(filedic, imgref, figname, npx=24):
    img = {}
    for lab in filedic.keys():
        img[lab] = np.average(np.load(filedic[lab]), axis=0)
    xvals=np.linspace(0,  7, npx+1)
    yvals=np.linspace(-3, 7, npx+1)

    if imgref is not None:
        img['Pythia 8']=imgref
    
    # get delta R slice. This is tuned for 24 pixel images
    delta_max = math.exp(-xvals[4])
    delta_min = math.exp(-xvals[7])
    ybins=np.array([0.5*(yvals[i]+yvals[i+1]) for i in range(len(yvals)-1)])
    fig = plt.figure(figsize=(5.5,3.2))
    for lab in img.keys():
        print(lab)
        delta_slice = np.sum(img[lab][4:7,:],axis=0)
        plt.loglog(np.exp(ybins), delta_slice, label=lab)
    plt.xlim([0.1, 100])
    plt.ylim([0.02, 0.4])
    plt.yticks(ticks=[0.02, 0.04, 0.1, 0.2, 0.4], labels=['0.02','0.04','0.1','0.2','0.4'])
    plt.xticks(ticks=[0.1, 1, 10, 100],labels=['0.1', '1', '10', '100'])
    plt.xlabel('$k_t$ [GeV]')
    plt.ylabel('$\\rho(\\mathrm{fixed}\;\\Delta, k_t)$')
    plt.text(0.15,0.025,'$%.2f < \\Delta < %.2f$' % (delta_min,delta_max))
    fig.subplots_adjust(bottom=0.15)
    plt.legend()
    plt.savefig(figname)
    plt.close()

#----------------------------------------------------------------------
def main(args):
    if args.data:
        reader=Jets(args.data, args.nev)
        events=reader.values()
        imgref=np.zeros((len(events), args.npx, args.npx))
        li_gen=LundImage(npxlx = args.npx)
        for i, jet in enumerate(events): 
            tree = JetTree(jet) 
            imgref[i]=li_gen(tree)
        imgref=np.average(imgref,axis=0)
    else:
        imgref=None
    folder = args.output.strip('/')+'/' if args.output else ''

    assert(len(args.label_data_pairs)%2==0)
    filedic={}
    for i in range(0,len(args.label_data_pairs),2):
        lab=args.label_data_pairs[i]
        filedic[lab] = args.label_data_pairs[i+1]

    print('Plotting delta slice')
    plot_slice_delta(filedic, imgref, folder+'delta_slice.pdf', args.npx)
    print('Plotting kt slice')
    plot_slice_kt(filedic, imgref, folder+'kt_slice.pdf', args.npx)

#----------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot a kt and delta slice.')
    parser.add_argument('--data', type=str, default=None, 
                        help='The reference data file')
    parser.add_argument('--output', type=str, default='', help='Output folder')
    parser.add_argument('--npx',type=int, default=24, help='Pixel number')
    parser.add_argument('--nev',type=int, default=-1, help='Pixel number')
    parser.add_argument('label_data_pairs',  type=str, nargs='+',
                        help='List of label and generated data files.')
    args = parser.parse_args()
    main(args)
