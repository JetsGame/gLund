# This file is part of gLund by S. Carrazza and F. A. Dreyer
import matplotlib.pyplot as plt
from glund.read_data import Jets 
from glund.JetTree import JetTree, LundImage
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import argparse,math

#----------------------------------------------------------------------
def plot_activation(filedic, imgs_ref, figname):
    """Plot a slice in kt of the lund image for different models and a reference sample."""
    act = {}
    for lab in filedic.keys():
        imgs = np.load(filedic[lab])
        act[lab] = []
        for i in range(len(imgs)):
            act[lab].append(np.sum(imgs[i]))
    act_ref = []
    for i in range(len(imgs_ref)):
        act_ref.append(np.sum(imgs_ref[i]))
    fig, ax = plt.subplots(figsize=(5,3.5))
    bins = np.arange(0, 101, 1)
    for lab in act.keys():
        plt.hist(act[lab], bins=bins, histtype='step', density=True, label=lab)
    plt.hist(act_ref, bins=bins, histtype='step', density=True, label='Pythia 8')
    ax.set_xlim((0,30))
    ax.set_ylim((0.0,0.14))
    ax.set_xlabel('# activated pixels')
    plt.legend()
    ax.grid(linestyle=':')
    plt.savefig(figname, bbox_inches='tight')
    plt.close()

#----------------------------------------------------------------------
def plot_slice_kt(filedic, imgref, figname, npx=24):
    """Plot a slice in kt of the lund image for different models and a reference sample."""
    img = {}
    for lab in filedic.keys():
        img[lab] = np.average(np.load(filedic[lab]), axis=0)
    yvals=np.linspace(-3, 7, npx+1)
    xvals=np.linspace(0,  7, npx+1)

    # get kt slice. This is tuned for 24 pixel images
    kt_min = math.exp(yvals[12])
    kt_max = math.exp(yvals[15])
    xbins=np.array([0.5*(xvals[i]+xvals[i+1]) for i in range(len(xvals)-1)])
    # plotting
    fig, ax = plt.subplots(figsize=(4.5,5))
    fig.subplots_adjust(left=0.15)
    ax.axis([1.0, 0.01, 0.0, 0.15])
    ax.set_xscale('log')
    plt.xticks(ticks=[1, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01],
               labels=['', '', '', '', '','', ''])
    ax.set_axisbelow(True)
    for lab in img.keys():
        kt_slice = np.sum(img[lab][:,12:15],axis=1)
        plt.plot(np.exp(-xbins), kt_slice, label=lab)
    plt.plot(np.exp(-xbins), np.sum(imgref[:,12:15], axis=1), label='Pythia 8')
    plt.legend()
    ax.grid(linestyle=':')
    ax.set_ylabel('$\\rho(\\Delta_{ab}, \\mathrm{fixed}\;k_t)$')
    plt.text(0.8,0.01,'$%.1f < k_t\, \mathrm{[GeV]} < %.1f$' % (kt_min,kt_max))
    # now the ratio
    divider = make_axes_locatable(ax)
    axratio = divider.append_axes("bottom",1.6, pad=0.0)
    axratio.axis([1.0, 0.01, 0.6, 1.4])
    plt.yticks(ticks=[0.6, 0.8, 1, 1.2])
    axratio.set_xscale('log')
    axratio.set_xlabel('$\\Delta_{ab}$')
    axratio.set_ylabel('ratio to Pythia 8')
    for lab in img.keys():
        if lab!='Pythia 8':
            kt_slice = np.sum(img[lab][:,12:15],axis=1)
            plt.plot(np.exp(-xbins), kt_slice/np.sum(imgref[:,12:15],axis=1), label=lab)
    plt.xticks(ticks=[1, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01],
               labels=['1', '0.5', '0.2', '0.1', '0.05','0.02', '0.01'])
    #fig.subplots_adjust(bottom=0.15)
    axratio.grid(linestyle=':')
    plt.savefig(figname)
    plt.close()

#----------------------------------------------------------------------
def plot_slice_delta(filedic, imgref, figname, npx=24):
    img = {}
    for lab in filedic.keys():
        img[lab] = np.average(np.load(filedic[lab]), axis=0)
    xvals=np.linspace(0,  7, npx+1)
    yvals=np.linspace(-3, 7, npx+1)

    # get delta R slice. This is tuned for 24 pixel images
    delta_max = math.exp(-xvals[4])
    delta_min = math.exp(-xvals[7])
    ybins=np.array([0.5*(yvals[i]+yvals[i+1]) for i in range(len(yvals)-1)])
    # plotting
    fig, ax = plt.subplots(figsize=(4.5,5))
    fig.subplots_adjust(left=0.15)
    ax.axis([0.1, 100, 0.02, 0.4])
    ax.set_xscale('log')
    ax.set_yscale('log')
    plt.yticks(ticks=[0.02, 0.04, 0.1, 0.2, 0.4], labels=['0.02','0.04','0.1','0.2','0.4'])
    plt.xticks(ticks=[0.1, 1, 10, 100],labels=['', '', '', ''])
    ax.set_axisbelow(True)
    for lab in img.keys():
        delta_slice = np.sum(img[lab][4:7,:],axis=0)
        plt.plot(np.exp(ybins), delta_slice, label=lab)
    plt.plot(np.exp(ybins), np.sum(imgref[4:7,:], axis=0), label='Pythia 8')
    plt.legend()
    ax.grid(linestyle=':')
    ax.set_ylabel('$\\rho(\\mathrm{fixed}\;\\Delta_{ab}, k_t)$')
    plt.text(0.15,0.025,'$%.2f < \\Delta_{ab} < %.2f$' % (delta_min,delta_max))
    # no the ratio
    divider = make_axes_locatable(ax)
    axratio = divider.append_axes("bottom",1.6, pad=0.0)
    axratio.axis([0.1, 100, 0.6, 1.4])
    plt.yticks(ticks=[0.6, 0.8, 1, 1.2])
    axratio.set_xscale('log')
    axratio.set_xlabel('$k_t$ [GeV]')
    axratio.set_ylabel('ratio to Pythia 8')
    for lab in img.keys():
        delta_slice = np.sum(img[lab][4:7,:],axis=0)
        plt.plot(np.exp(ybins), delta_slice/np.sum(imgref[4:7,:],axis=0), label=lab)
    plt.xticks(ticks=[0.1, 1, 10, 100],labels=['0.1', '1', '10', '100'])
    axratio.grid(linestyle=':')
    plt.savefig(figname)
    plt.close()

    
#----------------------------------------------------------------------
def plot_slice_kt_noratio(filedic, figname, npx=24):
    """Plot a slice in kt of the lund image for different models and a reference sample."""
    img = {}
    for lab in filedic.keys():
        img[lab] = np.average(np.load(filedic[lab]), axis=0)
    yvals=np.linspace(-3, 7, npx+1)
    xvals=np.linspace(0,  7, npx+1)

    # get kt slice. This is tuned for 24 pixel images
    kt_min = math.exp(yvals[12])
    kt_max = math.exp(yvals[15])
    xbins=np.array([0.5*(xvals[i]+xvals[i+1]) for i in range(len(xvals)-1)])
    fig = plt.figure(figsize=(5.5,3.2))
    for lab in img.keys():
        kt_slice = np.sum(img[lab][:,12:15],axis=1)
        plt.semilogx(np.exp(-xbins), kt_slice, label=lab)
    plt.xlim([1.0, 0.01])
    plt.ylim([0.0, 0.15])
    #plt.yticks(ticks=[0.02, 0.04, 0.1, 0.2, 0.4], labels=['0.02','0.04','0.1','0.2','0.4'])
    plt.xticks(ticks=[1, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01],
               labels=['1', '0.5', '0.2', '0.1', '0.05','0.02', '0.01'])
    plt.xlabel('$\\Delta$')
    plt.ylabel('$\\rho(\\Delta, \\mathrm{fixed}\;k_t)$')
    plt.text(0.8,0.01,'$%.1f < k_t\, \mathrm{[GeV]} < %.1f$' % (kt_min,kt_max))
    fig.subplots_adjust(bottom=0.15)
    plt.legend()
    plt.savefig(figname)
    plt.close()

#----------------------------------------------------------------------
def plot_slice_delta_noratio(filedic, figname, npx=24):
    img = {}
    for lab in filedic.keys():
        img[lab] = np.average(np.load(filedic[lab]), axis=0)
    xvals=np.linspace(0,  7, npx+1)
    yvals=np.linspace(-3, 7, npx+1)
    
    # get delta R slice. This is tuned for 24 pixel images
    delta_max = math.exp(-xvals[4])
    delta_min = math.exp(-xvals[7])
    ybins=np.array([0.5*(yvals[i]+yvals[i+1]) for i in range(len(yvals)-1)])
    fig = plt.figure(figsize=(5.5,3.2))
    for lab in img.keys():
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
        imgs_ref=np.zeros((len(events), args.npx, args.npx))
        li_gen=LundImage(npxlx = args.npx)
        for i, jet in enumerate(events): 
            tree = JetTree(jet) 
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

    print('Plotting slices')
    if imgref is not None:
        plot_activation(filedic, imgs_ref, folder+'activation.pdf')
        plot_slice_delta(filedic, imgref, folder+'delta_slice.pdf', args.npx)
        plot_slice_kt(filedic, imgref, folder+'kt_slice.pdf', args.npx)
    else:
        plot_slice_delta_noratio(filedic, folder+'delta_slice.pdf', args.npx)
        plot_slice_kt_noratio(filedic, folder+'kt_slice.pdf', args.npx)

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
