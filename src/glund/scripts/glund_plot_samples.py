# This file is part of gLund by S. Carrazza and F. A. Dreyer

"""This script provides diagnostic plots for generated lund images"""

import os, argparse
from glund.plotting import plot_lund, plot_lund_with_ref

        
#----------------------------------------------------------------------
def main():
    # read in the arguments
    parser = argparse.ArgumentParser(description='Plot a model.')
    parser.add_argument('--data', type=str, required=True, help='Generated images')
    parser.add_argument('--reference', type=str, default=None, help='Pythia reference')
    args = parser.parse_args()

    if args.reference:
        figname=args.data.split(os.extsep)[0]+'.pdf'
        plot_lund_with_ref(args.data, args.reference, figname)
    else:
        figname=args.data.split(os.extsep)[0]+'.pdf'
        plot_lund(args.data, figname)
