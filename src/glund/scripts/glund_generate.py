# This file is part of gLund by S. Carrazza and F. A. Dreyer

"""This script allows for the generation of new samples from a trained model"""

from glund.read_data import Jets
from glund.JetTree import JetTree, LundImage
from glund.preprocess import Averager
from glund.model import load_model_and_preprocessor
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import cm
import argparse, os, yaml

xval = [0.0, 7.0]
yval = [-3.0, 7.0]

def plot_events_debug(gen_sample, preproc, datafile, setup, folder):
    # load in the data
    reader=Jets(datafile, 5000)
    events=reader.values()
    img_data=np.zeros((len(events), setup['npx'], setup['npx'], 1))
    li_gen=LundImage(npxlx = setup['npx']) 
    for i, jet in enumerate(events): 
        tree = JetTree(jet) 
        img_data[i]=li_gen(tree).reshape(setup['npx'], setup['npx'], 1)

    # now reformat the training set as its average over n elements
    batch_averaged_img = np.zeros((len(img_data), setup['npx'], setup['npx'], 1))
    for i in range(len(img_data)):
        batch_averaged_img[i] = \
            np.average(img_data[np.random.choice(img_data.shape[0], setup['navg'],
                                                 replace=False), :], axis=0)
    img_input = batch_averaged_img

    # set up the preprocessed input
    img_unmask = preproc.unmask(preproc.transform(img_input))
    # set up the generated images
    gen_unmask = preproc.unmask(gen_sample)
    gen_final  = preproc.inverse(gen_sample)
    fig, ax=plt.subplots(figsize=(15,6), nrows=5,ncols=12)
    i=0 
    j=0 
    for row in ax: 
        for col in row:
            col.axis('off')
            if i%12<3:
                if i%3==1 and j==0:
                    col.set_title('Input image')
                col.imshow(img_input[i%3 + 3*j].reshape(setup['npx'],setup['npx']).transpose(),
                           vmin=0.0, vmax=0.5, origin='lower')
            elif i%12<6:
                if i%3==1 and j==0:
                    col.set_title('Preprocessed input')
                col.imshow(img_unmask[i%3 + 3*j].reshape(setup['npx'],setup['npx']).transpose(),
                           vmin=-1.0, vmax=1, cmap=cm.seismic, origin='lower')
            elif i%12<9:
                if i%3==1 and j==0:
                    col.set_title('Raw generated sample')
                col.imshow(gen_unmask[i%3 + 3*j].reshape(setup['npx'],setup['npx']).transpose(),
                           vmin=-1.0, vmax=1, cmap=cm.seismic, origin='lower')
            else:
                if i%3==1 and j==0:
                    col.set_title('Final generated sample')
                col.imshow(gen_final[i%3 + 3*j].reshape(setup['npx'],setup['npx']).transpose(),
                           vmin=0.0, vmax=0.5, origin='lower')
            i+=1
        j+=1
    plt.savefig(f'{folder}/plot_debug.pdf')

def plot_events(gen_sample, preproc, datafile, setup, folder):
    # load in the data
    reader=Jets(datafile, 5000)
    events=reader.values()
    img_data=np.zeros((len(events), setup['npx'], setup['npx'], 1))
    li_gen=LundImage(npxlx = setup['npx']) 
    for i, jet in enumerate(events): 
        tree = JetTree(jet) 
        img_data[i]=li_gen(tree).reshape(setup['npx'], setup['npx'], 1)

    # now reformat the training set as its average over n elements
    avg = Averager(setup['navg'])
    img_input = avg.transform(img_data)

    # set up the preprocessed input
    img_unmask = preproc.unmask(preproc.transform(img_input), invert_method=False)
    # set up the generated images
    gen_unmask = preproc.unmask(gen_sample, invert_method=False)
    gen_processed  = preproc.inverse(gen_sample)
    gen_final = avg.inverse(gen_processed)
    with PdfPages(f'{folder}/plot_events.pdf') as pdf:
        fig=plt.figure(figsize=(5,6))
        plt.title('Raw input')
        plt.imshow(img_data[0].reshape(setup['npx'],setup['npx']).transpose(),
                   vmin=-1.0, vmax=1.0, cmap=cm.seismic, origin='lower',
                   aspect='auto', extent=[xval[0], xval[1], yval[0], yval[1]])
        plt.xlabel('$\ln(1 / \Delta_{ab})$')
        plt.ylabel('$\ln(k_{t} / \mathrm{GeV})$')
        pdf.savefig()
        plt.close()
        
        fig=plt.figure(figsize=(5,6))
        plt.title('Averaged input')
        plt.imshow(img_input[0].reshape(setup['npx'],setup['npx']).transpose(),
                   vmin=-1.0, vmax=1.0, cmap=cm.seismic, origin='lower',
                   aspect='auto', extent=[xval[0], xval[1], yval[0], yval[1]])
        plt.xlabel('$\ln(1 / \Delta_{ab})$')
        plt.ylabel('$\ln(k_{t} / \mathrm{GeV})$')
        pdf.savefig()
        plt.close()
        
        fig=plt.figure(figsize=(5,6))
        plt.title('Preprocessed input')
        plt.imshow(img_unmask[0].reshape(setup['npx'],setup['npx']).transpose(),
                   vmin=-1.0, vmax=1.0, cmap=cm.seismic, origin='lower',
                   aspect='auto', extent=[xval[0], xval[1], yval[0], yval[1]])
        plt.xlabel('$\ln(1 / \Delta_{ab})$')
        plt.ylabel('$\ln(k_{t} / \mathrm{GeV})$')
        pdf.savefig()
        plt.close()

        fig=plt.figure(figsize=(5,6))
        plt.title('Raw generated sample')
        plt.imshow(gen_unmask[0].reshape(setup['npx'],setup['npx']).transpose(),
                   vmin=-1.0, vmax=1.0, cmap=cm.seismic, origin='lower',
                   aspect='auto', extent=[xval[0], xval[1], yval[0], yval[1]])
        plt.xlabel('$\ln(1 / \Delta_{ab})$')
        plt.ylabel('$\ln(k_{t} / \mathrm{GeV})$')
        pdf.savefig()
        plt.close()

        fig=plt.figure(figsize=(5,6))
        plt.title('Processed generated sample')
        plt.imshow(gen_processed[0].reshape(setup['npx'],setup['npx']).transpose(),
                   vmin=-1.0, vmax=1.0, cmap=cm.seismic, origin='lower',
                   aspect='auto', extent=[xval[0], xval[1], yval[0], yval[1]])
        plt.xlabel('$\ln(1 / \Delta_{ab})$')
        plt.ylabel('$\ln(k_{t} / \mathrm{GeV})$')
        pdf.savefig()
        plt.close()

        fig=plt.figure(figsize=(5,6))
        plt.title('Generated sample')
        plt.imshow(gen_final[0].reshape(setup['npx'],setup['npx']).transpose(),
                   vmin=-1.0, vmax=1.0, cmap=cm.seismic, origin='lower',
                   aspect='auto', extent=[xval[0], xval[1], yval[0], yval[1]])
        plt.xlabel('$\ln(1 / \Delta_{ab})$')
        plt.ylabel('$\ln(k_{t} / \mathrm{GeV})$')
        pdf.savefig()
        plt.close()

#----------------------------------------------------------------------
def main():
    # read in the arguments
    parser = argparse.ArgumentParser(description='Generate samples from a model.')
    parser.add_argument('model', action='store', type=str,
                        help='A folder containing a trained model')
    parser.add_argument('--ngen', '-n', type=int, default=10000, help='Generated images')
    parser.add_argument('--save', action='store_true', help='Save generated sample')
    parser.add_argument('--debugplots', action='store_true', help='Plot diagnostics')
    parser.add_argument('--plot', action='store_true', help='Plot diagnostics')
    parser.add_argument('--data', type=str, default=None, help='The reference data file')
    args = parser.parse_args()
    
    # check input is coherent
    if not (args.save or args.plot or args.debugplots):
        raise Exception('Invalid option: --save and/or --plot required')
    if not os.path.isdir(args.model):
        raise Exception('Invalid model: not a folder.')

    folder=args.model.strip('/')
    with open(f'{folder}/input-runcard.json','r') as stream: 
        setup=yaml.load(stream)

    model, preproc = load_model_and_preprocessor(folder, setup)

    gen_sample = model.generate(args.ngen)

    data = args.data if args.data else setup['data']
    if args.plot:
        plot_events(gen_sample, preproc, data, setup, folder)
    if args.debugplots:
        plot_events_debug(gen_sample, preproc, data, setup, folder)
    if args.save:
        np.save('%s/generated_images' % folder, gen_sample)
