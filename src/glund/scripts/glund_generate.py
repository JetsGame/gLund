# This file is part of gLund by S. Carrazza and F. A. Dreyer

"""This script allows for the generation of new samples from a trained model"""

from glund.read_data import Jets
from glund.JetTree import JetTree, LundImage
from glund.model import load_model_and_preprocessor
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import argparse, os, yaml

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
                col.imshow(img_input[j].reshape(setup['npx'],setup['npx']),vmin=0.0,vmax=0.5)
            elif i%12<6:
                if i%3==1 and j==0:
                    col.set_title('Preprocessed input')
                col.imshow(img_unmask[j].reshape(setup['npx'],setup['npx']),vmin=-1.0,vmax=1,cmap=cm.seismic)
            elif i%12<9:
                if i%3==1 and j==0:
                    col.set_title('Raw generated sample')
                col.imshow(gen_unmask[j].reshape(setup['npx'],setup['npx']),vmin=-1.0,vmax=1,cmap=cm.seismic)
            else:
                if i%3==1 and j==0:
                    col.set_title('Final generated sample')
                col.imshow(gen_final[j].reshape(setup['npx'],setup['npx']),vmin=0.0,vmax=0.5)
            i+=1
        j+=1
    plt.savefig(f'{folder}/plot_events.pdf')

#----------------------------------------------------------------------
def main():
    # read in the arguments
    parser = argparse.ArgumentParser(description='Generate samples from a model.')
    parser.add_argument('model', action='store', type=str,
                        help='A folder containing a trained model')
    parser.add_argument('--ngen', '-n', type=int, default=10000, help='Generated images')
    parser.add_argument('--save', action='store_true', help='Save generated sample')
    parser.add_argument('--plot', action='store_true', help='Plot diagnostics')
    parser.add_argument('--data', type=str, default=None, help='The reference data file')
    args = parser.parse_args()
    
    # check input is coherent
    if not (args.save or args.plot):
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
    if args.save:
        np.save('%s/generated_images' % folder, gen_sample)
