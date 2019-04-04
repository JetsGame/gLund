# This file is part of gLund by S. Carrazza and F. A. Dreyer
from glund.read_data import Jets
from glund.JetTree import JetTree, LundImage
from glund.tools import loss_calc
from glund.model import *
from glund.preprocess import PreprocessPCA, PreprocessZCA
import matplotlib.pyplot as plt
import numpy as np
import argparse, os, shutil, sys, datetime
import yaml


#----------------------------------------------------------------------
def load_yaml(runcard_file):
    """Loads yaml runcard"""
    with open(runcard_file, 'r') as stream:
        runcard = yaml.load(stream)
    return runcard


#----------------------------------------------------------------------
def main():
    """Parsing command line arguments"""
    parser = argparse.ArgumentParser(description='Train a generative model.')
    parser.add_argument('runcard', action='store', type=str,
                        help='A yaml file with the setup.')
    parser.add_argument('--output', '-o', type=str, required=True, 
                        help='The output folder')
    parser.add_argument('--force', '-f', action='store_true', dest='force',
                        help='Overwrite existing files if present.')
    args = parser.parse_args()

    # check input are coherent
    if not os.path.isfile(args.runcard):
        raise ValueError('Invalid runcard: not a file.')
    if args.force:
        print('WARNING: Running with --force option will overwrite existing model.')

    # load runcard
    setup = load_yaml(args.runcard)
    input_model = setup['model']

    # check that input is valid
    if input_model not in ('gan', 'dcgan', 'wgan', 'wgangp', 'vae', 'aae', 'bgan', 'lsgan'):
        raise ValueError('Invalid input: choose one model at a time.')
    if os.path.exists(args.output) and not args.force:
        raise Exception(f'{args.output} already exists, use "--force" to overwrite.')

    # for GAN or VAE, we want to flatten the input and preprocess it
    flat_input = input_model in ('gan', 'vae', 'bgan', 'aae', 'lsgan')

    # read in the data set
    if setup['data'] is 'mnist':
        from keras.datasets import mnist
        # for debugging purposes, we have the option of loading in the
        # mnist data and training the model on this.
        (img_train, _), (_, _) = mnist.load_data()
        # Rescale -1 to 1
        if input_model is not 'vae':
            img_train = (img_train.astype(np.float32) - 127.5) / 127.5
        else:
            img_train = img_train.astype('float32') / 255
        img_train = np.expand_dims(img_train, axis=3)
    else:
        # load in the jets from file, and create an array of lund images
        reader=Jets(setup['data'], setup['nev'])
        events=reader.values()
        img_train=np.zeros((len(events), setup['npx'], setup['npx'], 1))
        if setup['deterministic']:
            li_gen = LundImage(npxlx = setup['npx'])
        else:
            li_gen=LundImage(npxlx = setup['npx'], norm_to_one=True) 
        for i, jet in enumerate(events): 
            tree = JetTree(jet) 
            img_train[i]=li_gen(tree).reshape(setup['npx'], setup['npx'], 1)

    if not setup['deterministic']:
        # now reformat the training set as its average over n elements
        nev = max(len(img_train),setup['nev'])
        batch_averaged_img = np.zeros((nev, setup['npx'], setup['npx'], 1))
        for i in range(nev):
            batch_averaged_img[i] = \
                np.average(img_train[np.random.choice(img_train.shape[0], setup['navg'],
                                                    replace=False), :], axis=0)
        img_train = batch_averaged_img
        # img_train=np.array([np.average(img_train[args.navg*i:args.navg*i+args.navg],axis=0)
        #                     for i in range(len(img_train)//args.navg)])

    # if requested, set up a preprocessing pipeline
    if setup['pca']:
        preprocess = PreprocessPCA(setup['pca_fraction'], whiten=False)
    elif setup['zca']:
        preprocess = PreprocessZCA(flatten=flat_input, remove_zero=flat_input)

    # prepare the training data for the model training
    if setup['pca'] or setup['zca']:
        preprocess.fit(img_train)
        # NB: for ZCA, the zca factor is set in the process.transform call
        img_train = preprocess.transform(img_train)
    elif flat_input:
        img_train = img_train.reshape(-1, setup['npx']*setup['npx'])

    # now set up the model
    if input_model == 'wgan':
        model = WGAN(width=setup['npx'], height=setup['npx'], latent_dim=setup['latdim'])
    elif input_model == 'wgangp':
        model = WGANGP(width=setup['npx'], height=setup['npx'], latent_dim=setup['latdim'])
    elif input_model == 'vae':
        model = VAE(length=(img_train.shape[1]), latent_dim=setup['latdim'], mse_loss=False)
    elif input_model == 'dcgan':
        model = DCGAN(width=setup['npx'], height=setup['npx'], latent_dim=setup['latdim'])
    elif input_model == 'gan':
        model = GAN(length=(img_train.shape[1]), latent_dim=setup['latdim'])
    elif input_model == 'bgan':
        model = BGAN(length=(img_train.shape[1]), latent_dim=setup['latdim'])
    elif input_model == 'lsgan':
        model = LSGAN(length=(img_train.shape[1]), latent_dim=setup['latdim'])
    elif input_model == 'aae':
        model = AdversarialAutoencoder(length=(img_train.shape[1]), latent_dim=setup['latdim'])

    # train on the images
    model.train(img_train, epochs=setup['epochs'],
                batch_size = setup['batch_size'])

    # now generate a test sample and save it
    gen_sample = model.generate(setup['ngen'])

    # retransform the generated sample to image space
    if setup['pca'] or setup['zca']:
        gen_sample = preprocess.inverse(gen_sample)
    else:
        gen_sample = gen_sample.reshape(setup['ngen'], setup['npx'], setup['npx'])

    if not setup['deterministic']:
        # now interpret the probabilistic sample as physical images
        for i,v in np.ndenumerate(gen_sample):
            if v < np.random.uniform(0,1):
                gen_sample[i]=0.0
            else:
                gen_sample[i]=1.0
        
    # prepare the output folder
    if not os.path.exists(args.output):
        os.mkdir(args.output)
    elif args.force:
        shutil.rmtree(args.output)
        os.mkdir(args.output)
    else:
        raise Exception(f'{args.output} already exists, use "--force" to overwrite.')
    folder = args.output.strip('/')

    # for loss function, define epsilon and retransform the training sample
    epsilon=0.3
    # get reference sample and generated sample for tests
    if setup['pca'] or setup['zca']:
        img_train = preprocess.inverse(img_train)
    ref_sample = img_train.reshape(img_train.shape[0],setup['npx'],setup['npx'])\
        [np.random.choice(img_train.shape[0], len(gen_sample), replace=True), :]

    # write out a file with basic information on the run
    with open('%s/info.txt' % folder,'w') as f:
        print('# %s' % model.description(), file=f)
        print('# created on %s with the command:' % datetime.datetime.utcnow(), file=f)
        print('# '+' '.join(sys.argv), file=f)
        print('# loss = %f' % loss_calc(gen_sample,ref_sample,epsilon), file=f)

    # save the model to file
    model.save(folder)

    # save a generated sample to file and plot the average image
    genfn = '%s/generated_images' % folder
    np.save(genfn, gen_sample)
    plt.imshow(np.average(gen_sample, axis=0))
    plt.savefig('%s/average_image.png' % folder)
