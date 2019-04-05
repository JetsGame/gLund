# This file is part of gLund by S. Carrazza and F. A. Dreyer

"""glund.py: the entry point for glund."""
import keras.backend as K
from glund.read_data import Jets
from glund.JetTree import JetTree, LundImage
from glund.tools import loss_calc
from glund.model import build_model
from glund.preprocess import build_preprocessor
from hyperopt import fmin, tpe, hp, Trials, space_eval, STATUS_OK
from hyperopt.mongoexp import MongoTrials
import matplotlib.pyplot as plt
import numpy as np
from time import time
import argparse, os, shutil, sys, datetime, yaml, pprint, pickle


#----------------------------------------------------------------------
def run_hyperparameter_scan(search_space, max_evals, cluster, folder):
    """Running hyperparameter scan using hyperopt"""
    print('[+] Performing hyperparameter scan...')
    if cluster:
        trials = MongoTrials(cluster, exp_key='exp1')
    else:
        trials = Trials()
    best = fmin(build_and_train_model, search_space, algo=tpe.suggest, 
                max_evals=max_evals, trials=trials)
    best_setup = space_eval(search_space, best)
    print('\n[+] Best scan setup:')
    pprint.pprint(best_setup)
    log = '%s/hyperopt_log_{}.pickle'.format(time()) % folder
    with open(log, 'wb') as wfp:
        print(f'[+] Saving trials in {log}')
        pickle.dump(trials.trials, wfp)
    return best_setup


#----------------------------------------------------------------------
def load_yaml(runcard_file):
    """Loads yaml runcard"""
    with open(runcard_file, 'r') as stream:
        runcard = yaml.load(stream)
    for key, value in runcard.items():
        if 'hp.' in str(value):
            runcard[key] = eval(value)
    return runcard



def build_and_train_model(setup):
    """Training model"""
    print('[+] Training model')
    K.clear_session()
    input_model = setup['model']
    if input_model not in ('gan', 'dcgan', 'wgan', 'wgangp', 'vae',
                           'aae', 'bgan', 'lsgan'):
        raise ValueError('Invalid input: choose one model at a time.')

    # read in the data set
    if setup['data'] is 'mnist':
        print('[+] Loading mnist data')
        from keras.datasets import mnist
        # for debugging purposes, we have the option of loading in the
        # mnist data and training the model on this.
        (img_data, _), (_, _) = mnist.load_data()
        # Rescale -1 to 1
        if input_model is not 'vae':
            img_data = (img_data.astype(np.float32) - 127.5) / 127.5
        else:
            img_data = img_data.astype('float32') / 255
        img_data = np.expand_dims(img_data, axis=3)
    else:
        # load in the jets from file, and create an array of lund images
        print('[+] Reading jet data from file')
        reader=Jets(setup['data'], setup['nev'])
        events=reader.values()
        img_data=np.zeros((len(events), setup['npx'], setup['npx'], 1))
        if setup['deterministic']:
            li_gen = LundImage(npxlx = setup['npx'])
        else:
            li_gen=LundImage(npxlx = setup['npx'], norm_to_one=True) 
        for i, jet in enumerate(events): 
            tree = JetTree(jet) 
            img_data[i]=li_gen(tree).reshape(setup['npx'], setup['npx'], 1)

    if not setup['deterministic']:
        # now reformat the training set as its average over n elements
        nev = max(len(img_data),setup['nev'])
        batch_averaged_img = np.zeros((nev, setup['npx'], setup['npx'], 1))
        for i in range(nev):
            batch_averaged_img[i] = \
                np.average(img_data[np.random.choice(img_data.shape[0], setup['navg'],
                                                    replace=False), :], axis=0)
        img_train = batch_averaged_img
    else:
        img_train = img_data

    # set up a preprocessing pipeline
    preprocess = build_preprocessor(input_model, setup)
    
    # prepare the training data for the model training
    print('[+] Fitting the preprocessor')
    preprocess.fit(img_train)

    # NB: for ZCA, the zca factor is set in the process.transform call
    img_train = preprocess.transform(img_train)
    
    # now set up the model
    model = build_model(input_model, setup, length=(img_train.shape[1]))

    # train on the images
    print('[+] Training the generative model')
    model.train(img_train, epochs=setup['epochs'],
                batch_size = setup['batch_size'])

    # now generate a test sample and save it
    gen_sample = model.generate(setup['ngen'])

    # retransform the generated sample to image space
    gen_sample = preprocess.inverse(gen_sample)

    if not setup['deterministic']:
        # now interpret the probabilistic sample as physical images
        for i,v in np.ndenumerate(gen_sample):
            if v < np.random.uniform(0,1):
                gen_sample[i]=0.0
            else:
                gen_sample[i]=1.0

    # for loss function, define epsilon and retransform the training sample
    epsilon=0.5
    # get reference sample and generated sample for tests
    ref_sample = img_data.reshape(img_data.shape[0],setup['npx'],setup['npx'])\
        [np.random.choice(img_data.shape[0], len(gen_sample), replace=True), :]

    loss = loss_calc(gen_sample,ref_sample,epsilon)
    if setup['scan']:
        res = {'loss': loss, 'status': STATUS_OK}
    else:
        res = model, gen_sample, loss
    return res


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
    parser.add_argument('--hyperopt', default=None, type=int,
                        help='Enable hyperopt scan.')
    parser.add_argument('--cluster', default=None, 
                        type=str, help='Enable cluster scan.')
    args = parser.parse_args()

    # check input is coherent
    if not os.path.isfile(args.runcard):
        raise ValueError('Invalid runcard: not a file.')
    if args.force:
        print('WARNING: Running with --force option will overwrite existing model.')

    # prepare the output folder
    if not os.path.exists(args.output):
        os.mkdir(args.output)
    elif args.force:
        shutil.rmtree(args.output)
        os.mkdir(args.output)
    else:
        raise Exception(f'{args.output} already exists, use "--force" to overwrite.')
    folder = args.output.strip('/')

    # load runcard
    setup = load_yaml(args.runcard)
    
    if args.hyperopt:
        setup['scan'] = True
        setup = run_hyperparameter_scan(setup, args.hyperopt, 
                                        args.cluster, folder)
    setup['scan'] = False
    
    print('[+] Training best model:')
    # build and train the model
    model, gen_sample, loss = build_and_train_model(setup)

    # write out a file with basic information on the run
    with open('%s/info.txt' % folder,'w') as f:
        print('# %s' % model.description(), file=f)
        print('# created on %s with the command:'
              % datetime.datetime.utcnow(), file=f)
        print('# '+' '.join(sys.argv), file=f)
        print('# loss = %f' % loss, file=f)

    # save the model to file
    model.save(folder)

    # save a generated sample to file and plot the average image
    genfn = '%s/generated_images' % folder
    np.save(genfn, gen_sample)
    plt.imshow(np.average(gen_sample, axis=0))
    plt.savefig('%s/average_image.png' % folder)

    