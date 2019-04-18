# This file is part of gLund by S. Carrazza and F. A. Dreyer

"""glund.py: the entry point for glund."""
import keras.backend as K
from glund.read_data import Jets
from glund.JetTree import JetTree, LundImage
from glund.tools import loss_calc, loss_calc_raw
from glund.model import build_model
from glund.preprocess import build_preprocessor, Averager
from glund.plotting import plot_lund_with_ref
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
    with open('%s/best-model.yaml' % folder, 'w') as wfp:
        yaml.dump(best_setup, wfp, default_flow_style=False)
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


#----------------------------------------------------------------------
def build_and_train_model(setup):
    """Training model"""
    print('[+] Training model')
    K.clear_session()
    if setup['model'] not in ('gan', 'dcgan', 'wgan', 'wgangp', 'vae',
                              'aae', 'bgan', 'lsgan'):
        raise ValueError('Invalid input: choose one model at a time.')

    # read in the data set
    if setup['data'] == 'mnist':
        print('[+] Loading mnist data')
        from keras.datasets import mnist
        # for debugging purposes, we have the option of loading in the
        # mnist data and training the model on this.
        (img_data, _), (_, _) = mnist.load_data()
        # Rescale -1 to 1
        if setup['model'] is not 'vae':
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
        li_gen=LundImage(npxlx = setup['npx'])
        for i, jet in enumerate(events): 
            tree = JetTree(jet) 
            img_data[i]=li_gen(tree).reshape(setup['npx'], setup['npx'], 1)

    avg = Averager(setup['navg'])
    img_train = avg.transform(img_data)
    
    # set up a preprocessing pipeline
    preprocess = build_preprocessor(setup)
    
    # prepare the training data for the model training
    print('[+] Fitting the preprocessor')
    preprocess.fit(img_train)

    # NB: for ZCA, the zca factor is set in the process.transform call
    img_train = preprocess.transform(img_train)
    
    # now set up the model
    model = build_model(setup, length=(img_train.shape[1]))

    # train on the images
    print('[+] Training the generative model')
    model.train(img_train, epochs=setup['epochs'],
                batch_size = setup['batch_size'])

    # now generate a test sample and save it
    gen_sample = model.generate(setup['ngen'])

    # get the raw loss, evaluated on gan input and generated sample
    print('[+] Calculating loss on raw training data')
    loss_raw = loss_calc_raw(preprocess.unmask(gen_sample[:min(setup['ngen'],len(img_train))]),
                             preprocess.unmask(img_train[:min(setup['ngen'],len(img_train))]))
    
    # retransform the generated sample to image space
    gen_sample = preprocess.inverse(gen_sample)
    gen_sample = avg.inverse(gen_sample)

    # get reference sample and generated sample for tests
    ref_sample = img_data.reshape(img_data.shape[0],setup['npx'],setup['npx'])\
        [np.random.choice(img_data.shape[0], len(gen_sample), replace=True), :]

    print('[+] Calculating final loss after inverting preprocessing')
    loss = (loss_calc(gen_sample,ref_sample), loss_raw)
    if setup['scan']:
        res = {'loss': loss[0] if setup['monitor_final_loss'] else loss[1],
               'status': STATUS_OK}
    else:
        res = model, gen_sample, loss, preprocess
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
    parser.add_argument('--plot-samples', action='store_true', dest='plot_samples',
                        help='Generate document with reference and model samples')
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
    
    # build and train the model
    model, gen_sample, loss, preproc = build_and_train_model(setup)

    # write out a file with basic information on the run
    with open('%s/info.txt' % folder,'w') as f:
        print('# %s' % model.description(), file=f)
        print('# created on %s with the command:'
              % datetime.datetime.utcnow(), file=f)
        print('# '+' '.join(sys.argv), file=f)
        print('# final loss:\t%f\traw loss:\t%f' % loss, file=f)

    # copy runcard to output folder
    shutil.copyfile(args.runcard, f'{folder}/input-runcard.json')

    # save the model to file
    model.save(folder)

    # save the preprocessor
    preproc.save(folder)
    
    # save a generated sample to file and plot the average image
    genfn = '%s/generated_images' % folder
    np.save(genfn, gen_sample)

    if args.plot_samples:
        plot_lund_with_ref(f'{genfn}.npy', setup['data'], f'{genfn}.pdf')
