# This file is part of gLund by S. Carrazza and F. A. Dreyer

from glund.models.lsgan import LSGAN
from glund.models.gan import GAN
from glund.models.bgan import BGAN
from glund.models.dcgan import DCGAN
from glund.models.wgan_gp import WGANGP
from glund.models.wgan import WGAN
from glund.models.vae import VAE
from glund.models.aae import AdversarialAutoencoder
from glund.preprocess import load_preprocessor
import yaml

#----------------------------------------------------------------------
def build_model(setup, length=None):
    """Return one of the generative models"""
    length=length if length else setup['npx']**2
    input_model = setup['model']
    if input_model == 'wgan':
        print('[+] Setting up WGAN')
        model = WGAN(setup)
    elif input_model == 'wgangp':
        print('[+] Setting up WGANGP')
        model = WGANGP(setup)
    elif input_model == 'vae':
        print('[+] Setting up VAE')
        model = VAE(setup, length=length)
    elif input_model == 'dcgan':
        print('[+] Setting up DCGAN')
        model = DCGAN(setup)
    elif input_model == 'gan':
        print('[+] Setting up GAN')
        model = GAN(setup, length=length)
    elif input_model == 'bgan':
        print('[+] Setting up BGAN')
        model = BGAN(setup, length=length)
    elif input_model == 'lsgan':
        print('[+] Setting up LSGAN')
        model = LSGAN(setup, length=length)
    elif input_model == 'aae':
        print('[+] Setting up AAE')
        model = AdversarialAutoencoder(setup, length=length)
    else:
        raise Exception('build_model: invalid model choice')
    return model

#----------------------------------------------------------------------
def load_model_and_preprocessor(folder, setup):
    preproc = load_preprocessor(folder, setup)
    model = build_model(setup, preproc.length)
    model.load(folder)
    return model, preproc
