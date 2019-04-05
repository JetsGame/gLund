# This file is part of gLund by S. Carrazza and F. A. Dreyer

from glund.models.lsgan import LSGAN
from glund.models.gan import GAN
from glund.models.bgan import BGAN
from glund.models.dcgan import DCGAN
from glund.models.wgan_gp import WGANGP
from glund.models.wgan import WGAN
from glund.models.vae import VAE
from glund.models.aae import AdversarialAutoencoder

#----------------------------------------------------------------------
def build_model(input_model, setup, length=None):
    """Return one of the generative models"""
    length=length if length else setup['npx']**2
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
