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
        model = WGAN(setup)
    elif input_model == 'wgangp':
        model = WGANGP(setup)
    elif input_model == 'vae':
        model = VAE(setup, length=length)
    elif input_model == 'dcgan':
        model = DCGAN(setup)
    elif input_model == 'gan':
        model = GAN(setup, length=length)
    elif input_model == 'bgan':
        model = BGAN(setup, length=length)
    elif input_model == 'lsgan':
        model = LSGAN(setup, length=length)
    elif input_model == 'aae':
        model = AdversarialAutoencoder(setup, length=length)
    else:
        raise Exception('build_model: invalid model choice')
    return model
