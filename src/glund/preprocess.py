# This file is part of gLund by S. Carrazza and F. A. Dreyer

from glund.models.autoencoder import Autoencoder
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from glund.tools import zca_whiten
import numpy as np

#----------------------------------------------------------------------
def build_preprocessor(input_model, setup):
    flat_input = input_model in ('gan', 'vae', 'bgan', 'aae', 'lsgan')
    if setup['pca']:
        if not flat_input:
            raise Exception('build_preprocessor: pca unavailable for this model')
        print('[+] Setting up PCA preprocessing pipeline')
        preprocess = PreprocessorPCA(setup['pca_fraction'], whiten=False)
    elif setup['zca']:
        print('[+] Setting up ZCA preprocessing pipeline')
        preprocess = PreprocessorZCA(scaler=True, flatten=flat_input,
                                     remove_zero=flat_input)
    else:
        print('[+] Setting up minimal preprocessing pipeline')
        preprocess = Preprocessor(scaler=True, flatten=flat_input,
                                  remove_zero=flat_input)
    return preprocess


#======================================================================
class Preprocessor:
    """Preprocessing pipeline"""
    
    #----------------------------------------------------------------------
    def __init__(self, scaler, flatten, remove_zero):
        self.scaler = StandardScaler() if scaler else None
        if not flatten and remove_zero:
            raise Exception('Preprocess: can not mask zero entries for unflattened inputs')
        self.remove_zero = remove_zero
        self.flatten = flatten
        self.shape = None

    #----------------------------------------------------------------------    
    def fit(self, data):
        """Set up the preprocessing pipeline"""
        self.shape = data.shape[1:]
        data = data.reshape(-1, data.shape[1]*data.shape[2])
        self.zero_mask = np.where(np.sum(np.square(data), axis=0) > 0.0)[0]
        if self.remove_zero:
            data = data[:,self.zero_mask]
        if self.scaler:
            self.scaler.fit(data)
        self._method_fit(data)

    #----------------------------------------------------------------------
    def transform(self, data):
        """Apply preprocessing to input data"""
        data = data.reshape(-1, data.shape[1]*data.shape[2])
        if self.remove_zero:
            data = data[:,self.zero_mask]
        if self.scaler:
            data = self.scaler.transform(data)
        data = self._method_transform(data)
        if not self.flatten:
            return data.reshape((len(data),)+self.shape)
        return data

    #----------------------------------------------------------------------
    def inverse(self, data):
        """Return to image space from preprocessed input"""
        if not self.flatten:
            data = data.reshape(-1, self.shape[0]*self.shape[1])
        data = self._method_inverse(data)
        if self.scaler:
            data = self.scaler.inverse_transform(data)
        if self.remove_zero:
            result = np.zeros((len(data),self.shape[0]*self.shape[1]))
            result[:, self.zero_mask] = data[:,:]
        else:
            result = data
        if not self.flatten:
            result = self.mask(result)
        return result.reshape(len(data),self.shape[0],self.shape[1])

    #----------------------------------------------------------------------
    def unmask(self, data, invert_method=True):
        """Return to image space, but without inverting the scaler. Only for loss calculation!"""
        if not self.flatten:
            data = data.reshape(-1, self.shape[0]*self.shape[1])
        if invert_method:
            data = self._method_inverse(data)
        if self.remove_zero:
            result = np.zeros((len(data),self.shape[0]*self.shape[1]))
            result[:, self.zero_mask] = data[:,:]
        else:
            result = data
        return result.reshape(len(data),self.shape[0],self.shape[1])

    #----------------------------------------------------------------------
    def mask(self, data):
        """Apply a mask to set to zero all pixels that were unactivated in the input."""
        result = np.zeros(data.shape)
        result[:, self.zero_mask] = data[:, self.zero_mask]
        return result

    #----------------------------------------------------------------------
    def _method_fit(self, data):
        pass

    #----------------------------------------------------------------------
    def _method_transform(self, data):
        return data

    #----------------------------------------------------------------------
    def _method_inverse(self, data):
        return data


#======================================================================
class PreprocessorPCA(Preprocessor):
    """Preprocessing pipeline using PCA."""
    
    #----------------------------------------------------------------------
    def __init__(self, ncomp, whiten, scaler=True, flatten=True, remove_zero=True):
        Preprocessor.__init__(self, scaler=scaler, flatten=flatten,
                              remove_zero=remove_zero)
        self.pca = PCA(ncomp, whiten=whiten)

    #----------------------------------------------------------------------
    def _method_fit(self, data):
        self.pca.fit(data)

    #----------------------------------------------------------------------
    def _method_transform(self, data):
        return self.pca.transform(data)

    #----------------------------------------------------------------------
    def _method_inverse(self, data):
        return self.pca.inverse_transform(data)


#======================================================================
class PreprocessorZCA(Preprocessor):
    """Preprocessing pipeline using ZCA."""
    
    #----------------------------------------------------------------------
    def __init__(self, scaler=True, flatten=True, remove_zero=True):
        Preprocessor.__init__(self, scaler=scaler, flatten=flatten,
                              remove_zero=remove_zero)

    #----------------------------------------------------------------------
    def _method_fit(self, data):
        pass

    #----------------------------------------------------------------------
    def _method_transform(self, data):
        result, self.zca = zca_whiten(data)
        return result

    #----------------------------------------------------------------------
    def _method_inverse(self, data):
        return np.dot(data, self.zca)


#======================================================================
class PreprocessorAE(Preprocessor):
    """Preprocessing pipeline using autoencoder."""
    
    #----------------------------------------------------------------------
    def __init__(self, dim, epochs, scaler=True, flatten=True, remove_zero=True):
        Preprocessor.__init__(self, scaler=scaler, flatten=flatten,
                              remove_zero=remove_zero)
        self.epochs = epochs
        self.dim = dim

    #----------------------------------------------------------------------
    def _method_fit(self, data):
        self.ae = Autoencoder(length=np.sum(data.shape[1:]), dim=self.dim)
        self.ae.train(data, self.epochs)

    #----------------------------------------------------------------------
    def _method_transform(self, data):
        return self.ae.encode(data)

    #----------------------------------------------------------------------
    def _method_inverse(self, data):
        return self.ae.decode(data)
