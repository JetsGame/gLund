# This file is part of gLund by S. Carrazza and F. A. Dreyer

from glund.models.autoencoder import Autoencoder
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from glund.tools import ZCA
import numpy as np
import pickle

#----------------------------------------------------------------------
def build_preprocessor(setup):
    flat_input = setup['model'] in ('gan', 'vae', 'bgan', 'aae', 'lsgan')
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

#----------------------------------------------------------------------
def load_preprocessor(folder, setup):
    preprocess = build_preprocessor(setup)
    preprocess.load(folder)
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
        self.length = None

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
        self.length = data.shape[1]
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
    def save(self, folder):
        preprocessor = {'shape': self.shape,
                        'length': self.length,
                        'flatten': self.flatten,
                        'remove_zero': self.remove_zero,
                        'has_scaler': True if self.scaler else False,
                        'zero_mask': self.zero_mask}
        if self.scaler:
            preprocessor['scaler'] = {'scale': self.scaler.scale_,
                                      'mean': self.scaler.mean_,
                                      'var': self.scaler.var_,
                                      'n': self.scaler.n_samples_seen_}
        self._method_save(preprocessor)
        with open(f'{folder}/preprocessor.pkl','wb') as f:
            pickle.dump(preprocessor, f)

    def load(self, folder):
        with open(f'{folder}/preprocessor.pkl','rb') as f:
            preprocessor = pickle.load(f)
        self.shape       = preprocessor['shape']
        self.length      = preprocessor['length']
        self.flatten     = preprocessor['flatten']
        self.remove_zero = preprocessor['remove_zero']
        self.zero_mask   = preprocessor['zero_mask']
        self.scaler = StandardScaler() if preprocessor['has_scaler'] else None
        if self.scaler:
            self.scaler.scale_ = preprocessor['scaler']['scale']
            self.scaler.mean_  = preprocessor['scaler']['mean']
            self.scaler.var_   = preprocessor['scaler']['var']
            self.scaler.n_samples_seen_ = preprocessor['scaler']['n']
        self._method_load(preprocessor)

    #----------------------------------------------------------------------
    def _method_fit(self, data):
        pass

    #----------------------------------------------------------------------
    def _method_transform(self, data):
        return data

    #----------------------------------------------------------------------
    def _method_inverse(self, data):
        return data

    #----------------------------------------------------------------------
    def _method_save(self, preprocessor):
        pass

    #----------------------------------------------------------------------
    def _method_load(self, preprocessor):
        pass

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
        self.length = self.pca.n_components_

    #----------------------------------------------------------------------
    def _method_transform(self, data):
        return self.pca.transform(data)

    #----------------------------------------------------------------------
    def _method_inverse(self, data):
        return self.pca.inverse_transform(data)

    #----------------------------------------------------------------------
    def _method_save(self, preprocessor):
        preprocessor['pca'] = \
            {'n_components': self.pca.n_components,
             'n_features': self.pca.n_features_,
             'n_samples': self.pca.n_samples_,
             'n_components_': self.pca.n_components_,
             'components_': self.pca.components_,
             'whiten': self.pca.whiten,
             'components': self.pca.components_,
             'explained_var': self.pca.explained_variance_,
             'explained_var_ratio': self.pca.explained_variance_ratio_,
             'singular_values': self.pca.singular_values_,
             'mean': self.pca.mean_,
             'ncomp': self.pca.n_components_,
             'noise_var': self.pca.noise_variance_}

    #----------------------------------------------------------------------
    def _method_load(self, preprocessor):
        self.pca = PCA()
        self.pca.n_components = preprocessor['pca']['n_components']
        self.pca.n_features_ = preprocessor['pca']['n_features']
        self.pca.n_samples_ = preprocessor['pca']['n_samples']
        self.pca.n_components_ = preprocessor['pca']['n_components_']
        self.pca.components_ = preprocessor['pca']['components_']
        self.pca.whiten = preprocessor['pca']['whiten']
        self.pca.components_ = preprocessor['pca']['components']
        self.pca.explained_variance_ = preprocessor['pca']['explained_var']
        self.pca.explained_variance_ratio_ = preprocessor['pca']['explained_var_ratio']
        self.pca.singular_values_ = preprocessor['pca']['singular_values']
        self.pca.mean_ = preprocessor['pca']['mean']
        self.pca.n_components_ = preprocessor['pca']['ncomp']
        self.pca.noise_variance_ = preprocessor['pca']['noise_var']
        self.length = self.pca.n_components_
        

#======================================================================
class PreprocessorZCA(Preprocessor):
    """Preprocessing pipeline using ZCA."""
    
    #----------------------------------------------------------------------
    def __init__(self, scaler=True, flatten=True, remove_zero=True):
        Preprocessor.__init__(self, scaler=scaler, flatten=flatten,
                              remove_zero=remove_zero)

    #----------------------------------------------------------------------
    def _method_fit(self, data):
        self.zca = ZCA().fit(data)

    #----------------------------------------------------------------------
    def _method_transform(self, data):
        return self.zca.transform(data)

    #----------------------------------------------------------------------
    def _method_inverse(self, data):
        return self.zca.inverse_transform(data)

    #----------------------------------------------------------------------
    def _method_save(self, preprocessor):
        preprocessor['zca'] = \
            {'regularization': self.zca.regularization,
             'copy': self.zca.copy,
             'mean': self.zca.mean_,
             'whiten': self.zca.whiten_,
             'dewhiten': self.zca.dewhiten_}

    #----------------------------------------------------------------------
    def _method_load(self, preprocessor):
        self.zca = ZCA()
        self.zca.regularization = preprocessor['zca']['regularization']
        self.zca.copy = preprocessor['zca']['copy']
        self.zca.mean_ = preprocessor['zca']['mean']
        self.zca.whiten_ = preprocessor['zca']['whiten']
        self.zca.dewhiten_ = preprocessor['zca']['dewhiten']
        

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

    #----------------------------------------------------------------------
    def _method_save(self, preprocessor):
        raise Exception('PreprocessorAE can not be saved to file.')

    #----------------------------------------------------------------------
    def _method_load(self, preprocessor):
        raise Exception('PreprocessorAE can not be loaded from file.')
