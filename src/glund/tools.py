# This file is part of gLund by S. Carrazza and F. A. Dreyer

import numpy as np
from scipy import linalg
from skimage.measure import compare_ssim as ssim
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.utils.validation import check_is_fitted
from sklearn.utils import check_array, as_float_array

#----------------------------------------------------------------------
def loss_calc(imgs_gen, imgs_ref):
    """
    Calculate a loss function by comparing a set of generated 
    images with a reference sample.
    The loss contains three components:
     - img_loss:     the norm of the difference between the average
                     of the images of the two samples
     - act_avg_loss: the difference between the average number of
                     activated pixels in both samples
     - act_var_loss: the difference between variance of the average number
                     of activated pixels in both samples
    so that L = Δ(<activated pixels>)/50 + σ²(<activated pixels>)/200 + norm(Δ<image>)
    where a normalisation of 1/50 and 1/200 is added to the act_avg_loss 
    and act_var_loss respectively
    """
    if (len(imgs_gen)!=len(imgs_ref)):
        raise ValueError('loss_calc: generated and reference arrays need to be of same length.')
    # get average of sum of pixels
    gen_sum_all = np.zeros(len(imgs_gen))
    ref_sum_all = np.zeros(len(imgs_ref))
    for i in range(len(imgs_gen)):
        gen_sum_all[i]=np.sum(imgs_gen[i])
        ref_sum_all[i]=np.sum(imgs_ref[i])
    
    # now calculate the full loss
    act_avg_loss = abs(np.average(gen_sum_all) - np.average(ref_sum_all)) /  50.0
    act_var_loss = abs(np.var(gen_sum_all)     -     np.var(ref_sum_all)) / 200.0
    img_loss = np.linalg.norm(np.average(imgs_gen,axis=0)-np.average(imgs_ref,axis=0))
    loss  = act_avg_loss + act_var_loss + img_loss
    print('Total loss: %f' % loss)
    print('(act_avg: %f\tact_var: %f\timg_norm: %f)' 
          % (act_avg_loss, act_var_loss, img_loss))
    
    return loss

#----------------------------------------------------------------------
def loss_calc_raw(imgs_gen, imgs_ref):
    """ 
    Calculate a loss on the raw images. 
    The loss function contains two components:
     - img_loss:  the norm of the difference between the average of the 
                  images of the two samples
     - ssim_loss: difference in ssim values between 1000 random pairs of
                  reference samples, and reference+generated samples
    """
    img_loss  = np.linalg.norm(np.nan_to_num((np.average(imgs_gen,axis=0)
                                              -np.average(imgs_ref,axis=0))
                                             /np.average(imgs_ref,axis=0)))
    nv=5000
    ssim_in  = np.zeros(nv)
    ssim_out = np.zeros(nv)
    for i in range(nv):
        p=np.random.choice(min(len(imgs_gen),len(imgs_ref)),(2,1),replace=False)[:,0]
        ssim_in[i]  = ssim(imgs_ref[p[0]], imgs_ref[p[1]])
        ssim_out[i] = ssim(imgs_ref[p[0]], imgs_gen[p[1]])
    ssim_loss = abs(np.average(ssim_in) - np.average(ssim_out))
    
    gen_sum_all = np.zeros(len(imgs_gen))
    ref_sum_all = np.zeros(len(imgs_ref))
    for i in range(len(imgs_gen)):
        gen_sum_all[i]=np.sum(imgs_gen[i])
        ref_sum_all[i]=np.sum(imgs_ref[i])
    act_avg_loss = abs(np.average(gen_sum_all) - np.average(ref_sum_all)) /  50.0
    act_var_loss = abs(np.var(gen_sum_all)     -     np.var(ref_sum_all)) / 200.0
    loss=img_loss + ssim_loss + act_avg_loss + act_var_loss
    print('Raw loss: %f\n(img_norm: %f, ssim_loss: %f, act_avg: %f, act_var: %f)'
          % (loss,img_loss,ssim_loss,act_avg_loss,act_var_loss))
    return loss

#----------------------------------------------------------------------
def zca_whiten(X):
    """
    Applies ZCA whitening to the data (X)
    http://xcorr.net/2011/05/27/whiten-a-matrix-matlab-code/

    X: numpy 2d array
        input data, rows are data points, columns are features

    Returns: ZCA whitened 2d array
    """
    assert(X.ndim == 2)
    EPS = 10e-5

    #   covariance matrix
    cov = np.dot(X.T, X)
    #   d = (lambda1, lambda2, ..., lambdaN)
    d, E = np.linalg.eigh(cov)
    #   D = diag(d) ^ (-1/2)
    D = np.diag(1. / np.sqrt(d + EPS))
    #   W_zca = E * D * E.T
    W = np.dot(np.dot(E, D), E.T)

    X_white = np.dot(X, W)

    return X_white, np.linalg.inv(W)


#======================================================================
# adapted from: https://github.com/mwv/zca/blob/master/zca/zca.py
class ZCA(BaseEstimator, TransformerMixin):
    def __init__(self, regularization=1e-6, copy=False):
        self.regularization = regularization
        self.copy = copy

    def fit(self, X, y=None):
        """Compute the mean, whitening and dewhitening matrices.
        Parameters
        ----------
        X : array-like with shape [n_samples, n_features]
            The data used to compute the mean, whitening and dewhitening
            matrices.
        """
        X = check_array(X, accept_sparse=False, copy=self.copy,
                        ensure_2d=True)
        X = as_float_array(X, copy=self.copy)
        self.mean_ = X.mean(axis=0)
        X_ = X - self.mean_
        cov = np.dot(X_.T, X_) / (X_.shape[0]-1)
        U, S, _ = linalg.svd(cov)
        s = np.sqrt(S.clip(self.regularization))
        s_inv = np.diag(1./s)
        s = np.diag(s)
        self.whiten_ = np.dot(np.dot(U, s_inv), U.T)
        self.dewhiten_ = np.dot(np.dot(U, s), U.T)
        return self

    def transform(self, X, y=None, copy=None):
        """Perform ZCA whitening
        Parameters
        ----------
        X : array-like with shape [n_samples, n_features]
            The data to whiten along the features axis.
        """
        check_is_fitted(self, 'mean_')
        X = as_float_array(X, copy=self.copy)
        return np.dot(X - self.mean_, self.whiten_.T)

    def inverse_transform(self, X, copy=None):
        """Undo the ZCA transform and rotate back to the original
        representation
        Parameters
        ----------
        X : array-like with shape [n_samples, n_features]
            The data to rotate back.
        """
        check_is_fitted(self, 'mean_')
        X = as_float_array(X, copy=self.copy)
        return np.dot(X, self.dewhiten_) + self.mean_
