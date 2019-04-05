# This file is part of gLund by S. Carrazza and F. A. Dreyer

import numpy as np

#----------------------------------------------------------------------
def loss_calc(imgs_gen, imgs_ref, epsilon):
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
    so that L = act_avg_loss + act_var_loss + img_loss
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
    
    print('act_avg:%f\tact_var%f\timg_norm:%f' 
          % (act_avg_loss, act_var_loss, img_loss))
    
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
