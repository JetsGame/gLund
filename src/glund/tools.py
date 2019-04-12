# This file is part of gLund by S. Carrazza and F. A. Dreyer

import numpy as np
from skimage.measure import compare_ssim as ssim

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
    img_loss  = np.linalg.norm(np.average(imgs_gen,axis=0)-np.average(imgs_ref,axis=0))
    nv=1000
    ssim_in  = np.zeros(1000)
    ssim_out = np.zeros(1000)
    for i in range(nv):
        p=np.random.choice(min(len(imgs_gen),len(imgs_ref)),(2,1),replace=False)[:,0]
        ssim_in[i]  = ssim(imgs_ref[p[0]], imgs_ref[p[1]])
        ssim_out[i] = ssim(imgs_ref[p[0]], imgs_gen[p[1]])
    ssim_loss = 5* (np.average(ssim_in) - np.average(ssim_out))
    print('Raw loss: %f\t(%f, %f)' % (img_loss+ssim_loss,img_loss,ssim_loss))
    return img_loss + ssim_loss

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
