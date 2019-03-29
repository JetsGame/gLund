import numpy as np

def loss_calc(imgs_gen, imgs_ref, epsilon):
    """
    Calculate a loss function by comparing a set of generated 
    images with a reference sample.
    The loss contains three components:
     - delta_img: the difference between the average of the images
     - delta_act: the difference between the average number of
                  activated pixels, ie pixels with value > epsilon
     - var_act:   the difference between variance of the average number
                  of activated pixels, ie pixels with value > epsilon
     - delta_sum: the difference between the average of the sum
                  over all pixel values in the image
     - var_sum:   the difference between the variance of the average of 
                  the sum over all pixel values in the image
    so that L = delta_img + delta_sum + var_sum + delta_act + var_act
    TODO: sort out normalisation between components
    """
    if (len(imgs_gen)!=len(imgs_ref)):
        raise ValueError('loss_calc: generated and reference arrays need to be of same length.')
    # get average of sum of pixels
    gen_sum_all = np.zeros(len(imgs_gen))
    ref_sum_all = np.zeros(len(imgs_ref))
    gen_act_all = np.zeros(len(imgs_gen))
    ref_act_all = np.zeros(len(imgs_ref))
    for i in range(len(imgs_gen)):
        gen_sum_all[i]=np.sum(imgs_gen[i])
        ref_sum_all[i]=np.sum(imgs_ref[i])
        gen_act_all[i]=len(imgs_gen[i][np.where(imgs_gen[i] > epsilon)])
        ref_act_all[i]=len(imgs_ref[i][np.where(imgs_ref[i] > epsilon)])
    sum_pix = (np.average(gen_sum_all), np.average(ref_sum_all))
    sum_pix_var = (np.var(gen_sum_all), np.var(ref_sum_all))
    act_pix = (np.average(gen_act_all), np.average(ref_act_all))
    act_pix_var = (np.var(gen_act_all), np.var(ref_act_all))
    # now calculate the full loss
    loss  = abs(sum_pix[0]-sum_pix[1]) + abs(sum_pix_var[0]-sum_pix_var[1])
    loss += abs(act_pix[0]-act_pix[1]) + abs(act_pix_var[0]-act_pix_var[1])
    # finally add the distance metric between the images
    loss += np.linalg.norm(np.average(imgs_gen,axis=0)-np.average(imgs_ref,axis=0))
    # print('sum_avg: %f\tsum_var: %f\tact_avg:%f\tact_var%f' %\
    #       (abs(sum_pix[0]-sum_pix[1]),abs(sum_pix_var[0]-sum_pix_var[1]),
    #        abs(act_pix[0]-act_pix[1]),abs(act_pix_var[0]-act_pix_var[1])))
    # print('image loss: %f' % \
    #       np.linalg.norm(np.average(imgs_gen,axis=0)-np.average(imgs_ref,axis=0)))
    return loss

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
