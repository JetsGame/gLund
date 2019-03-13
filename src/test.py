from read_data import Reader, Jets 
from JetTree import *
import matplotlib.pyplot as plt
from gan import GAN

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

    return X_white

nev = 20000
npxlx = 10
npxly = 10

reader=Jets('../../data/valid/valid_WW_500GeV.json.gz',nev)
events=reader.values() 
lundimages=np.zeros((nev, npxlx, npxly, 1))
litest=[] 
li_gen=LundImage(npxlx = npxlx, npxly = npxly) 
for i, jet in enumerate(events): 
    tree = JetTree(jet) 
    lundimages[i]=li_gen(tree).reshape(npxlx, npxly, 1)

print(lundimages.shape)

# normalisation of images
lundimages = (lundimages - np.average(lundimages, axis=0))

# plt.imshow(np.average(lundimages, axis=0).transpose(),
#            origin='lower', aspect='auto')
# plt.show()

gan = GAN(width=npxlx, height=npxly, channels=1)
gan.train(lundimages)
