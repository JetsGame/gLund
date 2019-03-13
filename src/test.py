from read_data import Reader, Jets 
from JetTree import *
import matplotlib.pyplot as plt
from gan import GAN

nev = 50000
npxlx = 28
npxly = 28

reader=Jets('../../data/valid/valid_WW_500GeV.json.gz',nev)
events=reader.values() 
lundimages=np.zeros((nev, npxlx, npxly, 1))
litest=[] 
li_gen=LundImage(npxlx = npxlx, npxly = npxly) 
for i, jet in enumerate(events): 
    tree = JetTree(jet) 
    lundimages[i]=li_gen(tree).reshape(npxlx, npxly, 1)

print(lundimages.shape)

# plt.imshow(np.average(lundimages, axis=0).transpose(),
#            origin='lower', aspect='auto')
# plt.show()

gan = GAN(width=npxlx, height=npxly, channels=1)
gan.train(lundimages)
