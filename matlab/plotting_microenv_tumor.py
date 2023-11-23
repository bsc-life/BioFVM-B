# Code adapted from plot_microenvironment.py, originally build to plot results from
#    http://www.mathcancer.org/blog/biofvm-warmup-2d-continuum-simulation-of-tumor-growth/ 
import sys
import os.path
import scipy.io
import matplotlib.pyplot as plt
import numpy as np

""" ----  Sample use:

python plotting_microenv_tumor.py "filename.mat" z_cut
python plotting_microenv_tumor.py "final.mat" 2.025

----------"""

fname = sys.argv[1]
if (os.path.exists(fname) == False):
    print("File %s does not exist" % fname)
    sys.exit(0)
z = float(sys.argv[2])

info_dict = {}
scipy.io.loadmat(fname, info_dict)
M1 = info_dict['multiscale_microenvironment']
a2 = M1[:,:]
a3 = a2[:,a2[2,:] == z]
a7=a3.T
a6=a7[np.lexsort(( a7[:,1],a7[:,0]))]

fig, axs = plt.subplots(3, figsize=(6,9), dpi=300)

im00 = axs[0].imshow(a6[:,4].reshape(80,80), cmap='jet', extent=[0,4, 0,4])
fig.colorbar(im00, ax=axs[0])
axs[0].set_title("density_1")  
im10 = axs[1].imshow(a6[:,5].reshape(80,80), cmap='jet', extent=[0,4, 0,4])
fig.colorbar(im10, ax=axs[1])
axs[1].set_title("density_2")  
im20 = axs[2].imshow(a6[:,6].reshape(80,80), cmap='jet', extent=[0,4, 0,4])
fig.colorbar(im20, ax=axs[2])
axs[2].set_title("density_3")  

fig.savefig( fname.replace(".mat", "") + '.png')
