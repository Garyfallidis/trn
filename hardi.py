import numpy as np
from scipy.io import loadmat
from dipy.reconst.dsi import DiffusionSpectrum
from visualize_dsi import show_blobs

fname='/home/eg309/Software/Hardi/Training_3D_SF__SNR=40__SIGNAL.mat'
fgrads='/home/eg309/Software/Hardi/gradients_515.txt'

DATA=loadmat(fname)
data=np.ascontiguousarray(DATA['E'])

grads=np.loadtxt(fgrads)
bvecs=grads[:,:3]
bvals=grads[:,3]

ds=DiffusionSpectrum(data,bvals,bvecs,odf_sphere='symmetric642',save_odfs=True)

show_blobs(ds.ODF[:,:,0,:][:,:,None,:],ds.odf_vertices,ds.odf_faces)
