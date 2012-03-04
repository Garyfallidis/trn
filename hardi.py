#! /usr/bin/env python

import numpy as np
from scipy.io import loadmat
from dipy.reconst.dsi import DiffusionSpectrum
from dipy.reconst.gqi import GeneralizedQSampling
from dipy.reconst.dni import EquatorialInversion
from dipy.reconst.recspeed import peak_finding
from visualize_dsi import show_blobs

fname='/home/eg309/Software/Hardi/Training_3D_SF__SNR=40__SIGNAL.mat'
fgrads='/home/eg309/Software/Hardi/gradients_257.txt'

fvertices='/home/eg309/Software/Hardi/TrainingData/ODF_XYZ.mat'
vertices=loadmat(fvertices)
vertices=np.ascontiguousarray(vertices['ODF_XYZ'])

ffaces='/home/eg309/Software/Hardi/TrainingData/FACES.mat'
faces=loadmat(ffaces)
faces=np.ascontiguousarray(faces['K'])
faces=faces-1 #from matlab to numpy indexing

dummy_odf=np.zeros(len(vertices))
dummy_odf[10]=1
print peak_finding(dummy_odf,faces.astype(np.uint16))

DATA=loadmat(fname)
data=np.ascontiguousarray(DATA['E'])

grads=np.loadtxt(fgrads)
bvecs=grads[:,:3]
bvals=grads[:,3]

"""
z_inds=np.argsort(bvecs[:,2])
grads2=grads[z_inds]
#bvecs=bvecs[z_inds]
#bvals=bvals[z_inds]

fgrads2='/home/eg309/Software/Hardi/gradients_515_2.txt'
np.savetxt(fgrads2,grads2)

fgrads3='/home/eg309/Software/Hardi/gradients_257.txt'
np.savetxt(fgrads3,grads2[257:])
"""

odf_sphere=(vertices.astype(np.float32),faces.astype(np.uint16))

bvals=np.append(bvals.copy(),bvals[1:].copy())
bvecs=np.append(bvecs.copy(),-bvecs[1:].copy(),axis=0)
data=np.append(data.copy(),data[...,1:].copy(),axis=-1)

#ds2=DiffusionSpectrum(data,bvals,bvecs,odf_sphere='symmetric642',save_odfs=True)
ds=DiffusionSpectrum(data,bvals,bvecs,odf_sphere=odf_sphere,half_sphere_grads=False,save_odfs=True)

show_blobs(ds.ODF[:,:,0,:][:,:,None,:],ds.odf_vertices,ds.odf_faces)



"""
from dipy.viz import fvtk
res=bvecs*bvals.reshape(len(bvecs),1)
r=fvtk.ren()
fvtk.add(r,fvtk.point(res[:257],fvtk.red,point_radius=100))
fvtk.show(r)
"""

