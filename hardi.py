#! /usr/bin/env python

import numpy as np
from scipy.io import loadmat
from dipy.reconst.dti import Tensor
from dipy.reconst.dni import EquatorialInversion
from dipy.reconst.gqi import GeneralizedQSampling
from dipy.reconst.dsi import DiffusionSpectrum
from dipy.reconst.recspeed import peak_finding
from visualize_dsi import show_blobs
from dipy.core.geometry import sphere2cart, cart2sphere
from dipy.core.geometry import vec2vec_rotmat

def SticksAndBall(bvals,gradients,d=0.0015,S0=100,angles=[(0,0),(90,0)],fractions=[35,35],snr=20):
    """ Simulating the signal for a Sticks & Ball model 
    
    Based on the paper by Tim Behrens, H.J. Berg, S. Jbabdi, "Probabilistic Diffusion Tractography with multiple fiber orientations
    what can we gain?", Neuroimage, 2007. 
    
    Parameters
    -----------
    bvals : array, shape (N,)
    gradients : array, shape (N,3) also known as bvecs
    d : diffusivity value 
    S0 : unweighted signal value
    angles : array (K,2) list of polar angles (in degrees) for the sticks
        or array (K,3) with sticks as Cartesian unit vectors and K the number of sticks
    fractions : percentage of each stick
    snr : signal to noise ration assuming gaussian noise. Provide None for no noise.
    
    Returns
    --------
    S : simulated signal
    sticks : sticks in cartesian coordinates 
    
    """
    
    fractions=[f/100. for f in fractions]    
    f0=1-np.sum(fractions)    
    S=np.zeros(len(gradients))
    
    angles=np.array(angles)
    if angles.shape[-1]==3:
        sticks=angles
    if angles.shape[-1]==2:
        sticks=[ sphere2cart(1,np.deg2rad(pair[0]),np.deg2rad(pair[1]))  for pair in angles]    
        sticks=np.array(sticks)
    
    for (i,g) in enumerate(gradients[1:]):
        S[i+1]=f0*np.exp(-bvals[i+1]*d)+ np.sum([fractions[j]*np.exp(-bvals[i+1]*d*np.dot(s,g)**2) for (j,s) in enumerate(sticks)])
        S[i+1]=S0*S[i+1]    
    S[0]=S0    
    if snr!=None:
        std=S0/snr
        S=S+np.random.randn(len(S))*std
    
    return S,sticks



def SingleTensor(bvals,gradients,S0,evals,evecs,snr=None):
    """ Simulated signal with a Single Tensor
     
    Parameters
    ----------- 
    bvals : array, shape (N,)
    gradients : array, shape (N,3) also known as bvecs
    S0 : double,
    evals : array, shape (3,) eigen values
    evecs : array, shape (3,3) eigen vectors
    snr : signal to noise ratio assuming gaussian noise. 
        Provide None for no noise.
    
    Returns
    --------
    S : simulated signal    
    
    """
    S=np.zeros(len(gradients))
    D=np.dot(np.dot(evecs,np.diag(evals)),evecs.T)    
    #print D.shape
    for (i,g) in enumerate(gradients[1:]):
        S[i+1]=S0*np.exp(-bvals[i+1]*np.dot(np.dot(g.T,D),g))
    S[0]=S0
    if snr!=None:
        std=S0/snr
        S=S+np.random.randn(len(S))*std
    return S
    
def MultiTensor(bvals,gradients,S0,mf,mevals,mevecs):
    S=np.zeros(len(gradients))
    m=len(mf)    
    #print D.shape
    for (i,g) in enumerate(gradients[1:]):
        for (j,f) in enumerate(mf):
            evals=mevals[j]
            evecs=mevecs[j]
            D=np.dot(np.dot(evecs,np.diag(evals)),evecs.T)
            S[i+1]+=S0*f*np.exp(-bvals[i+1]*np.dot(np.dot(g.T,D),g))
    S[0]=S0
    return S

def lambda_ranges():
    print 'max', 1*10**(-3),'to',2*10**(-3)
    print 'other', 0.1*10**(-3),'to',0.6*10**(-3)
    lmin=np.linspace(0.1,0.6,20)*10**(-3)
    lmax=np.linspace(1,2,20)*10**(3)
    return lmax,lmin 

#fname='/home/eg309/Software/Hardi/Training_3D_SF__SNR=30__SIGNAL.mat'
fname='/home/eg309/Software/Hardi/TestData/'+\
        'Testing_'+'IV'+'__SNR='+'10'+'__SIGNAL.mat'

fgrads='/home/eg309/Software/Hardi/gradient_list_257_clean.txt'

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
dat=np.ascontiguousarray(DATA['E'])

grads=np.loadtxt(fgrads)
#bvecs=grads[:,:3]
#bvals=grads[:,3]

odf_sphere=(vertices.astype(np.float32),faces.astype(np.uint16))

bvals=np.zeros(515)
bvals[0]=0
bvals[1:258]=grads[:,3]
bvals[258:]=grads[:,3]
bvecs=np.zeros((515,3))
bvecs[0,:]=np.zeros(3)
bvecs[1:258,:]=grads[:,:3]
bvecs[258:,:]=-grads[:,:3]

data=np.zeros(dat.shape[:3]+(515,))
data[:,:,:,0]=1
data[:,:,:,1:258]=dat.copy()
data[:,:,:,258:]=dat.copy()

#bvals=np.append(bvals.copy(),bvals.copy())
#bvecs=np.append(bvecs.copy(),-bvecs.copy(),axis=0)
#data=np.append(data.copy(),data.copy(),axis=-1)

#stop

data=data[:,4:10,:,:]


#ten
ten = Tensor(100*data, bvals, bvecs)
FA = ten.fa()
famask=FA>=.2


#stop

#GQI
gqs=GeneralizedQSampling(data,bvals,bvecs,3.,
                odf_sphere=odf_sphere,
                mask=None,
                squared=True,
                save_odfs=True)
"""
#EIT
ei=EquatorialInversion(data,bvals,bvecs,
            odf_sphere=odf_sphere,
            mask=None,
            half_sphere_grads=False,
            auto=False,
            save_odfs=True,
            fast=True)

ei.radius=np.arange(0,5,0.2)
ei.gaussian_weight=0.02
ei.set_operator('laplap')#laplacian
ei.update()
ei.fit() 

#DSI
ds=DiffusionSpectrum(data,bvals,bvecs,            
            odf_sphere=odf_sphere,
            mask=None,
            half_sphere_grads=False,
            save_odfs=True)
"""

qg=gqs

show_blobs(qg.ODF[:,:,0,:][:,:,None,:],qg.odf_vertices,qg.odf_faces,size=1.5,scale=1.)

PK=qg.PK
IN=qg.IN

def count_peaks(PK):
    return np.sum(PK>0,axis=-1)

M=count_peaks(PK)

#def get_orientations(IN,verts):    
#get_orientations(gq.IN,gq.odf_vertices)

e0=qg.odf_vertices[np.int(qg.IN[0,4,0,0])]
e1=qg.odf_vertices[np.int(qg.IN[0,4,0,1])]

mf=[0.5,0.5]

mevals=np.array([[1.5,0.2,0.2],[1.5,0.2,0.2]])*10**(-3)

def all_evecs(e0):
    axes=np.array([[1.,0,0],[0,1.,0],[0,0,1.]])
    mat=vec2vec_rotmat(axes[2],e0)
    e1=np.dot(mat,axes[0])
    e2=np.dot(mat,axes[1])
    return np.array([e0,e1,e2])

mevecs=[all_evecs(e0),all_evecs(e1)]

S=MultiTensor(bvals,bvecs,1.,mf,mevals,mevecs)



"""
from dipy.viz import fvtk
res=bvecs*bvals.reshape(len(bvecs),1)
r=fvtk.ren()
fvtk.add(r,fvtk.point(res[:257],fvtk.red,point_radius=100))
fvtk.show(r)
"""

