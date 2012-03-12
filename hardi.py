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
from scipy.optimize import fmin as fmin_powell
from scipy.optimize import leastsq
from time import time

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

def ODF(vecs,mf,mevals,mevecs):
    odf=np.zeros(len(vecs))
    m=len(mf)
    for (i,v) in enumerate(vecs):
        for (j,f) in enumerate(mf):
            evals=mevals[j]
            evecs=mevecs[j]
            D=np.dot(np.dot(evecs,np.diag(evals)),evecs.T)
            iD=np.linalg.inv(D)
            nD=np.linalg.det(D)
            upper=(np.dot(np.dot(v.T,iD),v))**(-3/2.)
            lower=4*np.pi*np.sqrt(nD)
            odf[i]+=f*upper/lower
    return odf

def lambda_ranges():
    print 'max', 1*10**(-3),'to',2*10**(-3)
    print 'other', 0.1*10**(-3),'to',0.6*10**(-3)
    lmin=np.linspace(0.1,0.6,20)*10**(-3)
    lmax=np.linspace(1,2,20)*10**(-3)
    return lmax,lmin 

def count_peaks(PK):
    return np.sum(PK>0,axis=-1)

def all_evecs(e0):
    axes=np.array([[1.,0,0],[0,1.,0],[0,0,1.]])
    mat=vec2vec_rotmat(axes[2],e0)
    e1=np.dot(mat,axes[0])
    e2=np.dot(mat,axes[1])
    return np.array([e0,e1,e2])

def opt2(params,bvals,bvecs,signal,mevecs):
    mf=[params[0],1-params[0]]
    mevals=np.zeros((2,3))
    mevals[0,0]=params[1]
    mevals[0,1:]=params[2]
    mevals[1,0]=params[3]
    mevals[1,1:]=params[4]
    mevals=mevals*10**(-3)
    S=MultiTensor(bvals,bvecs,1.,mf,mevals,mevecs)
    return np.sum(np.sqrt((S-signal)**2))

def opt2lsq(params,bvals,bvecs,signal,mevecs):
    mf=[params[0],1-params[0]]
    mevals=np.zeros((2,3))
    mevals[0,0]=params[1]
    mevals[0,1:]=params[2]
    mevals[1,0]=params[3]
    mevals[1,1:]=params[4]
    mevals=mevals*10**(-3)
    S=MultiTensor(bvals,bvecs,1.,mf,mevals,mevecs)
    #return np.sum(np.sqrt((S-signal)**2))
    return S-signal

def opt3(params,bvals,bvecs,signal,mevecs):
    mf=[params[0],params[1],1-params[0]-params[1]]
    mevals=np.zeros((3,3))
    mevals[0,0]=params[2]
    mevals[0,1:]=params[3]
    mevals[1,0]=params[4]
    mevals[1,1:]=params[5]
    mevals[2,0]=params[6]
    mevals[2,1:]=params[7]
    mevals=mevals*10**(-3)
    S=MultiTensor(bvals,bvecs,1.,mf,mevals,mevecs)
    return np.sum(np.sqrt((S-signal)**2))

def unpackopt2(xopt):
    params=xopt
    mf=[params[0],1-params[0]]
    mevals=np.zeros((2,3))
    mevals[0,0]=params[1]
    mevals[0,1:]=params[2]
    mevals[1,0]=params[3]
    mevals[1,1:]=params[4]
    mevals=mevals*10**(-3)
    return mf, mevals

def unpackopt3(xopt):
    params=xopt
    mf=[params[0],params[1],1-params[0]-params[1]]
    mevals=np.zeros((3,3))
    mevals[0,0]=params[2]
    mevals[0,1:]=params[3]
    mevals[1,0]=params[4]
    mevals[1,1:]=params[5]
    mevals[2,0]=params[6]
    mevals[2,1:]=params[7]
    mevals=mevals*10**(-3)
    return mf,mevals



def load_data(test,type,snr):

    if test=='train':
        fname='/home/eg309/Software/Hardi/Training_'+type+'__SNR='+snr+'__SIGNAL.mat'
    if test=='test':
        fname='/home/eg309/Software/Hardi/TestData/Testing_'+type+'__SNR='+snr+'__SIGNAL.mat'

    fgrads='/home/eg309/Software/Hardi/gradient_list_257_clean.txt'
    fvertices='/home/eg309/Software/Hardi/TrainingData/ODF_XYZ.mat'
    vertices=loadmat(fvertices)
    vertices=np.ascontiguousarray(vertices['ODF_XYZ'])

    ffaces='/home/eg309/Software/Hardi/TrainingData/FACES.mat'
    faces=loadmat(ffaces)
    faces=np.ascontiguousarray(faces['K'])
    faces=faces-1 #from matlab to numpy indexing

    DATA=loadmat(fname)
    dat=np.ascontiguousarray(DATA['E'])

    grads=np.loadtxt(fgrads)
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

    return data,bvals,bvecs,odf_sphere

def dump():

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
    pass


def analyze_peaks(data,ten,qg):

    PK=qg.PK
    IN=qg.IN

    M=count_peaks(PK)

    R={}

    for index in np.ndindex(M.shape):
        print index, M[index]
        if M[index]==1:
            mf=[1.]
            mevals=[ten[index].evals]
            mevecs=[ten[index].evecs]
        if M[index]==2:
            e0=qg.odf_vertices[np.int(qg.IN[index+(0,)])]
            e1=qg.odf_vertices[np.int(qg.IN[index+(1,)])]
            signal = data[index]
            mevecs=[all_evecs(e0),all_evecs(e1)]
            xopt=fmin_powell(opt2,\
            [0.5,0.5,0.5,0.5,0.5],\
            (bvals,bvecs,signal,mevecs),\
            xtol=10**(-6),\
            ftol=10**(-6),\
            maxiter=10**6,\
            disp=False)
            mf,mevals=unpackopt2(xopt)
        if M[index]==3:
            e0=qg.odf_vertices[np.int(qg.IN[index+(0,)])]
            e1=qg.odf_vertices[np.int(qg.IN[index+(1,)])]
            e2=qg.odf_vertices[np.int(qg.IN[index+(2,)])]
            signal = data[index]
            mevecs=[all_evecs(e0),all_evecs(e1),all_evecs(e2)]
            xopt=fmin_powell(opt2,\
            [0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5],\
            (bvals,bvecs,signal,mevecs),\
            xtol=10**(-6),\
            ftol=10**(-6),\
            maxiter=10**6,\
            disp=False)
            mf,mevals=unpackopt3(xopt)        
        odf=ODF(qg.odf_vertices,mf,mevals,mevecs)
        R[index]={'m':M[index],'f':mf,'evals':mevals,'evecs':mevecs,'odf':odf}

    return M,R
    
def show_no_fibs(M,R):

    for index in np.ndindex(M.shape):
        print index
        print R[index]['m']



if __name__ == '__main__':

    data,bvals,bvecs,odf_sphere=load_data('train','SF','30')

    #data=data[:,4:10,:,:]

    #ten
    ten = Tensor(100*data, bvals, bvecs)
    FA = ten.fa()
    famask=FA>=.2

    #GQI
    gqs=GeneralizedQSampling(data,bvals,bvecs,4.5,
                    odf_sphere=odf_sphere,
                    mask=None,
                    squared=True,
                    save_odfs=True)

    qg=gqs

    #M,R=analyze_peaks(data,ten,qg)
    
    show_blobs(qg.ODF[:,:,0,:][:,:,None,:],qg.odf_vertices,qg.odf_faces,size=1.5,scale=1.)
