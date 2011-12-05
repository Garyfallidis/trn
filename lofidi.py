#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Author: Eleftherios Garyfallidis
Description: Local Fiber Direction Construction

'''

import form
import lights
import general
import scipy as sp
import external
import time
from scipy import linalg as la
import os
import Pycluster as pcl

#from enthought.mayavi.mlab import *

def primarydirections(voxsz,normalize=True):
    
    X=voxsz[0]
    Xmin=-X/2.0    
    Xmax=X/2.0
    
    Y=voxsz[1]
    Ymin=-Y/2.0    
    Ymax=Y/2.0
    
    Z=voxsz[2]
    Zmin=-Z/2.0    
    Zmax=Z/2.0
    
    D=sp.array([[0,Ymin,Zmax],
                                       [Xmax,Ymin,Zmax],
                                       [Xmax,Ymin,0],
                                        [Xmax,Ymin,Zmin],                                      
                                       [Xmax,0,Zmax],
                                       [Xmax,0,0],
                                        [Xmax,0,Zmin], 
                                        [0,Ymax,Zmax],
                                       [Xmax,Ymax,Zmax],
                                       [Xmax,Ymax,0],
                                        [Xmax,Ymax,Zmin],                                         
                                        [0,0,Zmax],
                                        [0,Ymax,0]])
                                           

    #print 'D',D
    
    if normalize:
        
        D1=sp.sqrt(D[:,0]**2+D[:,1]**2+D[:,2]**2)
  
        D=D.transpose()/D1

        D=D.transpose()
       
        #D2=sp.sqrt(D[:,0]**2+D[:,1]**2+D[:,2]**2)
    
    return D


def normalizeln(x,xp=sp.array([0,255]),fp=sp.array([0,1]),option='minmax'):
   
    #sp.interp(0,sp.array([0,255]),sp.array([-1,1]),left=-2,right=2)   

    try:
        sh=x.shape
        x=sp.interp(x.flatten(),xp,fp)
        x=x.reshape(sh)
    except:
        x=sp.interp(x,xp,fp)
        
    return x

def histeq(im,nbr_bins=256):

    '''
        Histogram Equalization
        Code and Information from 
        http://jesolem.blogspot.com/search/label/histogram
    '''
    #get image histogram
    imhist,bins = sp.histogram(im.flatten(),nbr_bins,normed=True,new=True)
    cdf = imhist.cumsum() #cumulative distribution function
    #cdf = 255 * cdf / cdf[-1] #normalize
    cdf=(nbr_bins-1)*cdf/cdf[-1]
    #use linear interpolation of cdf to find new pixel values
    #im2 = sp.interp(im.flatten(),bins[50:-1],cdf[50:])
    im2 = sp.interp(im.flatten(),bins[:-1],cdf[:])

    return im2.reshape(im.shape), cdf

def axialfilter(arr,no):

    sz=arr.shape
    ker=sp.zeros(sz)
    msk=sp.ones((sz[1],sz[2]))  
    ker[no]=msk
    arr=arr*ker
   
    return arr,ker

def coronalfilter(arr,no):

    sz=arr.shape
    ker=sp.zeros(sz)
    msk=sp.ones((sz[0],sz[2]))  
    ker[:,no]=msk
    arr=arr*ker
   
    return arr,ker

def saggitalfilter(arr,no):

    sz=arr.shape
    ker=sp.zeros(sz)
    msk=sp.ones((sz[0],sz[1]))  
    ker[:,:,no]=msk
    arr=arr*ker
   
    return arr,ker

def slicerfilter(arr,point=(0,0,0)):

    if arr.ndim != 3:
        print('arr needs to be 3d')    

    arr1,ker1=axialfilter(arr,point[0])
    arr2,ker2=coronalfilter(arr,point[1])
    arr3,ker3=saggitalfilter(arr,point[2])
    
    ker12=sp.logical_or(ker1,ker2)
    ker=sp.logical_or(ker12,ker3)

    arr=arr*ker

    return arr

def loadeleftheriosdata(fname_dwi_all,dicom_dir):

    arr,voxsz,affine = form.loadvol(fname_dwi_all)

    print 'Array shape:',arr.shape

    binfo=sp.array(form.loadbinfodir(dicom_dir))
    
    bvecs=binfo[:,1:4]
    bvals=binfo[:,7]
    
    print 'Binfo shape:',binfo.shape
    print 'Bvecs shape:',bvecs.shape
    print 'Bvals shape:',bvals.shape
    print 'Bvals:',bvals
    
    return arr,voxsz,affine,bvecs,bvals

def loadfibrecupdata(filename,directionsfname,bval):
    
    arr,voxsz,affine = form.loadvol(filename)
    
    print 'Array shape:',arr.shape
    
    bvecs=form.loadbvecs(directionsfname)
    
    bvals=sp.ones(bvecs.shape[0])*bval
    
    print 'Bvecs shape:',bvecs.shape
    print 'Bvals shape:',bvals.shape
    
    return arr,voxsz,affine,bvecs,bvals

def morph_mask(vol):
    
    pass

def mask(vol,thr=0):
    
    mask=vol.copy()
    mask[vol<=thr]=0
    mask[vol>thr]=1        
    mask=mask.astype('bool')
    
    return mask

def voxeldiffusivity(arr,voxx,voxy,voxz,bvals):

    S=arr[voxx,voxy,voxz,:]
    
    S0=S[0]
    
    S1=S[1:]

    S0=S0.astype(sp.float64)
    S1=S1.astype(sp.float64)

    bval=bvals[1:]

    bval=bval.astype(sp.float64)
    
    D=-(1/bval)*sp.log(S1/S0)

    return D

def subsample(arr,option='default'):
    
    pass
    
def medianfilter():
    pass

def volumediffusivity(arr,bvals,S0ind=0,Sind=sp.arange(1,65),S0thr=100,remnans=1,reminfs=1,remnegs=1):
    
    '''
    Calculates the diffusivity for every gradient direction of every voxel in a volume. 
    '''

    if arr.ndim!=4:
        
        print('Incorrect number of dimensions for arr - it has to be 4d')
        
        return

    nvoxs=arr.shape[0]*arr.shape[1]*arr.shape[2]

    S0=arr[:,:,:,S0ind]    
    
    if S0thr:
        S0[S0<S0thr]=0
    
    S0=S0.astype(sp.float64)
    
    S1=arr[:,:,:,Sind]
    
    lS=len(Sind)
    S0=S0.reshape(nvoxs)
    S1=S1.reshape(nvoxs,lS).transpose()
        
    B=bvals[Sind].astype(sp.float64)
    
    D=-1/B*sp.log(S1/S0).transpose()
    
    if remnans:
        D[sp.isnan(D)]=0.0
    if reminfs:
        D[sp.isinf(D)]=0.0
    if remnegs:
        D[D<0]=0.0
    
    D=D.reshape(arr.shape[0],arr.shape[1],arr.shape[2],lS)
    
    return D

def volumediffusivityprojected(D,voxsz,bvecs,option='square'):
    
    '''
    Projects all the bvecs on the pvecs and then multiplies the square of the result 
    with the relative diffusivity and adds them up for every pvec.
    
    i.e. sum(d(dot(b,p)^2) for every p  where b=b-vec, p=p-vec, d = diffusivity 
    
    This algorithm returns a 4D array pD and pvecs.
        
    '''
    
        #Get primary directions
    pvecs=primarydirections(voxsz,normalize=True)
    
    print 'No of primary vectors', pvecs.shape[0]
    print 'bvecs.shape',bvecs.shape
    print 'pvecs',pvecs
    print 'pvecs.shape',pvecs.shape
    
    #Multiply primary directions with bvecs
    #projvecs=sp.dot(bvecs[Sind],pvecs.T)
    projvecs=sp.dot(bvecs,pvecs.T)

    if option=='square':
        projvecs=projvecs**2
    elif option=='fourth':
        projvecs=projvecs**4
        
        
        
    #print 'projvecs',projvecs
    print 'projvecs.shape',projvecs.shape
    
    print 'Dshape',D.shape
    
    #Multiply diffusivities with projected vectors

    Dsh=D.shape

    D=D.reshape(Dsh[0]*Dsh[1]*Dsh[2],Dsh[3])
    itrD=iter(D)
    
    pD=sp.zeros((Dsh[0],Dsh[1],Dsh[2],pvecs.shape[0]))
    ndi=sp.ndindex(Dsh[0],Dsh[1],Dsh[2])
    
    if option=='square' or option=='fourth':
    
        while True:
            
            try:
                d=itrD.next()
                #print 'd.shape',d.shape
                pD[ndi.next()]=sp.sum(sp.dot(sp.diag(d),projvecs),axis=0)
                #pD[ndi.next()]=sp.sum(sp.absolute(sp.dot(sp.diag(d),projvecs))**2,axis=0)
            except:
                print general.exceptinfo()
                #print('Done')
                break
        
        
    else:
        while True:
            
            try:
                d=itrD.next()
                #print 'd.shape',d.shape
                pD[ndi.next()]=sp.sum(sp.absolute(sp.dot(sp.diag(d),projvecs)),axis=0)
                #pD[ndi.next()]=sp.sum(sp.absolute(sp.dot(sp.diag(d),projvecs))**2,axis=0)
            except:
                print general.exceptinfo()
                #print('Done')
                break
            
    
    
    print 'pD.shape',pD.shape
    print 'pD.min',pD.min()
    print 'pD.max',pD.max()

    #print 'Normalize pD'
       

    return pD,pvecs



def volumediffusivitystats(D):
    
    print 'Dshape',D.shape

    indn=sp.where(D<0.0)
    
    print 'indn',indn
    print 'len indn', len(indn)
    
    print 'D[indn]',D[indn]
    print 'len D[indn]',len(D[indn]),len(D[indn])/float(D.size)*100,'%'    
    
    del indn
    indnan=sp.where(sp.isnan(D))
    
    print 'indnan',indnan
    print 'len indnan', len(indnan)
    
    print 'D[indnan]',D[indnan]
    print 'len D[indnan]',len(D[indnan]),len(D[indnan])/float(D.size)*100,'%'
    del indnan

    indinf=sp.where(sp.isinf(D))
    
    print 'indinf',indinf
    print 'len indinf', len(indinf)
    
    print 'D[indinf]',D[indinf]
    print 'len D[indinf]',len(D[indinf]),len(D[indinf])/float(D.size)*100,'%'
    
    del indinf

    indneginf=sp.where(sp.isneginf(D))
    
    print 'indneginf',indneginf
    print 'len indneginf', len(indneginf)
    
    print 'D[indneginf]',D[indneginf]
    print 'len D[indneginf]',len(D[indneginf]),len(D[indneginf])/float(D.size)*100,'%'
    del indneginf


    indposinf=sp.where(sp.isposinf(D))
    
    print 'indposinf',indposinf
    print 'len indposinf', len(indposinf)
    
    print 'D[indposinf]',D[indposinf]
    print 'len D[indposinf]',len(D[indposinf]),len(D[indposinf])/float(D.size)*100,'%'
    del indposinf


def voxelsignal(arr,voxx,voxy,voxz):
    
    S=arr[voxx,voxy,voxz,:]
    
    return S

def voxelneighb3x3x3(arr,centx,centy,centz,gap=4,shift=sp.array([0,0,0])):
    '''
    Inputs are a 3d numpy array and voxel coordinates centx,centy,centz
    Returns the voxel indices of the 3x3x3 neighborhood and the position of the centers of the neighb. voxels 
    in space after mult. with gap and adding shift    
    '''    
    trans=sp.array([[0,0,0],[1,0,0],[1,-1,0],[0,-1,0],[-1,-1,0],[-1,0,0],[-1,1,0],[0,1,0],[1,1,0],
                    [0,0,1],[1,0,1],[1,-1,1],[0,-1,1],[-1,-1,1],[-1,0,1],[-1,1,1],[0,1,1],[1,1,1],    
                    [0,0,-1],[1,0,-1],[1,-1,-1],[0,-1,-1],[-1,-1,-1],[-1,0,-1],[-1,1,-1],[0,1,-1],[1,1,-1]])

    center=sp.array([centx,centy,centz])
    
    return trans+center,gap*trans+shift
        

def showvoxelneighb3x3x3(ren,voxinds,centers,R,IND,x,y,z,u,v,w,colr,colg,colb,opacity,texton=1):
        
    itrv=iter(voxinds)
    itrc=iter(centers)
    itrR=iter(R)
    itrin=iter(IND)
    #print voxinds.shape
    #print centers.shape
    #print R.shape
    
    while   True:
        try:

            voxi=itrv.next()
            centi=itrc.next()    
            r=itrR.next()
            ind=itrin.next()
            
            xn,yn,zn,un,vn,wn,rn=x[ind],y[ind],z[ind],u[ind],v[ind],w[ind],r[ind]
            colrn,colgn,colbn,opacityn= colr[ind],colg[ind],colb[ind],opacity[ind]
            
            #lights.pipeplot(ren,x+centi[0],y+centi[1],z+centi[2],u+centi[0],v+centi[1],w+centi[2],r,colr,colg,colb,opacity,texton)
            lights.pipeplot(ren,xn+centi[0],yn+centi[1],zn+centi[2],un+centi[0],vn+centi[1],wn+centi[2],rn,colrn,colgn,colbn,opacityn,texton)        
            
        except StopIteration:

            #print general.exceptinfo()
            break
    
    return        

    
    
def showvoxelneighb3x3x3urchine(ren,voxinds,centers,R,IND,x,y,z,u,v,w,colr,colg,colb,opacity,texton=1):
        
    itrv=iter(voxinds)
    itrc=iter(centers)
    itrR=iter(R)
    itrin=iter(IND)
    print voxinds.shape
    print centers.shape
    print R.shape
    
    while   True:
        try:

            voxi=itrv.next()
            centi=itrc.next()    
            r=itrR.next()
            ind=itrin.next()
            
            xn,yn,zn,un,vn,wn,rn=x[ind],y[ind],z[ind],u[ind],v[ind],w[ind],r[ind]
            colrn,colgn,colbn,opacityn= colr[ind],colg[ind],colb[ind],opacity[ind]
            
            #lights.pipeplot(ren,x+centi[0],y+centi[1],z+centi[2],u+centi[0],v+centi[1],w+centi[2],r,colr,colg,colb,opacity,texton)
            lights.pipeplot(ren,xn+centi[0],yn+centi[1],zn+centi[2],un+centi[0],vn+centi[1],wn+centi[2],rn,colrn,colgn,colbn,opacityn,texton)        
            
        except StopIteration:

            #print general.exceptinfo()
            break
    
    return        



def indvoxelneighb(arr,voxinds,operation='default',option='default',bvals=None):

    R=[]
    itrv=iter(voxinds)
    IND=[]
    while   True:
        
        try:
            voxi=itrv.next()            
            r=arr[voxi[0],voxi[1],voxi[2]]  
                        
            if operation=='default':                
                r=r*1.0
                r=1/r                    
            else:                
                pass
                            
            if option=='default':
                try:
                    lr=len(r)    
                    IND.append(sp.arange(lr))                    
                except:
                    print('general',general.exceptinfo())                    
                
            elif option=='min':
                
                IND.append(vectormanip(r,option='min'))
                
            elif option=='max':
                
                IND.append(vectormanip(r,option='max'))
            
            elif option=='sort':
                
                IND.append(vectormanip(r,option='max'))                
            
            else:
                print('I do not know what to do')
                IND.append(sp.NaN)                      
            
            R.append(r)
            
        except StopIteration:

            #print general.exceptinfo()
            R=sp.array(R)
            IND=sp.array(IND)
            break

    return R,IND

def vectormanip(v,option='max',mask=None,avgs=1):
    '''
        Manipulate a scipy one-dimensional array v and output its indices

        Example:
        
            v=sp.array([5, 4, 3, 2])
            ind=lofidi.vectormanip(v,option='boolean',mask=sp.array([1,0,0,1]))
            v[ind]=sp.array([5, 2])        
        
    '''
    
    if option=='max':
        try:
            ind=v.argmax()        
        except:
            ind = 0
    
    elif option=='min':
        try:
            ind=v.argmax()
        except:
            ind = 0
    
    elif option=='sort':
        try:
            ind=v.argsort()
        except:
            ind = 0
            
    elif option=='boolean':

        if mask!=None:
            try:
                ind = sp.where(mask>0)
            except:
                ind = sp.arange(1)
                
    elif option=='avgs':
                
        if avgs>1:
            
            list_ind=[] 
            
            for i in xrange(avgs):                
                ind=sp.where(mask==i)                
                list_ind.append(ind)
                
            return list_ind
            
        else:
            
            ind=0    
        
    else:
        
        try:
            lv=len(v)
        except:
            lv=1
        
        ind = sp.arange(lv)
        
    return ind    
        
        
def simpletensor(arr,bvals,bvecs,S0ind,Sind,thr=50.0): 
    '''
    Calculate tensors from a 4d numpy array and return an FA image and much more.
    bvals and bvecs must be provided as well.

    FA calculated from Mori et.al, Neuron 2006
    
    See also David Tuch PhD thesis p. 64 and Mahnaz Maddah thesis p. 44 for the tensor derivation.
    
    What this algorithm does? Solves a system of equations for every voxel j
    
    g0^2*d00 + g1^2*d11+g2^2*d22+ 2*g1*g0*d01+ 2*g0*g2*d02+2*g1*g2*d12 = - ln(S_ij/S0_j)/b_i
    
    where b_i the current b-value and g_i=[g0,g1,g2] the current gradient direction. dxx are the values of 
    the symmetric matrix D. dxx are also the unknown variables.
    
    D=  [[d00 ,d01,d02],
             [d01,d11,d12],
             [d02,d12,d22]]
            
    Output:
    
    '''
    if arr.ndim!=4:
        print('Please provide a 4d numpy array as arr here')
        return      
 
    B=bvals[Sind].astype('float32')
    G=bvecs[Sind].astype('float32')
    
    print 'B.shape',B.shape
    print 'G.shape',G.shape

    arsh=arr.shape
    volshape=(arsh[0],arsh[1],arsh[2])
    
    voxno=arsh[0]*arsh[1]*arsh[2]
    
    directionsno=len(Sind)
 
    arr=arr.astype('float32')

    #A=sp.zeros((directionsno,6))
        
    #FA=sp.zeros(volshape,dtype='float32')
    #msk=sp.zeros(volshape,dtype='float32')    
    
    S=arr[:,:,:,Sind]    
    S0=arr[:,:,:,S0ind]
        
    S0[S0<thr]=0.0
    
    print 'S.shape',S.shape
    print 'S0.shape',S0.shape
           
    S=S.reshape(voxno,directionsno)    
    S0=S0.reshape(voxno)
    
    S=S.transpose()    
    #S[S<1.0]=1.0
    #S0[S0<1.0]=1.0
    
    print '#voxno',voxno
    print '#directionsno',directionsno
    print '#S.shape',S.shape
    print '#S0.shape',S0.shape
        
    #S[S<thr]=0
    
    S=sp.log(S/S0)
    
    print 'S.shape',S.shape
    print 'S0.shape',S0.shape    

    S=S.transpose()
    print 'S.shape',S.shape
    print 'S0.shape',S0.shape

    #Remove NaNs (0/0) and Inf (very small numbers in log)
    S[sp.isnan(S)]=0
    S[sp.isinf(S)]=0
    

    S=-S/B
    
    itrG=iter(G)
    #itrA=iter(A)
       
    A=[] # this variable will hold the matrix of the Ax=S system  which we will solve for every voxel
     
    while True:
        
        try:
            g=itrG.next()        
            #g1,g2,g3=g[0],g[1],g[2]        
            #A.append(sp.array([g1*g1,g2*g2,g3*g3,2*g1*g2,2*g1*g3,2*g2*g3]))
            A.append(sp.array([g[0]*g[0],g[1]*g[1],g[2]*g[2],2*g[0]*g[1],2*g[0]*g[2],2*g[1]*g[2]]))    
            
        except StopIteration:
            A=sp.array(A)
            break
        
    print 'A.shape',A.shape
    print 'S.shape',S.shape
    print 'S0.shape',S0.shape
    
    S=S.transpose()
    
    #Remove NaNs (0/0) and Inf (very small numbers in log)
    #S[sp.isnan(S)]=1
    #S[sp.isinf(S)]=1
    
    d,resids,rank,sing=la.lstsq(A,S)
    
    print 'd.shape',d.shape
    
    d=d.transpose()
    
    print 'd.shape',d.shape
        
    itrd=iter(d)
    
    tensors=[]
    
    while True:
        
        try:
        
            d00,d11,d22,d01,d02,d12=itrd.next()
            #print x0,x1,x2,x3,x4,x5
            
            D=sp.array([[d00, d01, d02],[d01,d11,d12],[d02,d12,d22]])
                            
            evals,evecs=la.eigh(D)
                
            l1=evals[0]; l2=evals[1]; l3=evals[2]
                 
            FA=sp.sqrt( ( (l1-l2)**2 + (l2-l3)**2 + (l3-l1)**2 )/( 2*(l1**2+l2**2+l3**2) )  )                       
            
            #tensors.append(sp.array([l1,l2,l3,evecs[0,0],evecs[1,0],evecs[2,0],evecs[0,1],evecs[1,1],evecs[2,1],evecs[0,2],evecs[1,2],evecs[2,2],FA]))
            tensors.append([l1,l2,l3,evecs[0,0],evecs[1,0],evecs[2,0],evecs[0,1],evecs[1,1],evecs[2,1],evecs[0,2],evecs[1,2],evecs[2,2],FA])
            
        except StopIteration:
            
            tensors=sp.array(tensors)
               
            break
    
    tensors[sp.isnan(tensors)]=0
    tensors[sp.isinf(tensors)]=0
    
    tensors=tensors.reshape((arsh[0],arsh[1],arsh[2],13))
    
    print 'tensors.shape:', tensors.shape
        
    return tensors      

def testshowsignalfibrecup():  
    
    
    '''
    See also showdiffusivityfibrecup

    When option='min' in indvoxelneighb look the changes in shape
    
    Array shape: (64, 64, 3, 130)
    Bvecs shape: (130, 3)
    Bvals shape: (130,)
    IND [ 21 120  52  58 102 116  28  80  86  56  14 103  87   6 120  85  63 107
    58  68 106  96 102  56  60  38  46]
    voxinds.shape (27, 3)
    centers.shape (27, 3)
    R.shape (27, 130)
    IND.shape (27,)
    x.shape (130,)
    
    Showvoxelneighb here will visualize the min 1/S where S is the signal for every voxel
    
    When option='default'  in indvoxelneighb i.e. all the signal is visualized therefore if len(S)=130 then 
    every ind of IND will have 130 elements.

    Array shape: (64, 64, 3, 130)
    Bvecs shape: (130, 3)
    Bvals shape: (130,)
    IND [[  0   1   2 ..., 127 128 129]
    [  0   1   2 ..., 127 128 129]
    [  0   1   2 ..., 127 128 129]
    ..., 
    [  0   1   2 ..., 127 128 129]
    [  0   1   2 ..., 127 128 129]
    [  0   1   2 ..., 127 128 129]]
    voxinds.shape (27, 3)
    centers.shape (27, 3)
    R.shape (27, 130)
    IND.shape (27, 130)
    x.shape (130,)

    
    ''' 

    #fname='/home/eg01/Data/Fibre_Cup/3x3x3/dwi-b0650.nii'
    #directions='/home/eg01/Data/Fibre_Cup/3x3x3/diffusion_directions.txt'
    
    fname='/home/eg309/Data/Fibre_Cup/3x3x3/dwi-b0650.nii'
    directions='/home/eg309/Data/Fibre_Cup/3x3x3/diffusion_directions.txt'
        
    seeds3x3x3mm=sp.array([[51,23,1],[46,21,1],[51,34,1],[47, 32, 1],[41, 33, 1], [46, 38, 1], [44, 46, 1], [38, 48, 1],	 [31, 39, 1] , [21, 48, 1] , [17, 45, 1], [12, 40, 1], [11, 25, 1] , [12, 17, 1], [24, 24, 1], [36, 9, 1] ])
    seeds6x6x6mm=sp.array([[40,26,0],[38,25,0],[40,31,0],[38, 30, 0],[36, 32, 0], [38, 34, 0], [37, 37, 0], [34, 39, 0],	 [30, 34, 0] , [26, 39, 0] , [24, 37, 0], [21, 35, 0], [20, 27, 0] , [22, 23, 0], [28, 26, 0], [33, 20, 0] ])
    
    arr,voxsz,affine,bvecs,bvals=loadfibrecupdata(fname,directions,bval=650)    
   
    centx=arr.shape[0]/2
    centy=arr.shape[1]/2
    centz=arr.shape[2]/2

    indS1=sp.arange(1,65)

    u,v,w=bvecs[:,0],bvecs[:,1],bvecs[:,2]
    #u,v,w=bvecs[indS1,0],bvecs[indS1,1],bvecs[indS1,2]
    
    x,y,z=-u,-v,-w    
    lu=len(u)
    colr,colg,colb,opacity=sp.ones(lu),sp.zeros(lu),sp.zeros(lu),sp.ones(lu)
    
    
    curx,cury,curz=centx,centy,centz
    
    #curx,cury,curz=45,31,1
    
    #curx,cury,curz=38,48,1   
    
    voxinds,centers=voxelneighb3x3x3(arr,centx=curx,centy=cury,centz=curz,gap=4,shift=sp.array([0,0,0]))    
    R,IND=indvoxelneighb(arr,voxinds,option='default')    
    print 'IND',IND       
    
    print 'voxinds.shape',voxinds.shape
    print 'centers.shape',centers.shape
    print 'R.shape',R.shape
    print 'IND.shape',IND.shape
    print 'x.shape',x.shape
        
    ren=lights.renderer()      

    showvoxelneighb3x3x3(ren,voxinds,centers,R,IND,x,y,z,u,v,w,colr,colg,colb,opacity,texton=0)

    ax=lights.axes(scale=(6,6,6),opacity=0.5)
    ren.AddActor(ax)
    ren.ResetCamera()
    ap=lights.AppThread(frame_type=0,ren=ren,width=1024,height=800)    
    

def testdiffusivityprojection():

    '''
    fname='/home/eg01/Data/Fibre_Cup/3x3x3/dwi-b0650.nii'
    directions='/home/eg01/Data/Fibre_Cup/3x3x3/diffusion_directions.txt'
    
    #fname='/home/eg309/Data/Fibre_Cup/3x3x3/dwi-b0650.nii'
    #directions='/home/eg309/Data/Fibre_Cup/3x3x3/diffusion_directions.txt'
    
    
    seeds3x3x3mm=sp.array([[51,23,1],[46,21,1],[51,34,1],[47, 32, 1],[41, 33, 1], [46, 38, 1], [44, 46, 1], [38, 48, 1],	 [31, 39, 1] , [21, 48, 1] , [17, 45, 1], [12, 40, 1], [11, 25, 1] , [12, 17, 1], [24, 24, 1], [36, 9, 1] ])

    seeds6x6x6mm=sp.array([[40,26,0],[38,25,0],[40,31,0],[38, 30, 0],[36, 32, 0], [38, 34, 0], [37, 37, 0], [34, 39, 0],	 [30, 34, 0] , [26, 39, 0] , [24, 37, 0], [21, 35, 0], [20, 27, 0] , [22, 23, 0], [28, 26, 0], [33, 20, 0] ])
    
    arr,voxsz,affine,bvecs,bvals=loadfibrecupdata(fname,directions,bval=650)    
    '''
    
    #'''
    #fname_dwi_all='/backup/Data/Eleftherios/CBU090133_METHODS/20090227_145404/Series_003_CBU_DTI_64D_iso_1000/dtk_dti_out/dwi_all.nii'    
    #fname_dwi_all='/home/eg309/Data/Eleftherios/CBU090133_METHODS/20090227_145404/Series_003_CBU_DTI_64D_iso_1000/dtk_dti_out/dwi_all.nii'        
    
    dicom_dir='/backup/Data/Eleftherios/CBU090133_METHODS/20090227_145404/Series_003_CBU_DTI_64D_iso_1000'
    #dicom_dir='/home/eg309/Data/Eleftherios/CBU090133_METHODS/20090227_145404/Series_003_CBU_DTI_64D_iso_1000'
 
    fname_dwi_all=dicom_dir+'/dtk_dti_out/dwi_all.nii'
       
    arr,voxsz,affine,bvecs,bvals=loadeleftheriosdata(fname_dwi_all,dicom_dir)    
    #'''
    
    #Demo dataset
    
    #arr=sp.array([[[0,100,100,0],[100,100,100,100],[0,100,100,0]],[[0,100,100,0],[100,100,100,100],[0,100,100,0]],[[0,100,100,0],[100,100,100,100],[0,100,100,0]]])

    
    
    print 'Arr.shape',arr.shape
    centx=arr.shape[0]/2
    centy=arr.shape[1]/2
    centz=arr.shape[2]/2
    
    Sind=sp.arange(1,65)
    
    print 'Arr.min', arr.min()
    print 'Arr.max', arr.max()
    

    now=time.clock()
    D=volumediffusivity(arr,bvals,S0ind=0,Sind=Sind,S0thr=0)

    print 'Time:',time.clock()-now

    print 'Dmin',D.min()
    print 'Dmax',D.max()
    
    import draw

    draw.pyplot.figure(1)
    draw.pyplot.hist(D,100)
    draw.pyplot.show()
    
    form.savevol(os.path.dirname(fname_dwi_all)+'/D.nii',D,affine)
    #return

    #D=normalizeln(D,xp=sp.array([D.min(),D.max()]),yp=sp.array([0,255]))
    
    #print 'After normalization',D.min(), D.max()

    now=time.clock()
    pD,pvecs=volumediffusivityprojected(D,voxsz,bvecs[Sind])
    
    print 'Time:',time.clock()-now

    print('pD.MaX',pD.max())
    
    '''
    
    #form.savevol('Dvol.nii',D,affine)
    
    #Dhist,bins=sp.histogram(pD,20,new=True)
    
    import draw

    draw.pyplot.figure(1)
    draw.pyplot.hist(pD,20)
    draw.pyplot.show()
    
    '''
    
    
    #pD=normalizeln(pD,xp=sp.array([pD.min(),pD.max()]),fp=sp.array([0,1]))
    pD=normalizeln(pD,xp=(pD.min(),pD.max()),fp=sp.array([0.0,1.0]))
    
    print 'pD.shape',pD.shape
    print 'pD.min',pD.min()
    print 'pD.max',pD.max()

    '''
    D,cdf=histeq(D)
    
    draw.pyplot.figure(2)
    draw.pyplot.hist(D,20)
    draw.pyplot.show()   
    
    print 'After hist. equalization', D.min(),D.max()    
    form.savevol('Dvoleq.nii',D,affine)
    '''
    
    ren=lights.renderer()

    form.savevol(os.path.dirname(fname_dwi_all)+'/pD2.nii',pD,affine)
    #form.savevol('/home/eg309/Data/Test/pD.nii',pD,affine)
    pvecs.tofile(os.path.dirname(fname_dwi_all)+'/pvecs')
    #pvecs.tofile('/home/eg309/Data/Test/pvecs')
        
    #'''
    pd=pD[centx,centy,centz]
    print 'pd',pd
    print 'pvecs.shape',pvecs.shape
    
    i=0
    for vec in pvecs:
        print 'vec',vec
        if pd[i]>0:
            ren.AddActor(lights.tube(point1=(0,0,0),point2=pd[i]*vec,color=(1,0,0.7),opacity=1,radius=0.05,capson=1))
            #lights.label(ren,text=str(sp.around(vec,decimals=2)),pos=vec,scale=(0.05,0.05,0.05),color=(1,1,1))
        lights.label(ren,text=str(sp.around(pd[i],decimals=2)),pos=vec,scale=(0.05,0.05,0.05),color=(1,1,1))
        
    
        i=i+1
    
    #ren.AddActor(lights.cube())
    ren.AddActor(lights.axes(scale=(2,2,2),opacity=0.3))
    ren.ResetCamera()
    ap=lights.AppThread(frame_type=0,ren=ren,width=1024,height=800)    
    #'''

def testgeneratelabels():
    
    
    dname='/backup/Data/Eleftherios/CBU090133_METHODS/20090227_145404/Series_003_CBU_DTI_64D_iso_1000/dtk_dti_out'
    fname=dname+'/pD.nii'
    #fname='/home/eg01/Data/Test/pD.nii'
    arr,voxsz,affine=form.loadvol(fname)  
    
    pvecs=sp.fromfile(dname+'/pvecs')    
    print 'pvecs.shape',pvecs.shape
    pvecs=pvecs.reshape(13,3)    
    
    fname_spm_segment=dname+'/c3dti_b0.nii'
    mask_spm,voxsz2,affine2=form.loadvol(fname_spm_segment)      
    
    print 'mask_spm.shape',mask_spm.shape
    arr[sp.where(mask_spm>0.2)]=sp.zeros(arr.shape[-1])
    
    import draw

    draw.pyplot.figure(1)
    draw.pyplot.hist(arr,100)
    draw.pyplot.show()
    
    centx=arr.shape[0]/2
    centy=arr.shape[1]/2
    centz=arr.shape[2]/2
        
    arr=sp.interp(arr,xp=sp.array([0.1,0.4]),fp=sp.array([0,1]))
    
    th=15
    pD=arr[:,:,centz-1:centz+1,:]
    #pD=arr[centx-th:centx+th,centy:centy+th,centz,:]
    #pD=arr[centx-th:centx+th,centy:centy+2,centz:centz+2]
    #pD=arr[centx-th:centx+th,centy-th:centy+th,centz-th:centz+th]
    #pD=arr[centx-th:centx+th,centy-th:centy+th,centz:centz+4]
    first=0
    second=0
    third=1
    
    print 'pD.shape',pD.shape    
    print 'pD.max',pD.max()
    print 'pD.min',pD.min()
    
    #demo
    '''
    tl=sp.linspace(0,1,pD.shape[-1])
    sh0,sh1,sh2,sh3=pD.shape[0],pD.shape[1],pD.shape[2],pD.shape[-1]
    pD=sp.tile(tl,sh0*sh1*sh2)
    pD=pD.reshape(sh0,sh1,sh2,sh3)
    '''
    #print 'pD.index',pD.ndindex
    '''
    for index, x in sp.ndenumerate(pD):
        
        print index,x
    '''
    ren=lights.renderer()
    #'''

    #red=sp.linspace(pD.min(),pD.max(),pD.shape[3])
    
    if pD.ndim>3:
    
        for index in sp.ndindex(pD.shape[0],pD.shape[1],pD.shape[2]):
            #print index
            #shift.append(index)
            shift=sp.array(index)
            #print shift
            spD=sp.sort(pD[index[0],index[1],index[2],:])
            mxpD=spD[-1]
            mxpD2=spD[-2]
            mxpD3=spD[-3]
            i=0
            for vec in pvecs:
                #print i
                if pD[index[0],index[1],index[2],i]>0:
                    #ren.AddActor(lights.tube(point1=shift,point2=pD[index[0],index[1],index[2],i]*vec+shift,color=(1,0,0),opacity=1,radius=0.05,capson=1))
                    '''
                    pd=pD[index[0],index[1],index[2],i]
                    if pd==mxpD:
                        ren.AddActor(lights.tube(point1=-pd*vec+shift,point2=pd*vec+shift,color=(pd,0,1-pd),opacity=1,radius=0.04,capson=1,specular=0,sides=3))
                        #ren.AddActor(lights.tube(point1=-0.9*vec+shift,point2=0.9*vec+shift,color=(pd,0,1-pd),opacity=1,radius=0.04,capson=1,specular=0))
                    
                    if pd==mxpD2:
                        ren.AddActor(lights.tube(point1=-pd*vec+shift,point2=pd*vec+shift,color=(pd,0,1-pd),opacity=1,radius=0.04,capson=1,specular=0,sides=3))
                        #ren.AddActor(lights.tube(point1=-0.9*vec+shift,point2=0.9*vec+shift,color=(pd,0,1-pd),opacity=1,radius=0.04,capson=1,specular=0))
                    '''
                    pd=pD[index[0],index[1],index[2],i]
                    ren.AddActor(lights.tube(point1=-pd*vec+shift,point2=pd*vec+shift,color=(pd,0,1-pd),opacity=1,radius=0.04,capson=0,specular=0,sides=1))
                    i=i+1
                else:
                    i=i+1

    elif pD.ndim==3:
        
        for index in sp.ndindex(pD.shape[0],pD.shape[1]):
            #print index
            #shift.append(index)
            #shift=sp.array(index)
            #shift=sp.array([shift[0],shift[1],0])
            if third:
                shift=sp.array([index[0],index[1],0])
            if first:
                shift=sp.array([0,index[0],index[1]])
            if second:
                shift=sp.array([index[0],0,index[1]])
            
            #print shift
            
            #mxpD=pD[index[0],index[1],:].max()
            spD=sp.sort(pD[index[0],index[1],:])
            mxpD=spD[-1]
            mxpD2=spD[-2]
            mxpD3=spD[-3]
            
            i=0
            for vec in pvecs:
                #print i
                if pD[index[0],index[1],i]>0:
                    #ren.AddActor(lights.tube(point1=shift,point2=pD[index[0],index[1],index[2],i]*vec+shift,color=(1,0,0),opacity=1,radius=0.05,capson=1))
                    pd=pD[index[0],index[1],i]
                    if pd==mxpD:
                        #ren.AddActor(lights.tube(point1=-pd*vec+shift,point2=pd*vec+shift,color=(pd,0,1-pd),opacity=1,radius=0.04,capson=1,specular=0))
                        ren.AddActor(lights.tube(point1=-vec+shift,point2=vec+shift,color=(pd,0,1-pd),opacity=1,radius=0.04,capson=1,specular=0))
                    '''    
                    if pd==mxpD2:
                        ren.AddActor(lights.tube(point1=-pd*vec+shift,point2=pd*vec+shift,color=(pd,0,1-pd),opacity=1,radius=0.04,capson=1,specular=0))                    
                    if pd==mxpD3:
                        ren.AddActor(lights.tube(point1=-pd*vec+shift,point2=pd*vec+shift,color=(pd,0,1-pd),opacity=1,radius=0.04,capson=1,specular=0))                    
                    '''
                    i=i+1
                else:
                    i=i+1

        
            
        
        
    
    #'''
    #ren=lights.scatterplot(shift)
    

    #ren.AddVolume(lights.volume(255*pD[:,:,:,0],voxsz=voxsz))
    
    #form.savevol('smallpD.nii',pD,affine)
            
    
            
    
    '''
    pd=pD[39,39,1]
    print 'pd',pd
    print 'pvecs.shape',pvecs.shape
    
    i=0
    for vec in pvecs:
        print 'vec',vec
        if pd[i]>0:
            ren.AddActor(lights.tube(point1=(0,0,0),point2=pd[i]*vec,color=(1,0,0.7),opacity=1,radius=0.05,capson=1))
            #lights.label(ren,text=str(sp.around(vec,decimals=2)),pos=vec,scale=(0.05,0.05,0.05),color=(1,1,1))
        lights.label(ren,text=str(sp.around(pd[i],decimals=2)),pos=vec,scale=(0.05,0.05,0.05),color=(1,1,1))
        
    
        i=i+1    
    '''
        
    ren.ResetCamera()
    
    #ap=lights.AppThread(frame_type=0,ren=ren,width=1024,height=800)    
    
    ap=lights.AppThread(ren=ren)
    
def testshowdiffusivityfibrecup():   
    
    '''

    We load our diffusion datasets in array arr of shape voxsz where the 3 first dimensions are for the volu
    me and the last dimention is for the actual signal in every voxel e.g. arr[voxx,voxy,voxz,S]. 
    
    Our plan is to calculate the diffusivity for every gradient direction and visualize it for every direction in a neighbourhood 
    of voxels with volume coordinates of the center curx,cury,curz.
    
    The function volumediffusivity will calculate the diffusivity D of the entire volume given as input the part of S that we need to
    use as some times they are multiple repetitions and we do not want to work with all of them.
    
    The shape of D  is now (voxx,voxy,voxz,64) (initialy S was 130 not 64).
    
    Every gradient direction is given by the vector [x_i,y_i,z_i] [u_i,v_i,w_i] and len(x) needs to be 64. See Sind.
    
    It is important to have the correct correlation between the gradients and the relative diffusivity. Therefore, we use 
    
    Therefore we are using voxelneighb3x3x3 to get the neighb. voxels' indeces and their centers and then we use the function
    
    indvoxelneighb wich returns the list R of the diffusivities for each direction of each voxel of the neighborhood and the indices
    
    IND of the gradient directions 
    

    
    Array shape: (64, 64, 3, 130)
    Bvecs shape: (130, 3)
    Bvals shape: (130,)
    Arr.shape (64, 64, 3, 130)
    D.shape (64, 64, 3, 64)
    D.max 0.00792665806
    D.min -0.00388573637586
    No of Negative Diffusivities: 112853
    Percentage of the Volume having Negative Diff.: 14.3500010173 %
    IND [[ 0  1  2 ..., 61 62 63]
    [ 0  1  2 ..., 61 62 63]
    [ 0  1  2 ..., 61 62 63]
    ..., 
    [ 0  1  2 ..., 61 62 63]
    [ 0  1  2 ..., 61 62 63]
    [ 0  1  2 ..., 61 62 63]]
    voxinds.shape (27, 3)
    centers.shape (27, 3)
    R.shape (27, 64)
    IND.shape (27, 64)
    x.shape (64,)

    The function indvoxelneighb is very important because we can use it to calculate many important features of the Signal like
    the min and max and then visualize it without any copies in memory only with indexing.
    
    '''

    #fname='/home/eg01/Data/Fibre_Cup/3x3x3/dwi-b0650.nii'
    #directions='/home/eg01/Data/Fibre_Cup/3x3x3/diffusion_directions.txt'
    
    fname='/home/eg309/Data/Fibre_Cup/3x3x3/dwi-b0650.nii'
    directions='/home/eg309/Data/Fibre_Cup/3x3x3/diffusion_directions.txt'
    
    seeds3x3x3mm=sp.array([[51,23,1],[46,21,1],[51,34,1],[47, 32, 1],[41, 33, 1], [46, 38, 1], [44, 46, 1], [38, 48, 1],	 [31, 39, 1] , [21, 48, 1] , [17, 45, 1], [12, 40, 1], [11, 25, 1] , [12, 17, 1], [24, 24, 1], [36, 9, 1] ])

    seeds6x6x6mm=sp.array([[40,26,0],[38,25,0],[40,31,0],[38, 30, 0],[36, 32, 0], [38, 34, 0], [37, 37, 0], [34, 39, 0],	 [30, 34, 0] , [26, 39, 0] , [24, 37, 0], [21, 35, 0], [20, 27, 0] , [22, 23, 0], [28, 26, 0], [33, 20, 0] ])
    
    arr,voxsz,affine,bvecs,bvals=loadfibrecupdata(fname,directions,bval=650)    

    print 'Arr.shape',arr.shape
    centx=arr.shape[0]/2
    centy=arr.shape[1]/2
    centz=arr.shape[2]/2
    
    Sind=sp.arange(1,65)

    D=volumediffusivity(arr,bvals,S0ind=0,Sind=Sind)
    
    indNEG=sp.where(D<0)
         
    print 'D.shape',D.shape
    print 'D.max',D.max()
    print 'D.min',D.min()  
    print 'D.indNEG',indNEG
    print 'D.NEG',D[indNEG]  
    print 'No of Negative Diffusivities:',len(D[indNEG])   
    print 'Percentage of the Volume having Negative Diff.:', (len(D[indNEG])/float(D.size))*100,'%'
            
    D=D*10000    
    
    u,v,w=bvecs[Sind,0],bvecs[Sind,1],bvecs[Sind,2]
    
    x,y,z=-u,-v,-w    
    lu=len(u)
    colr,colg,colb,opacity=sp.ones(lu),sp.zeros(lu),sp.zeros(lu),sp.ones(lu)
    
    curx,cury,curz=centx,centy,centz
    
    #curx,cury,curz=45,31,1    
    #curx,cury,curz=38,48,1   
    
    voxinds,centers=voxelneighb3x3x3(D,centx=curx,centy=cury,centz=curz,gap=4,shift=sp.array([0,0,0]))    
    R,IND=indvoxelneighb(D,voxinds,option='default')    
    print 'IND',IND           
    #print 'R.shape',R.shape
    
    print 'voxinds.shape',voxinds.shape
    print 'centers.shape',centers.shape
    print 'R.shape',R.shape
    print 'IND.shape',IND.shape
    print 'x.shape',x.shape
        
    ren=lights.renderer()      

    showvoxelneighb3x3x3(ren,voxinds,centers,R,IND,x,y,z,u,v,w,colr,colg,colb,opacity,texton=0)

    ax=lights.axes(scale=(6,6,6),opacity=0.5)
    ren.AddActor(ax)
    ren.ResetCamera()
    ap=lights.AppThread(frame_type=0,ren=ren,width=1024,height=800)    

def testsimpletensor():

    '''
    fname='/home/eg01/Data/Fibre_Cup/3x3x3/dwi-b0650.nii'
    directions='/home/eg01/Data/Fibre_Cup/3x3x3/diffusion_directions.txt'
    
    #fname='/home/eg309/Data/Fibre_Cup/3x3x3/dwi-b0650.nii'
    #directions='/home/eg309/Data/Fibre_Cup/3x3x3/diffusion_directions.txt'
    
    seeds3x3x3mm=sp.array([[51,23,1],[46,21,1],[51,34,1],[47, 32, 1],[41, 33, 1], [46, 38, 1], [44, 46, 1], [38, 48, 1],	 [31, 39, 1] , [21, 48, 1] , [17, 45, 1], [12, 40, 1], [11, 25, 1] , [12, 17, 1], [24, 24, 1], [36, 9, 1] ])

    seeds6x6x6mm=sp.array([[40,26,0],[38,25,0],[40,31,0],[38, 30, 0],[36, 32, 0], [38, 34, 0], [37, 37, 0], [34, 39, 0],	 [30, 34, 0] , [26, 39, 0] , [24, 37, 0], [21, 35, 0], [20, 27, 0] , [22, 23, 0], [28, 26, 0], [33, 20, 0] ])
    
    arr,voxsz,affine,bvecs,bvals=loadfibrecupdata(fname,directions,bval=650)    
    
    '''
    
    #'''
    #fname_dwi_all='/backup/Data/Eleftherios/CBU090133_METHODS/20090227_145404/Series_003_CBU_DTI_64D_iso_1000/dtk_dti_out/dwi_all.nii'
    #fname_dwi_all='/home/eg309/Data/Eleftherios/CBU090133_METHODS/20090227_145404/Series_003_CBU_DTI_64D_iso_1000/dtk_dti_out/dwi_all.nii'
    
    dicom_dir='/backup/Data/Eleftherios/CBU090133_METHODS/20090227_145404/Series_003_CBU_DTI_64D_iso_1000'
    #dicom_dir='/home/eg309/Data/Eleftherios/CBU090133_METHODS/20090227_145404/Series_003_CBU_DTI_64D_iso_1000'
    
    fname_dwi_all= dicom_dir+'/dtk_dti_out/dwi_all.nii'
    
    arr,voxsz,affine,bvecs,bvals=loadeleftheriosdata(fname_dwi_all,dicom_dir)    

    #'''
    
    print 'arr.shape',arr.shape
    
    S0ind=0
    Sind=sp.arange(1,65)
    
    tensors=simpletensor(arr,bvals,bvecs,S0ind,Sind)

    form.savevol(os.path.dirname(fname_dwi_all)+'/T.nii',tensors,affine)
    
    print 'tensors.shape',tensors.shape
    
    FA=tensors[:,:,:,12]
    
    print 'FA.shape',FA.shape
    print 'FA.max', FA.max()
    print 'FA.min', FA.min()
            
    form.savevol(os.path.dirname(fname_dwi_all)+'/FA.nii',FA,affine)
    
    '''
    ren=lights.renderer()
    
    FA=slicerfilter(FA,point=(64,64,32))
    vol=lights.volume(FA*300.0,voxsz=voxsz)
    
    ren.AddVolume(vol)
    
    ren.ResetCamera()
    ap=lights.AppThread(frame_type=0,ren=ren,width=1024,height=800)    
   
    '''
    

    
def maskoutfibrecup():
    
    fpath1_list, fpath2_list, fdirections1, fdirections2=external.Fibre_Cup_files()    
    
    print fpath2_list
    print fdirections2
    
    #fname=fpath1_list[0]
    #directions=fdirections1
    #arr,voxsz,affine,bvecs,bvals=loadfibrecupdata(fname,directions,bval=650)        
    
    fname=fpath1_list[0]
    directions=fdirections1
    arr,voxsz,affine,bvecs,bvals=loadfibrecupdata(fname,directions,bval=650)    
    
    S0=arr[:,:,:,0]

    print 'S0.shape', S0.shape
    
    N,bins=sp.histogram(S0,20,new=True)
    print 'N ',N
    print 'bins ',bins
    
    print 'Max S0 ', S0.max()
    print 'Min S0 ', S0.min()
    
    msk=mask(S0,thr=100.0)

    S0=S0[msk]
    
    print 'msk.shape',msk.shape
    print 'Max S0 ', S0.max()
    print 'Min S0 ', S0.min()
    print 'S0.shape',S0.shape
        
    form.savevol('mask_3x3x3.nii',msk.astype(int),affine)
           
    fname=fpath2_list[0]
    directions=fdirections2
    arr,voxsz,affine,bvecs,bvals=loadfibrecupdata(fname,directions,bval=650)    
    
    S0=arr[:,:,:,0]

    print 'S0.shape', S0.shape
    
    N,bins=sp.histogram(S0,20,new=True)
    print 'N ',N
    print 'bins ',bins
    
    print 'Max S0 ', S0.max()
    print 'Min S0 ', S0.min()
    
    msk=mask(S0,thr=100.0)

    S0=S0[msk]
    
    print 'msk.shape',msk.shape
    print 'Max S0 ', S0.max()
    print 'Min S0 ', S0.min()
    print 'S0.shape',S0.shape
        
    form.savevol('mask_6x6x6.nii',msk.astype(int),affine)
    
    '''
    ren=lights.renderer()
    
    vol=lights.volume(FA*300.0,voxsz=voxsz)
    
    ren.AddVolume(vol)
    
    ren.ResetCamera()
    ap=lights.AppThread(frame_type=0,ren=ren,width=1024,height=800)    
   
    '''


    
def testdirections():
        
    ren=lights.renderer()

    #vecs=primarydirections((2,2,2),normalize=False)
    vecs=primarydirections((2,2,2),normalize=True)
    
    #u,v,w=vecs[:,0],vecs[:,1],vecs[:,2]
    #x,y,z=0*u,0*u,0*u
    
    i=0
    for vec in vecs:
        
        print vec
        
        ren.AddActor(lights.tube(point1=(0,0,0),point2=vec,color=(1,0,0.7),opacity=1,radius=0.1,capson=1))
        #lights.label(ren,text=str(sp.around(vec,decimals=2)),pos=vec,scale=(0.05,0.05,0.05),color=(1,1,1))
        lights.label(ren,text=str(sp.around(i,decimals=2)),pos=1.1*vec,scale=(0.05,0.05,0.05),color=(1,1,1))
        
        
        i=i+1
    
    #ren.AddActor(lights.cube())
    ren.AddActor(lights.axes(scale=(2,2,2),opacity=0.4))
    ren.ResetCamera()
    ap=lights.AppThread(frame_type=0,ren=ren,width=1024,height=800)
    
def testexportdiffusivitiestxt():
    
    dname='/backup/Data/Eleftherios/CBU090133_METHODS/20090227_145404/Series_003_CBU_DTI_64D_iso_1000/dtk_dti_out'
    dicom_dir='/backup/Data/Eleftherios/CBU090133_METHODS/20090227_145404/Series_003_CBU_DTI_64D_iso_1000'

    fnameD=dname+'/D.nii'
    fnameDtxt=dname+'/D.txt'
    
    D,voxsz,affine=form.loadvol(fnameD)   
    
    print D.min()
    print D.max()
    #return
    
    D=D.reshape(D.shape[0]*D.shape[1]*D.shape[2],D.shape[3])

    try:
        print('Saving...')
        sp.savetxt(fnameDtxt,D,fmt='%0.6f')
        print('Array saved in '+fnameDtxt+'.')
    except:
        
        print(general.exceptinfo())
    
    #D2=sp.loadtxt(fnameDtxt)
    #print D2.shape
    #print D2[0]
    
    
def  testdemosignal(demo='gaussian'):

    dicom_dir='/home/eg01/Data/Eleftherios/CBU090133_METHODS/20090227_145404/Series_003_CBU_DTI_64D_iso_1000'
    #dicom_dir='/home/eg309/Data/Eleftherios/CBU090133_METHODS/20090227_145404/Series_003_CBU_DTI_64D_iso_1000'
    binfo=sp.array(form.loadbinfodir(dicom_dir))    
    bvecs=binfo[:,1:4]
    #bvals=binfo[:,7]
    
    b=1000.0
    
    #bvecs=form.loadbvecs(dname+'/bvecs')
    bvecs=bvecs[1:]
    
    if demo=='gaussian':

        D=sp.matrix([[0.010,0,0],[0,0.001,0],[0,0,0.001]])
        
        S0=100.0

        S=[S0*sp.exp(-b*(bv.reshape(1,3)*D*bv.reshape(3,1))[0,0]) for bv in bvecs]
        
        S=sp.array(S)
        print 'S',S
        
        #return
    
        '''
        S=50*sp.ones(bvecs.shape[0])
        
        
        width,mu,sigma,height=10,0,3,30
        
        x=sp.linspace(-width,width,S.shape[0])
        gaussian=height*sp.exp( -0.5*((x-mu)/sigma)**2)   
        
        #gaussian=height/(sigma*sp.sqrt(2*sp.pi))*sp.exp(-0.5*((x-mu)/sigma)**2)
            
        #gaussian=1/(sigma*sp.sqrt(2*sp.pi))*sp.exp(-0.5*((x-mu)/sigma)**2)
        
        S=S-gaussian
        
        print 'gaussian sum',sum(gaussian)
        '''
        
    else:

        S=50*sp.ones(bvecs.shape[0])
        
    
    #print S
    S0=100.0
    
    print 'Signal', S
    
    d=-1/b * sp.log(S/S0)
    
    print 'Diffusivities', d    
    
    print 'B-value',b
    
    pvecs=primarydirections((2,2,2))
    
    print 'pvecs',pvecs
    
    projvecs=sp.dot(bvecs,pvecs.T)
    
    print 'projvecs.shape',projvecs.shape
    
    projvecs=projvecs**20
        
    pd=sp.sum(sp.absolute(sp.dot(sp.diag(d),projvecs)),axis=0)
    
    print 'pd',pd
    print 'pd.shape',pd.shape
    
    print 'Calculating the tensor...'
    
    A=[] # this variable will hold the matrix of the Ax=S system  which we will solve for every voxel

    itrB=iter(bvecs)
    
    while True:
        
        try:
            g=itrB.next()        
            #g1,g2,g3=g[0],g[1],g[2]        
            #A.append(sp.array([g1*g1,g2*g2,g3*g3,2*g1*g2,2*g1*g3,2*g2*g3]))
            A.append(sp.array([g[0]*g[0],g[1]*g[1],g[2]*g[2],2*g[0]*g[1],2*g[0]*g[2],2*g[1]*g[2]]))    
            
        except StopIteration:
            A=sp.array(A)
            break
    
    #Check for this
    #A=-A
    print 'A.shape',A.shape
    
    S2=sp.log(S/S0)
    S2=-S2/b
    
    D,resids,rank,sing=la.lstsq(A,S2)
    
    print 'D.shape',D.shape
       
    d00,d11,d22,d01,d02,d12=D
    #print x0,x1,x2,x3,x4,x5
            
    D=sp.array([[d00, d01, d02],[d01,d11,d12],[d02,d12,d22]])
                            
    evals,evecs=la.eigh(D)
    print 'Evecs', evecs
                
    l1=evals[0]; l2=evals[1]; l3=evals[2]
                 
    FA=sp.sqrt( ( (l1-l2)**2 + (l2-l3)**2 + (l3-l1)**2 )/( 2*(l1**2+l2**2+l3**2) )  )

    print 'Tensor Done!'
    
    print 'FA',FA
    print 'Eigenvalues',l1,l2,l3
    
    se1,se2,se3=l1,l2,l3
    sv1,sv2,sv3=evecs[0],evecs[1],evecs[2]
    
    #'''
    
    print 'd',d
    print 'pd',pd
    #print 't',t
    print 'sv',sv1,sv2,sv3
    print 'se',se1,se2,se3
    
    #R=sp.array([5000*sp.absolute(t[0])*t[3:6],5000*sp.absolute(t[1])*t[6:9],5000*sp.absolute(t[2])*t[9:12]])
    #R=sp.array([10000*se1*sv1,10000*se2*sv2,10000*se3*sv3])
    #R=sp.array([sv1,sv2,sv3])
    
    #R=sp.array([1000*se1*sp.array([1,0,0]),1000*se2*sp.array([0,1,0]),1000*se3*sp.array([0,0,1])])
    #R=R.T
    #print 'R',R
 
    #R=sp.array([[20,0,0],[0,1,0],[0,0,1]])
    
    R=20000*D
    ren=lights.renderer()
    ren.AddActor(lights.ellipsoid(R,thetares=50,phires=50,opacity=0.9))
    ren.AddActor(lights.axes(scale=(50,50,50),opacity=0.2))   
    
    #'''
    P=sp.dot(sp.diag(600*pd),pvecs)
    #P=sp.vstack([P,-P])
    #ren.AddActor(lights.triangulate(P))
    i=0
    for p in P:
        ren.AddActor(lights.tube(point1=-p,point2=p,color=(0,1,1),opacity=0.9))
        ren.AddActor(lights.label(ren=ren,text=str((200*pd[i]).round(2)),pos=p,scale=(0.1,0.1,0.1)))
        i=i+1
    
    ren.AddActor(lights.triangulate(sp.vstack([P,-P]),color=(0.2,0.9,0.4),opacity=0.5))
    
    B=sp.dot(sp.diag(10000*d),bvecs)
    #B=sp.vstack([B,-B])
    #ren.AddActor(lights.triangulate(B))
    i=0
    for b in B:
        #ren.AddActor(lights.tube(point1=-b,point2=b,opacity=0.9))
        #ren.AddActor(lights.label(ren=ren,text=str((10000*d[i]).round(2)),pos=b,scale=(0.1,0.1,0.1)))
        i=i+1
        
    #'''
  
    ren.ResetCamera()    
    #ren.SetBackground(1,1,1)
    ap=lights.AppThread(frame_type=0,ren=ren,width=1024,height=800)
    
    #'''
    
def diffusivites2mat():
    
    dname='/backup/Data/Eleftherios/CBU090133_METHODS/20090227_145404/Series_003_CBU_DTI_64D_iso_1000/dtk_dti_out'
    fnameD=dname+'/D.nii'
    
    D,voxsz,affine=form.loadvol(fnameD)
    D=D.reshape(D.shape[0]*D.shape[1]*D.shape[2],D.shape[3])
    
    print D.shape
    Ddict={'D':D}
    form.savemat('D.mat',Ddict)
    

def testkmeans():
    dname='/backup/Data/Eleftherios/CBU090133_METHODS/20090227_145404/Series_003_CBU_DTI_64D_iso_1000/dtk_dti_out'
    dicom_dir='/backup/Data/Eleftherios/CBU090133_METHODS/20090227_145404/Series_003_CBU_DTI_64D_iso_1000'

    fnameD=dname+'/D.nii'
    
    fmask=dname+'/mdti_b0.nii'
    
    csfmask=dname+'/c3dti_b0.nii'
    
    wmfmask=dname+'/c2dti_b0.nii'
        
    D,voxsz,affine=form.loadvol(fnameD)   
    
    #Apply the background mask 
    #M,voxsz,affine=form.loadvol(fmask)    
    #D[M<50]=0.0
    
    #Apply the CSF mask 
    CSF,voxsz,affine=form.loadvol(csfmask)    
    D[CSF==1.0]=sp.zeros(D.shape[-1])
    
    #Apply the white matter mask 
    WM,voxsz,affine=form.loadvol(wmfmask)    
    D[WM==0.0]=sp.zeros(D.shape[-1])
    
    D=D.reshape(D.shape[0]*D.shape[1]*D.shape[2],D.shape[3])
    
    from scipy.cluster.vq import kmeans,whiten
    
    #D=whiten(D)
    
    #return

    for i in xrange(10):
        
        t1 = time.clock()
        
        #centroids,distortion=kmeans(D,30)
        cid,e,n=pcl.kcluster(D,100)
        centroids,cmask=pcl.clustercentroids(D,clusterid=cid)
        
        t2 = time.clock()
        print str(i)+':Time:', round(t2-t1, 3)

        #print 'centroids',centroids
        print 'centroids.shape',centroids.shape
        #print 'distortion',distortion
        
        sp.save('CentroidsKMeans100_pcl_masked'+str(i),centroids)
        sp.save('CIDsKMeans100_pcl_masked'+str(i),cid)
        

def testkmeansoutput():
    '''
    SSCP
    '''
    
    dname='/backup/Data/Eleftherios/CBU090133_METHODS/20090227_145404/Series_003_CBU_DTI_64D_iso_1000/dtk_dti_out'
    dicom_dir='/home/eg01/Data/Eleftherios/CBU090133_METHODS/20090227_145404/Series_003_CBU_DTI_64D_iso_1000'
    #dicom_dir='/home/eg309/Data/Eleftherios/CBU090133_METHODS/20090227_145404/Series_003_CBU_DTI_64D_iso_1000'
    binfo=sp.array(form.loadbinfodir(dicom_dir))    
    bvecs=binfo[:,1:4]
    #bvals=binfo[:,7]
    
    b=1000.0
    
    #bvecs=form.loadbvecs(dname+'/bvecs')
    bvecs=bvecs[1:]

    fnameD=dname+'/D.nii'
    
    D,voxsz,affine=form.loadvol(fnameD)   
    
    print 'D dims',D.shape[0],D.shape[1],D.shape[2],D.shape[3]
    
    dx,dy,dz=D.shape[0],D.shape[1],D.shape[2]
    
    D=D.reshape(D.shape[0]*D.shape[1]*D.shape[2],D.shape[3])

    

    from scipy.cluster.vq import vq,whiten
    
    D=whiten(D)
    
    C0=sp.load('CentroidsKMeansRun300.npy')    
    C1=sp.load('CentroidsKMeansRun301.npy')    
            
    C=C0
    
    #Sort the centroids by the mean diffusivity
    m=sp.mean(C,axis=1)
    C=C[m.argsort()]
    
    vox2clust,dist=vq(D,C)
    
    Nvox=D.shape[0]

    Nclusters=C.shape[0]
    
    Ndims=D.shape[-1]
    
    print 'N',Nvox, Nclusters, Ndims
    #print ''code
    print dist
    
    '''
    for v in voxels
        c[v]=class of v
        V[c[v]] += outer (d[v],d[v].T) 65x65  

    V1-V10 are matrices of sum of outer products of diffusivity vectors for each class
    
    V*(c) = V(c)/NoC - m(c)m(c).T
    
    and then calculate the trace V*(c)
    
    '''
    V=sp.zeros((Nclusters,Ndims,Ndims))
    #Below it calculates theSSCP
    vox=0
    for d in D:
    
        V[vox2clust[vox]]+=sp.outer(d,d)
        vox+=1
    
    Sc=[]    
    for clust in xrange(Nclusters):
        
        Sclust=vox2clust.tolist().count(clust) #Size of cluster
        Sc.append(Sclust)
        print 'clust',clust, 'size', Sclust, 'centroid vals', C[clust]
        
        #V[clust]=V[clust]/float(Sclust) - sp.dot(C[clust],C[clust])
        V[clust]=V[clust]/float(Sclust) - sp.outer(C[clust],C[clust])
    
    #print 'For the 30 clusters Vstar is '
    trV=sp.trace(V)
    
    m=sp.mean(C,axis=1)
    
    #vox2clust=vox2clust[]
    
    #return trV,V,Sc,C,D,vox2clust

    return vox2clust.reshape(dx,dy,dz),D,C
    
def showdiffusivities():

    dname='/backup/Data/Eleftherios/CBU090133_METHODS/20090227_145404/Series_003_CBU_DTI_64D_iso_1000/dtk_dti_out'
    dicom_dir='/backup/Data/Eleftherios/CBU090133_METHODS/20090227_145404/Series_003_CBU_DTI_64D_iso_1000'

    fnameD=dname+'/D.nii'
    
    fmask=dname+'/mdti_b0.nii'
    
    csfmask=dname+'/c3dti_b0.nii'
    
    wmfmask=dname+'/c2dti_b0.nii'
        
    D,voxsz,affine=form.loadvol(fnameD)   
    
    #Apply the background mask 
    #M,voxsz,affine=form.loadvol(fmask)    
    #D[M<50]=0.0
    
    #Apply the CSF mask 
    CSF,voxsz,affine=form.loadvol(csfmask)    
    D[CSF==1.0]=sp.zeros(D.shape[-1])
    
    #Apply the white matter mask 
    WM,voxsz,affine=form.loadvol(wmfmask)    
    D[WM==0.0]=sp.zeros(D.shape[-1])
       
    binfo=sp.array(form.loadbinfodir(dicom_dir))    
    bvecs=binfo[:,1:4]
    
    bvecs=bvecs[1:]
    #print bvecs
    
    centx=D.shape[0]/2
    centy=D.shape[1]/2
    centz=D.shape[2]/2
    
    
    
    return
    
def testshowcentroids3(sorted=1):

    dicom_dir='/home/eg01/Data/Eleftherios/CBU090133_METHODS/20090227_145404/Series_003_CBU_DTI_64D_iso_1000'
    #dicom_dir='/home/eg309/Data/Eleftherios/CBU090133_METHODS/20090227_145404/Series_003_CBU_DTI_64D_iso_1000'
    binfo=sp.array(form.loadbinfodir(dicom_dir))    
    bvecs=binfo[:,1:4]
    #bvals=binfo[:,7]
    
    #b=1000.0
    
    #bvecs=form.loadbvecs(dname+'/bvecs')
    bvecs=bvecs[1:]
    
    C0=sp.load('CentroidsKMeans100_pcl_masked0.npy')
    C1=sp.load('CentroidsKMeans100_pcl_masked1.npy')
    C2=sp.load('CentroidsKMeans100_pcl_masked2.npy')
    C3=sp.load('CentroidsKMeans100_pcl_masked3.npy')
    C4=sp.load('CentroidsKMeans100_pcl_masked4.npy')
    
    '''
    C0=sp.load('CentroidsKMeans30_pcl_masked0.npy')
    C1=sp.load('CentroidsKMeans30_pcl_masked1.npy')
    C2=sp.load('CentroidsKMeans30_pcl_masked2.npy')
    C3=sp.load('CentroidsKMeans30_pcl_masked3.npy')
    C4=sp.load('CentroidsKMeans30_pcl_masked4.npy')
    C5=sp.load('CentroidsKMeans30_pcl_masked5.npy')
    C6=sp.load('CentroidsKMeans30_pcl_masked6.npy')
    C7=sp.load('CentroidsKMeans30_pcl_masked7.npy')
    C8=sp.load('CentroidsKMeans30_pcl_masked8.npy')
    C9=sp.load('CentroidsKMeans30_pcl_masked9.npy')
    '''
    '''
    C0=sp.load('CentroidsKMeans30_masked0.npy')    
    C1=sp.load('CentroidsKMeans30_masked1.npy')    
    C2=sp.load('CentroidsKMeans30_masked2.npy')
    C3=sp.load('CentroidsKMeans30_masked3.npy')
    C4=sp.load('CentroidsKMeans30_masked4.npy')
    '''
    
    Cs=[C0]
    #!!!! Cs=[C0,C1,C2,C3,C4]

    #Cs=[C0,C1,C2,C3,C4]
    #C0=1000*C0
    #Cs=[C0,C1,C2,C3,C4,C5,C6,C7,C8,C9]

    ren=lights.renderer()
    
    k=0
    for C in Cs: # for each volume
        print 'C.shape',C.shape, 'k', k
        if sorted:
            #sort with the mean diffusivities
            m=sp.mean(C,axis=1)
            C=C[m.argsort()]
            C=C*1000
        
        for j in range(C.shape[0]): #for each centroid
        
            d=C[j]            
            B=sp.dot(sp.diag(50*d),bvecs)
            #B=sp.vstack([B,-B])
            #ren.AddActor(lights.triangulate(B))
            i=0
            shift=sp.array([300,0,0])
            exsh=sp.array([0,k*500,0])
            exsh2=sp.array([0,k*500+200,0])
            
            for b in B: #for each direction 
                print sp.linalg.norm(b)
                ren.AddActor(lights.tube(point1=-b+j*shift+exsh,point2=b+j*shift+exsh,radius=10,color=(0,0,1),opacity=1))
                #ren.AddActor(lights.label(ren=ren,text=str((10000*d[i]).round(2)),pos=b,scale=(0.1,0.1,0.1)))
                i=i+1
                
            #'''
            #ren.AddActor(lights.label(ren=ren,text=str(sp.mean(d).round(2)),pos=j*shift+exsh2,scale=(20,20,20)))
            
        k=k+1

    ren.ResetCamera()    
    #ren.SetBackground(1,1,1)
    ap=lights.AppThread(frame_type=0,ren=ren,width=1024,height=800)

    
def testshowcentroids2(sorted=1):

    dicom_dir='/home/eg01/Data/Eleftherios/CBU090133_METHODS/20090227_145404/Series_003_CBU_DTI_64D_iso_1000'
    #dicom_dir='/home/eg309/Data/Eleftherios/CBU090133_METHODS/20090227_145404/Series_003_CBU_DTI_64D_iso_1000'
    binfo=sp.array(form.loadbinfodir(dicom_dir))    
    bvecs=binfo[:,1:4]
    #bvals=binfo[:,7]
    
    b=1000.0
    
    #bvecs=form.loadbvecs(dname+'/bvecs')
    bvecs=bvecs[1:]
    
    C0=sp.load('CentroidsKMeansRun1000.npy')    
    C1=sp.load('CentroidsKMeansRun1001.npy')    
  
    print 'C0.shape',C0.shape
    print 'C1.shape',C1.shape


    Cs=[C0,C1]

    ren=lights.renderer()
    
    #return
    
    k=0
    for C in Cs: # for each volume

        if sorted:
            #sort with the mean diffusivities
            m=sp.mean(C,axis=1)
            C=C[m.argsort()]
        
        for j in range(C.shape[0]): #for each centroid
        
            d=C[j]            
            B=sp.dot(sp.diag(50*d),bvecs)
            #B=sp.vstack([B,-B])
            #ren.AddActor(lights.triangulate(B))
            i=0
            shift=sp.array([300,0,0])
            exsh=sp.array([0,k*500,0])
            exsh2=sp.array([0,k*500+200,0])
            
            for b in B: #for each direction 
                print sp.linalg.norm(b)
                ren.AddActor(lights.tube(point1=-b+j*shift+exsh,point2=b+j*shift+exsh,radius=10,color=(0,0,1),opacity=1))
                #ren.AddActor(lights.label(ren=ren,text=str((10000*d[i]).round(2)),pos=b,scale=(0.1,0.1,0.1)))
                i=i+1
                
            #'''
            #ren.AddActor(lights.label(ren=ren,text=str(d[j].round(2)),pos=j*shift+exsh2,scale=(20,20,20)))
            
        k=k+1
    


    ren.ResetCamera()    
    #ren.SetBackground(1,1,1)
    ap=lights.AppThread(frame_type=0,ren=ren,width=1024,height=800)
    
    
def testshowcentroids(sorted=1):

    dicom_dir='/home/eg01/Data/Eleftherios/CBU090133_METHODS/20090227_145404/Series_003_CBU_DTI_64D_iso_1000'
    #dicom_dir='/home/eg309/Data/Eleftherios/CBU090133_METHODS/20090227_145404/Series_003_CBU_DTI_64D_iso_1000'
    binfo=sp.array(form.loadbinfodir(dicom_dir))    
    bvecs=binfo[:,1:4]
    #bvals=binfo[:,7]
    
    b=1000.0
    
    #bvecs=form.loadbvecs(dname+'/bvecs')
    bvecs=bvecs[1:]
    
    C0=sp.load('CentroidsKMeansRun0.npy')    
    C1=sp.load('CentroidsKMeansRun1.npy')    
    C2=sp.load('CentroidsKMeansRun2.npy')    
    C3=sp.load('CentroidsKMeansRun3.npy')    
    C4=sp.load('CentroidsKMeansRun4.npy')

    Cs=[C0,C1,C2,C3,C4]

    ren=lights.renderer()
    
    
    
    k=0
    for C in Cs: # for each volume

        if sorted:
            #sort with the mean diffusivities
            m=sp.mean(C,axis=1)
            C=C[m.argsort()]
        
        for j in range(10): #for each centroid
        
            d=C[j]            
            B=sp.dot(sp.diag(50*d),bvecs)
            #B=sp.vstack([B,-B])
            #ren.AddActor(lights.triangulate(B))
            i=0
            shift=sp.array([300,0,0])
            exsh=sp.array([0,k*500,0])
            exsh2=sp.array([0,k*500+200,0])
            
            for b in B: #for each direction 
                print sp.linalg.norm(b)
                ren.AddActor(lights.tube(point1=-b+j*shift+exsh,point2=b+j*shift+exsh,radius=10,color=(0,0,1),opacity=1))
                #ren.AddActor(lights.label(ren=ren,text=str((10000*d[i]).round(2)),pos=b,scale=(0.1,0.1,0.1)))
                i=i+1
                
            #'''
            ren.AddActor(lights.label(ren=ren,text=str(sp.mean(d).round(2)),pos=j*shift+exsh2,scale=(20,20,20)))
            
        k=k+1
    


    ren.ResetCamera()    
    #ren.SetBackground(1,1,1)
    ap=lights.AppThread(frame_type=0,ren=ren,width=1024,height=800)
    
    
def  testshowdifferenceswithtensors():  
    
    #dname='/backup/Data/Eleftherios/CBU090133_METHODS/20090227_145404/Series_003_CBU_DTI_64D_iso_1000/dtk_dti_out'
    #dicom_dir='/backup/Data/Eleftherios/CBU090133_METHODS/20090227_145404/Series_003_CBU_DTI_64D_iso_1000'

    dname='/home/eg309/Data/Eleftherios/CBU090133_METHODS/20090227_145404/Series_003_CBU_DTI_64D_iso_1000/dtk_dti_out'
    dicom_dir='/home/eg309/Data/Eleftherios/CBU090133_METHODS/20090227_145404/Series_003_CBU_DTI_64D_iso_1000'
    
    
    fnamepD=dname+'/pD.nii'
    fnameD=dname+'/D.nii'
    fnameT=dname+'/T.nii'
    fnamee1=dname+'/dti_e1.nii'
    fnamee2=dname+'/dti_e2.nii'
    fnamee3=dname+'/dti_e3.nii'
    
    fnamev1=dname+'/dti_v1.nii'
    fnamev2=dname+'/dti_v2.nii'
    fnamev3=dname+'/dti_v3.nii'
    
    pvecs=sp.fromfile(dname+'/pvecs')    
    print 'pvecs.shape',pvecs.shape
    pvecs=pvecs.reshape(13,3)   
    
    binfo=sp.array(form.loadbinfodir(dicom_dir))    
    bvecs=binfo[:,1:4]
    bvals=binfo[:,7]
    
    #bvecs=form.loadbvecs(dname+'/bvecs')
    bvecs=bvecs[1:]
    print 'bvecs.shape',bvecs.shape    
    
    D,voxsz,affine=form.loadvol(fnameD)      
    pD,voxsz,affine=form.loadvol(fnamepD)  
    T,voxsz,affine=form.loadvol(fnameT)
    
    e1,voxsz,affine=form.loadvol(fnamee1)
    e2,voxsz,affine=form.loadvol(fnamee2)
    e3,voxsz,affine=form.loadvol(fnamee3)
    v1,voxsz,affine=form.loadvol(fnamev1)
    v2,voxsz,affine=form.loadvol(fnamev2)
    v3,voxsz,affine=form.loadvol(fnamev3)

    print 'e1.shape',e1.shape
    print 'v1.shape',v1.shape
    print 'e2.shape',e2.shape
    print 'v2.shape',v2.shape
    print 'e3.shape',e3.shape
    print 'v3.shape',v3.shape

    print 'Loaded in Memory:',(T.nbytes+pD.nbytes+D.nbytes+e1.nbytes+e2.nbytes+e3.nbytes+v1.nbytes+v2.nbytes+v3.nbytes)/1024.0**2, 'MBytes.'

    #'''
    centx=D.shape[0]/2
    centy=D.shape[1]/2 
    centz=D.shape[2]/2
    #'''
    
    '''
    centx=73
    centy=82
    centz=32
    '''
    
    d=D[centx,centy,centz]
    pd=pD[centx,centy,centz]
    t=T[centx,centy,centz]
    sv1=v1[centx,centy,centz,0]
    sv2=v2[centx,centy,centz,0]
    sv3=v3[centx,centy,centz,0]
    se1=e1[centx,centy,centz]
    se2=e2[centx,centy,centz]
    se3=e3[centx,centy,centz]
    
    print 'd',d
    print 'pd',pd
    print 't',t
    print sv1,sv2,sv3
    print se1,se2,se3
    
    #R=sp.array([5000*sp.absolute(t[0])*t[3:6],5000*sp.absolute(t[1])*t[6:9],5000*sp.absolute(t[2])*t[9:12]])
    R=sp.array([1000*se1*sv1,1000*se2*sv2,1000*se3*sv3])
    R=R.T
    print R
 
    #R=sp.array([[2,0,0],[0,1,0],[0,0,1]])
    
    ren=lights.renderer()
    ren.AddActor(lights.ellipsoid(R,thetares=20,phires=20,opacity=0.3))
    ren.AddActor(lights.axes(scale=(5,5,5),opacity=0.2))   
    
    
    P=sp.dot(sp.diag(4*pd),pvecs)
    #P=sp.vstack([P,-P])
    #ren.AddActor(lights.triangulate(P))
    i=0
    for p in P:
        ren.AddActor(lights.tube(point1=-p,point2=p,color=(0,1,1),opacity=0.4))
        ren.AddActor(lights.label(ren=ren,text=str((4*pd[i]).round(2)),pos=p,scale=(0.1,0.1,0.1)))
        i=i+1
    
    B=sp.dot(sp.diag(500*d),bvecs)
    #B=sp.vstack([B,-B])
    #ren.AddActor(lights.triangulate(B))
    i=0
    for b in B:
        ren.AddActor(lights.tube(point1=-b,point2=b,opacity=0.2))
        ren.AddActor(lights.label(ren=ren,text=str((500*d[i]).round(2)),pos=b,scale=(0.1,0.1,0.1)))
        i=i+1
  
    ren.ResetCamera()
    
    ap=lights.AppThread(frame_type=0,ren=ren,width=1024,height=800)

    
def testdelaunay():
    
    #ren=lights.renderer()
    
    vecs=primarydirections((2,2,2),normalize=True)
    
    vecs=sp.vstack((vecs,-vecs))
    
    ren=lights.surfaceplot(points=vecs)
    
    lights.scatterplot(points=vecs,ren=ren)
    ren.ResetCamera()
    ap=lights.AppThread(frame_type=0,ren=ren,width=1024,height=800)
    
def testweaveinline():
    
    from scipy import weave
    from scipy.weave import converters
    
    g=sp.rand(10000000)
    
    lg=len(g)
    sum=0
    
    code = """
                tmp=0;
                for (int i=0;i<lg;i++){
                    tmp=tmp+g(i);
                }
                return_val = tmp
                """
    sum=weave.inline(code,['g','lg'],type_converters=converters.blitz,compiler='gcc')
    
    print sum
    
    return
    
if __name__ == "__main__":

    #showsignalfibrecup()
    #showdiffusivityfibrecup()
    #import cProfile
    #cProfile.run('testsimpletensor()','fooprof')   
    
    #showsignalfibrecup()
    
    #testsimpletensor()
    #maskoutfibrecup()
    
    #testshowdiffusivityfibrecup()
    #testdiffusivityprojection()
    #testgeneratelabels()
    #testdelaunay()
    #testweaveinline()
    
    #testdirections()
    #pass
    
    #TODO 
    #PICK SOME VOXELS RANDOMLY E.G. 100 VOXELS AND 
    #testshowdifferenceswithtensors()
    #testdemosignal()
    #testkmeans()
    #testshowcentroids()
    #testshowcentroids2()
    #testexportdiffusivitiestxt()
    #testkmeansoutput()
    testshowcentroids3()
    
    
    
    