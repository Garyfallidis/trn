#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
The Dipy team for the PBC Competition

Examples
------------
>>> import pbc
>>> path='/home/eg01/Data/PBC/pbc2009icdm'
>>> G,hdr=pbc.load_training_set(path)
>>> pbc.show_training_set(G)
>>> pbc.calculate_properties_training_set(G,hdr)
>>> pbc.print_properties_training_set(G)

Notes
-------
X -> Front to Back
Y -> Left to Right
Z -> Up to Down

'''

import os
import lights
import form
import time
import scipy as sp
import numpy as np
from copy import copy,deepcopy
import numpy.linalg as npla

import volumeimages as vi
from dipy.core import track_performance as pf

from dipy.viz import fos
#from dipy.viz import phos

from dipy.core import track_metrics as tm
from dipy.core import track_learning as tl

from scipy.interpolate import splprep, splev
import scipy.ndimage as nd

import pbc1109
from pbc1109 import track_volumes as tv

import mdp

def training_set_as_dict(T,L):
    ''' Represent training set a dict whith keys 0 to 8. Where 0 to 8 corresponds to labels 
    'None','Arcuate','Cingulum','Corticospinal','Forceps Major','Fornix','Inferior Occipitofrontal Fasciculus','Superior Longitudinal Fasciculus','Uncinate'
    
    Input
    ------
    T : list of tracks represented as arrays of shape Mx3
    L : array shape N,2 representing the labels where first column is the index of the track and second column is the corresponding label
    
    Returns
    ---------
    G : dict 
    '''
    
    node={'label_name':'','tracks':[],'indices':[],'centers':sp.array([0,0,0]),'curvatures':[],'lengths':[],'mids2center':[]}
    
    G={ 0:deepcopy(node),1:deepcopy(node), 2:deepcopy(node),3:deepcopy(node),
            4:deepcopy(node),5:deepcopy(node),  6:deepcopy(node),7:deepcopy(node),
            8:deepcopy(node)}
    
    Labels=['None','Arcuate','Cingulum','Corticospinal','Forceps Major','Fornix','Inferior Occipitofrontal Fasciculus','Superior Longitudinal Fasciculus','Uncinate']
   
    for i in [0,1,2,3,4,5,6,7,8]:

        G[i]['label_name']=Labels[i]

    L=L.astype(int)
    L[:,0]=L[:,0]-1            
        
    print 'Indexing tracks to their corresponding label'
    for (i,label) in L:       
        G[label]['tracks'].append(T[i])
        G[label]['indices'].append(i)
                       
    return G

def load_training_set(path):    
    ''' Load brain1 scan1 
    '''
    
    ds=pbc1109.PBCData(path)
    print 'Loading training set which is brain 1 scan 1'
    b11=ds.get_data(1,1)
    
    #copy streamlines
    print 'Copying streamlines'
    S=b11.streams
    
    #copy hdr
    print 'Copying header'
    hdr=b11.hdr
    
    #copy track points in tracks
    print 'Copying all tracks in a list'
    tracks=[s[0] for s in S]    

    #copy labels
    labels=b11.labels

    #create a graph representing the training set
    print 'Creating a graph representing the training set'
    G=training_set_as_dict(tracks,labels)
    
    del ds
    del S
    
    R={1: 197816, 2: 15009, 3: 157189,4: 64423,5: 118191,6: 168055,7: 123041,8: 88647}

    print 'Printing indices for references '
    print R
    
    return G,hdr,R

def load_training_set_plus(path):    
    ''' Load brain1 scan1 
        plus relative indices and reference fibers
    '''
    
    ds=pbc1109.PBCData(path)
    print 'Loading training set which is brain 1 scan 1'
    b11=ds.get_data(1,1)
    
    #copy streamlines
    print 'Copying streamlines'
    S=b11.streams
    
    #copy hdr
    print 'Copying header'
    hdr=b11.hdr
    
    #copy track points in tracks
    print 'Copying all tracks in a list'
    tracks=[s[0] for s in S]    

    #copy labels
    labels=b11.labels

    #create a graph representing the training set
    print 'Creating a graph representing the training set'
    G=training_set_as_dict(tracks,labels)
    
    del ds
    del S
    
    R={1: 197816, 2: 15009, 3: 157189,4: 64423,5: 118191,6: 168055,7: 123041,8: 88647}

    relR = {}
    ref = {}
    for b in [1,2,3,4,5,6,7,8]:
        relR[b] = G[b]['indices'].index(R[b])
        ref[b] = G[b]['tracks'][relR[b]]

    print 'Printing indices for reference fibers'
    print R
    print 'Printing relative indices for reference fibers'
    print relR
    
    return G,hdr,R,relR,ref

def load_approximate_training_set(path):    
    ''' Load brain1 scan1 
    '''
    
    ds=pbc1109.PBCData(path)
    print 'Loading training set which is brain 1 scan 1'
    b11=ds.get_data(1,1)
    
    #copy streamlines
    print 'Copying streamlines'
    S=b11.streams
    
    #copy hdr
    print 'Copying header'
    hdr=b11.hdr
    
    #copy track points in tracks
    print 'Copying all tracks in a list'
    #tracks=[s[0] for s in S]    
    #!!!!!!!!!!!!
    tracksa=load_approximate_tracks(path,1,1)

    #copy labels
    labels=b11.labels

    #create a graph representing the training set
    print 'Creating a graph representing the training set'
    G=training_set_as_dict(tracksa,labels)
    
    del ds
    del S    
    
    R={1: 79032, 2: 115651, 3: 76983, 4: 17556, 5: 206781, 6: 168055, 7: 59215, 8: 88647}    

    print 'Printing indices for references '
    print R
        
    return G,hdr,R


def show_training_set(G,show=True):
    ''' Show training_set represented as a dictionary using fos
        The plan is to see the labeled bundles with different colors along with their label_names.
        In plus the start point of every track is visualized with green and then end point with red.
        The label names are hanging from the start points of the tracks.
        The blue sphere represents the center of the volume  (91,109,91).
    '''
    
    r=fos.ren()
    
    #colors_labels=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8]
    colors_labels=np.random.rand(9,3)
    #colors_labels=
    
    for g in [1,2,3,4,5,6,7,8]:
        fos.add(r,fos.line(G[g]['tracks'],colors=colors_labels[g]))
        
    for g in [1,2,3,4,5,6,7,8]:
        #for track in G[g]['tracks']:
        T=G[g]['tracks']
        start=np.array([t[0] for t in T])
        end=np.array([t[-1] for t in T])
        fos.add(r,fos.dots(start,color=(0,1,0.2)))
        fos.add(r,fos.dots(end,color=(1,0,0)))            
        fos.label(r,text='  '+str(g)+'.'+G[g]['label_name'],pos=start[0],scale=(2,2,2))
    
    #fos.add(r,fos.sphere(position=(91,109,91),radius=5,thetares=8,phires=8,color=(0,0,1),opacity=1,tessel=0))
    r.ResetCamera()    
    if show:
        fos.show(r)


def mori_atlas_labels():
    
    labels={
        0:'Unclassified',
        1:'Middle cerebellar peduncle',
        2:'Pontine crossing tract (a part of MCP)',
        3:'Genu of corpus callosum',
        4:'Body of corpus callosum',
        5:'Splenium of corpus callosum',
        6:'Fornix (column and body of fornix)',
        7:'Corticospinal tract R',
        8:'Corticospinal tract L',
        9:'Medial lemniscus R',
        10:'Medial lemniscus L',
        11:'Inferior cerebellar peduncle R',
        12:'Inferior cerebellar peduncle L',
        13:'Superior cerebellar peduncle R',
        14:'Superior cerebellar peduncle L',
        15:'Cerebral peduncle R',
        16:'Cerebral peduncle L',
        17:'Anterior limb of internal capsule R',
        18:'Anterior limb of internal capsule L',
        19:'Posterior limb of internal capsule R',
        20:'Posterior limb of internal capsule L',
        21:'Retrolenticular part of internal capsule R',
        22:'Retrolenticular part of internal capsule L',
        23:'Anterior corona radiata R',
        24:'Anterior corona radiata L',
        25:'Superior corona radiata R',
        26:'Superior corona radiata L',
        27:'Posterior corona radiata R',
        28:'Posterior corona radiata L',
        29:'Posterior thalamic radiation (include optic radiation) R',
        30:'Posterior thalamic radiation (include optic radiation) L',
        31:'Sagittal stratum (include inferior longitidinal fasciculus and inferior fronto-occipital fasciculus) R',
        32:'Sagittal stratum (include inferior longitidinal fasciculus and inferior fronto-occipital fasciculus) L' ,
        33:'External capsule R',
        34:'External capsule L',
        35:'Cingulum (cingulate gyrus) R',
        36:'Cingulum (cingulate gyrus) L',
        37:'Cingulum (hippocampus) R',
        38:'Cingulum (hippocampus) L',
        39:'Fornix (cres) / Stria terminalis (can not be resolved with current resolution) R' ,
        40:'Fornix (cres) / Stria terminalis (can not be resolved with current resolution) L' ,
        41:'Superior longitudinal fasciculus R',
        42:'Superior longitudinal fasciculus L',
        43:'Superior fronto-occipital fasciculus (could be a part of anterior internal capsule) R',
        44:'Superior fronto-occipital fasciculus (could be a part of anterior internal capsule) L' ,
        45:'Uncinate fasciculus R',
        46:'Uncinate fasciculus L',
        47:'Tapetum R',
        48:'Tapetum L'}
        
    print('Problem with labels')
    return None
    #return labels

def load_loni_atlas(path):
    
    #path='/backup/Data/ICBM_Atlas/ICBM_Wmpm'        
    
    atlasfn=path+'/ICBM_WMPM.nii'
    labelfn=path+'/LabelLookupTable.txt'
    
    if os.path.isfile(atlasfn)==False:
        
        print('File does not exist')
        
        return None,None,None

    img = vi.load(atlasfn)      

    arr = np.array(img.get_data())
    voxsz = img.get_metadata().get_zooms()
    aff = img.get_affine()
    
    #labels  = np.fromfile(labels_fname, dtype=np.uint, sep=' ')
    lines = []
    with open(labelfn, 'rt') as f:
        for line in f:
            lines.append([val.strip() for val in line.split('\t') if val.strip()])
    
    lines[-1][2]=lines[-1][2]+' '+lines[-1][3]
    lines[-1]=[lines[-1][0],lines[-1][1],lines[-1][2]]
    labels=lines    
    
    return arr,voxsz,aff,labels

def copy_object(arr,thr):
    
    arr_tmp=np.zeros(arr.shape).astype('uint8')      
    arr_tmp[np.where(arr==thr)]=1
                    
    return arr_tmp

def load_mori_atlas(atlasfn=None):
    
    
    if atlasfn==None:
        atlasfn='/home/eg01/Data/Mori_Template/FSL/JHU-WhiteMatter-labels-1mm.nii'
    
    if os.path.isfile(atlasfn)==False:
        
        print('File does not exist')
        
        return None,None,None

    img = vi.load(atlasfn)      

    arr = np.array(img.get_data())
    voxsz = img.get_metadata().get_zooms()
    aff = img.get_affine()
    
    labels=mori_atlas_labels()
    
    #return arr,voxsz,aff,labels
    print('Problem with the labels do not use this function')
    return None
    
def show_atlas(arr,voxsz,aff,labels,G):

    opacitymap=np.vstack((np.arange(1,51),np.linspace(0.02,0.1,50))).T
    opacitymap=np.vstack((opacitymap,np.array([0,0])))
    opacitymap=np.sort(opacitymap,axis=0)
    
    v=np.linspace(0,1,51)
    
    red=np.interp(v,[0,0.35,0.66,0.89,1],[0,0,1,1,0.5])
    green=np.interp(v,[0,0.125,0.375,0.64,0.91,1],[0,0,1,1,0,0])
    blue=np.interp(v,[0,0.11,0.34,0.65,1],[0.5,1,1,0,0])
    
    #colormap=np.vstack((np.arange(51),np.linspace(0,1,51),0*np.linspace(0,1,51),1-np.linspace(0,1,51))).T
    colormap=np.vstack((np.arange(51),red,green,blue)).T
    colormap=colormap.astype('float32')
    
    r=fos.ren()
    fos.clear(r)
    
    print 'min',arr.min(), 'max',arr.max()
    arr2=np.zeros(arr.shape).astype('uint8')
    
    #47	UNC-R		Uncinate fasciculus right
    ind=np.where(arr==47)    
    arr2[ind]=255    

    #48	UNC-L		Uncinate fasciculus left
    ind=np.where(arr==48)    
    arr2[ind]=255
        
    #7	CST-R		Corticospinal tract right
    ind=np.where(arr==7)    
    arr2[ind]=255
    
    #8	CST-L		Corticospinal tract left
    ind=np.where(arr==8)    
    arr2[ind]=255
    
    v=fos.volume(arr2,voxsz,aff)    
    #v=fos.volume(arr2,voxsz,aff,info=1,opacitymap=opacitymap,colormap=colormap)
    
    #c=fos.contour(arr,voxsz,levels=np.arange(1,51),colors=np.random.rand(50,3),opacities=np.linspace(0.45,0.5,50))
    
    #c=fos.contour(arr,voxsz,levels=np.arange(1,51),colors=np.random.rand(50,3),opacities=np.ones(50))
    
    #c=fos.contour(arr,voxsz,levels=np.arange(1,3),colors=np.random.rand(2,3),opacities=np.ones(2))
    #c=fos.contour(arr,voxsz,levels=np.arange(45,51),colors=np.random.rand(50,3),opacities=np.ones(50))
    
    #c=fos.contour(arr,voxsz,levels=[7,8],colors=np.random.rand(2,3),opacities=np.ones(2))
    
    #colors_labels=np.random.rand(9,3)
            

    #fos.add(r,c)
    fos.add(r,v)
    
    '''
    for g in [1,2,3,4,5,6,7,8]:
        
        bundle=G[g]['tracks']
        #bundle=[t+np.array([-90,-108,-90]) for t in bundle]
        fos.add(r,fos.line(bundle,colors=colors_labels[g]))
    '''    
    fos.show(r)
    
    
    
def calculate_properties_training_set(G,hdr):

    center=hdr['dim']/2.0
    center=pbc1109.vox2mm(center,hdr) 
    
    for g in G:       
        
        G[g]['curvatures']=[tm.mean_curvature(t) for t in G[g]['tracks'] ]        
        G[g]['mean_cluster_curvature']=np.mean(np.array(G[g]['curvatures']))
        
        G[g]['lengths']=[tm.length(t) for t in G[g]['tracks'] ]
        G[g]['mean_length']=np.mean(np.array(G[g]['lengths']))

        G[g]['mids2center']=[tm.midpoint2point(t,center) for t in G[g]['tracks']]  
        G[g]['mean_mids2center']=np.mean(np.array(G[g]['mids2center']))
             
def print_properties_training_set(G):
    
    for g in G:
        print str(g),'.',G[g]['label_name'],'\n mean_cluster_curvature', round(G[g]['mean_cluster_curvature'],3),'mean_length',round(G[g]['mean_length'],3),'mean_mids2center',round(G[g]['mean_mids2center'],2)


def load_tracks(path,brain,scan):  
    ''' Load tracks from a PBC path
    
    Parameters:
    ---------------
    path :  string
        
    brain : int
                from 1 to 3
    scan :  int
                from 1 to 2
    
    Returns:
    ---------
    tracks : sequence of arrays
    
    vol_dims : tuple
                volume dimensions
    
    '''

    ds=pbc1109.PBCData(path)
    print 'Loading training set which is brain 1 scan 1'
    b11=ds.get_data(brain,scan)
    
    #copy streamlines
    print 'Copying streamlines'
    S=b11.streams
    
    #copy hdr
    print 'Copying header'
    hdr=b11.hdr
    vol_dims=hdr['dim']
    print hdr['dim']
    
    #copy track points in tracks
    print 'Copying all tracks in a list'
    tracks=[s[0] for s in S]    
    
    #print 'Counting ...'
   
    return tracks,vol_dims

def save_pickle(fname,dix):
    import cPickle
    out=open(fname,'wb')
    cPickle.dump(dix,out)
    out.close()

def load_pickle(fname):
    import cPickle
    inp=open(fname,'rb')
    dix=cPickle.load(inp)
    inp.close()
    return dix

def helix(segs=100):
    ''' Return a helix with some noise please    
    '''

    # make ascending spiral in 3-space
    t=np.linspace(0,1.75*2*np.pi,segs)

    x = np.sin(t)
    y = np.cos(t)
    z = t

    # add noise
    #x+= np.random.normal(scale=0.1, size=x.shape)
    #y+= np.random.normal(scale=0.1, size=y.shape)
    #z+= np.random.normal(scale=0.1, size=z.shape)
    
    xyz=np.vstack((x,y,z)).T    
    return xyz

def sine(segs=100):
    
    t=np.linspace(0,1.75*2*np.pi,segs)
    
    x =t 
    y=5*np.sin(5*t)
    z=np.zeros(x.shape)
    
    xyz=np.vstack((x,y,z)).T    
    return xyz
        

def show_simultaneously_3_brains(path):
    
    tracks11, vol_dims11=load_tracks(path,1,1)
    tracks21, vol_dims21=load_tracks(path,2,1)
    tracks31, vol_dims21=load_tracks(path,3,1)
    
    tracks11z=[tm.downsample(t,10) for t in tracks11] 
    tracks21z=[tm.downsample(t,10) for t in tracks21] 
    tracks31z=[tm.downsample(t,10) for t in tracks31] 
        
    r=fos.ren()
    tracks11zshift=[t+np.array([-100,0,0]) for t in tracks11z]
    tracks21zshift=[t+np.array([50,0,0]) for t in tracks21z]
    tracks31zshift=[t+np.array([200,0,0]) for t in tracks31z]
    
    fos.clear(r)
    
    fos.add(r,fos.line(tracks11zshift,np.array([1,0,0]),opacity=0.01))
    fos.add(r,fos.line(tracks21zshift,np.array([0,0,1]),opacity=0.01))
    fos.add(r,fos.line(tracks31zshift,np.array([0,1,0]),opacity=0.01))

    fos.add(r,fos.axes(scale=(100,100,100)))

    r11_21=np.array([[47148, 105906],    
                            [15009, 61271],    
                            [33939, 88274],    
                            [88045, 91305],    
                            [118191, 206474],    
                            [19383, 237434],    
                            [192033, 184828],    
                            [187807, 3638]])
    
    r11_31=np.array([[47148, 147694],
                            [15009, 55511],
                            [33939, 179046],
                            [88045, 114472],
                            [118191, 58137],
                            [19383, 202141],
                            [192033, 212647],
                            [187807, 12772]])    
    
    for ref in r11_21:
        
        fos.add(r,fos.line(tracks11zshift[ref[0]],np.array([1,1,0]),opacity=1))
        fos.add(r,fos.line(tracks21zshift[ref[1]],np.array([1,1,0]),opacity=1))
        
    for ref in r11_31:
        fos.add(r,fos.line(tracks31zshift[ref[1]],np.array([1,1,0]),opacity=1))
        
    fos.show(r)


def show_similarity_track_bundle(tno,bundle,S):
    ''' Show similarity of a specific track using a blue (far) red (close) colormap    
    '''
           
    r=tm.track_bundle_similarity(tno,S)

    blue=np.interp(r,[r.min(),r.max()],[0,1])
    green=np.zeros(blue.shape)
    red=blue[::-1]
    colormap=np.vstack((red,green,blue)).T


    #colormap=fos.colors(ref,'jet')
    r=fos.ren()
    fos.add(r,fos.line(bundle,colormap))
    fos.add(r,fos.line(bundle[tno],np.array([1,1,0])))
    fos.show(r)

def show_similarity_most_similar_track_bundle(bundle,S):    
    ''' Show similarity of the most similar track using a blue (far) red (close) colormap.    
    '''
        
    tno=tm.most_similar_track(S)
    show_similarity_track_bundle(tno,bundle,S)

def show_cut_planes(tracks,ref):
    ''' 
    
    '''
    r=fos.ren()
    
    hit,div=pf.cut_plane(tracks,ref)
    
    for p in range(len(hit)):
 
        D=[ np.vstack((hit[p][d], hit[p][d]+div[p][d])) for d in range(len(hit[p])) ]

        D2=[ hit[p][d] for d in range(len(hit[p])) ]
        
        fos.add(r,fos.line(D,fos.red))
        fos.add(r,fos.dots(np.array(D2),fos.green))
        
    fos.add(r,fos.line(ref,fos.golden))
    
    #uncinate2=[t+np.array([30,0,0]) for t in tracks]
    
    #fos.add(r,fos.line(uncinate2,fos.blue))

    fos.show(r)

def show_cut_planes2(hit,div,ref):
    ''' 
    
    '''
    r=fos.ren()
    
    #
    
    for p in range(len(hit)):
 
        D=[ np.vstack((hit[p][d], hit[p][d]+div[p][d])) for d in range(len(hit[p])) ]

        D2=[ hit[p][d] for d in range(len(hit[p])) ]
        
        fos.add(r,fos.line(D,fos.red))
        fos.add(r,fos.dots(np.array(D2),fos.green))
        
    fos.add(r,fos.line(ref,fos.golden))
    
    #uncinate2=[t+np.array([30,0,0]) for t in tracks]
    
    #fos.add(r,fos.line(uncinate2,fos.blue))

    fos.show(r)
    
def show_cut_color(hit,ref,bundle=None):
    
    r=fos.ren()
    fos.clear(r)
    
    #C=fos.colors(v,colormap='jet')  
    
    cnt=0
    for i in range(len(hit)):        
        
        #lh=len(hit[i][:,:3])
        
        #print lh#, len(C[i*cnt:i*cnt+lh])
        
        #fos.add(r,fos.point(hit[i][:,:3],C[cnt:cnt+lh] ))
        #fos.label(r,str(hit[i][:,3]),hit
        '''
        for h in hit[i]:
            #fos.label(r,str(np.round(100*h[3])),pos=(h[0],h[1],h[2]),scale=(0.05,0.05,0.05),color=(h[3],0,1-h[3]))
            fos.add(r,fos.dots(h[:3],color=(h[3],0,1-h[3])))
        '''
        #red=hit[i][:,3]
        #green=np.zeros(red.shape)
        #blue=1-red
        
        v=hit[i][:,3]
        red=np.interp(v,[0,0.35,0.66,0.89,1],[0,0,1,1,0.5])
        green=np.interp(v,[0,0.125,0.375,0.64,0.91,1],[0,0,1,1,0,0])
        blue=np.interp(v,[0,0.11,0.34,0.65,1],[0.5,1,1,0,0])
        
        colors=np.vstack((red,green,blue)).T
        fos.add(r,fos.point(hit[i][:,:3],colors,point_radius=0.05))
        
        #cnt=cnt+lh
        
    fos.add(r,fos.line(ref,fos.golden))
    
    if bundle!=None:
        fos.add(r,fos.line(bundle,fos.green,opacity=0.1))
        
    
    fos.show(r)

def all_hits_all_references(path):
    
    
    G,hdr,R=load_training_set(path)
    #Ga,hdr,Ra=load_approximate_training_set(path)
    
    tracks,vol_dims=load_tracks(path,1,1)
    tracksa=load_approximate_tracks(path,1,1)
    
    
    hits={}
    for g in [1,2,3,4,5,6,7,8]:        
        
        print 'Processing ', G[g]['label_name']
        
        ref=tracks[R[g]]
        tracksar,indicesr=tl.rm_far_tracks(ref,tracksa,dist=25.)
        hits[g]=pf.cut_plane(tracksar,ref)
    
    
    return hits,R
    
    
def all_bundles_references(path):
    
    #G,hdr=load_training_set(path)
    G,hdr=load_approximate_training_set(path)
    
    
    print 'Loading labeled bundles and indices'    
    '''
    1 arcuate  2 cingulum  3 corticospinal  4 forceps_major  5 fornix  6 inferior_occipitofrontal_fasciculus  
    7 superior_longitudinal_fasciculus  8 uncinate
    '''
    B={}
    Bi={}
    for g in [1,2,3,4,5,6,7,8]:
        
        B[g]=G[g]['tracks']
        Bi[g]=G[g]['indices']
    
    R={}
    print 'Find ref fibers'
    for g in [1,2,3,4,5,6,7,8]:
        si,s = pf.most_similar_track_mam(B[g])
        R[g]=Bi[g][si]
    
    return B,Bi,R
    
    
def save_approximate_tracks(path):
    ''' Save approximate tracks in PBC directory    
    '''
    
    # B1S1 is already saved
    
    print 'Saving at '+ path+'/B1S2a.pkl'   
    tracks,vol_dim=load_tracks(path,1,2)    
    tracksa=[tm.approximate_trajectory_partitioning(t) for t in tracks]    
    save_pickle(path+'/B1S2a.pkl',tracksa) 
    
    print 'Saving at '+ path+'/B2S1a.pkl'   
    tracks,vol_dim=load_tracks(path,2,1)    
    tracksa=[tm.approximate_trajectory_partitioning(t) for t in tracks]    
    save_pickle(path+'/B2S1a.pkl',tracksa)       
    
    # B2S2 is not provided yet
    
    print 'Saving at '+ path+'/B3S1a.pkl'   
    tracks,vol_dim=load_tracks(path,3,1)    
    tracksa=[tm.approximate_trajectory_partitioning(t) for t in tracks]    
    save_pickle(path+'/B3S1a.pkl',tracksa) 
    
    print 'Saving at '+ path+'/B3S2a.pkl'   
    tracks,vol_dim=load_tracks(path,3,2)    
    tracksa=[tm.approximate_trajectory_partitioning(t) for t in tracks]    
    save_pickle(path+'/B3S2a.pkl',tracksa) 

def load_approximate_tracks(path,brain,scan):
    ''' Load approximate tracks for specific brain and scan   
    '''
    
    return load_pickle(path+'/B'+str(brain)+'S'+str(scan)+'a.pkl')
    
def show_similarity_most_similar_track_all_training_set(G,option='avg'):
    ''' Show similarity between most similar tracks in the bundles of the training set
        Yellow tracks are the most similar tracks.
        
        Parameters:
        --------------

        option : string
                'avg' or 'min' or 'max'
        
    '''
    t1=time.clock()
    r=fos.ren()

    for g in [1,2,3,4,5,6,7,8]:
    #for g in [8]:
        
        #bundle=[tm.downsample(t,10) for t in G[g]['tracks']]
        bundle=G[g]['tracks']
        print 'len_bundle',len(bundle)
        #tno,s=tm.bundle_similarities_mam_fast(bundle)
        tno=G[g]['index_most_similar_avg']
        s=G[g]['similarity_avg']
        print 'No_tracks',len(bundle)
        print s.shape
        blue=np.interp(s,[s.min(),s.max()],[0,1])
        #blue=np.interp(s,[0,10],[0,1])
        green=np.zeros(blue.shape)
        red=blue[::-1]
        colormap=np.vstack((red,green,blue)).T 
        fos.add(r,fos.line(bundle,colormap))
        fos.add(r,fos.line(bundle[tno],np.array([1,1,0])))

    print 'time:',time.clock()-t1
    
    
    
    fos.show(r)    
        
    

def show_longest_tracks_and_bottlenecks_training_set(G):
    
    
    r=fos.ren()
    
    for g in [1,2,3,4,5,6,7,8]:
        
        print g,G[g]['label_name']
        
        bundle=G[g]['tracks']
               
        colors=np.random.rand(8,3)
                
        l=fos.line(bundle, colors=colors)        
        fos.add(r,l)
        
        all=[tm.length(t) for t in bundle]
        all=np.array(all)        
        ilongest=all.argmax()
        
        print g,ilongest

        l2=fos.line(bundle[ilongest],colors=np.array([1,0,0]))
        fos.add(r,l2)        

        #downsample first
        bundlez=[tm.downsample(t,10) for t in bundle]           
        
        bc,br=bottleneck(bundlez,0.98)       
        
        G[g]['big_ball_center']=bc
        G[g]['big_ball_radius']=br
        
        fos.add(r,fos.sphere(position=bc,radius=br,opacity=0.5))
        
        '''
        fbc,fbr=bottleneck_fit(bundle,bc,br) 
        
        G[g]['ball_center']=fbc
        G[g]['ball_radius']=fbr               

        fos.add(r,fos.sphere(position=fbc,radius=fbr,opacity=0.5))
        '''
        
    fos.show(r)



    

def bottleneck(bundle,ref=None,alpha=1):
    ''' Find the bottleneck of a bundle of tracks
    
    Notes:
    --------
    Description of the algorithm
    
    A. Find the bottleneck riding a fiber in the bundle {usually the longest or another fiber}.
    
    1. for each point along longest track in the bundle:
            for  each radius from the set of radii
                for each track in the bundle
                    calculate the number of intersecting tracks with the sphere
                    
    2. Find the spheres which had the most fibers intersecting them.
    3. From these spheres return only the one with the smallest radius.
       
    
    '''
    
    if ref==None:
        
        all=[tm.length(t) for t in bundle]
        all=np.array(all)        
        ilongest=all.argmax()    
                
    else:

        ilongest=ref        

    longest=bundle[ilongest]
    
    radii_set=np.linspace(2,20,50)
    
    lc=len(longest)
    lr=len(radii_set)
    
    CR=np.zeros((lc,lr))       
    
    ic=0        
    for c in longest:
        ir=0
        for r in radii_set:
            ts=[ tm.inside_sphere(t,c,r) for t in bundle]                
            CR[ic,ir]=ts.count(True)    
            ir+=1
          
        ic+=1    
        
    iCR=np.where(CR>=alpha*CR.max())

    iminr=iCR[1].argmin()

    bigsph_center=longest[iCR[0][iminr]]
    bigsph_radius=radii_set[iCR[1][iminr]]    
    
    return bigsph_center,bigsph_radius


    
def bottleneck_fit(bundle,bigsph_center=None,bigsph_radius=None):
    '''
    
    Notes :
    ---------
    for all points inside big sphere
        for all radii
            
    '''
    
    radii_set=np.linspace(2,20,50)
    
    if bigsph_center !=None and bigsph_radius != None:        
        
        centers=[ tm.inside_sphere_points(t,bigsph_center,bigsph_radius) for t in bundle]        
        centers=np.vstack(centers)
        
    else:
        
        centers =np.vstack(bundle)
    
    random_index=np.round(centers.shape[0]*np.random.rand(30)).astype(int)
    
    centers=centers[random_index]
        
    CR=np.zeros((centers.shape[0],len(radii_set)))        
    for i in range(centers.shape[0]):
        for r in radii_set:
            
            ts=[ tm.inside_sphere(t,centers[i],r) for t in bundle]                
            CR[i,r]=ts.count(True)    
    
    iCR=np.where(CR==CR.max())    
    iminr=iCR[1].argmin()
    
    sph_center=centers[iCR[0][iminr]]
    sph_radius=radii_set[iCR[1][iminr]]
    
    return sph_center,sph_radius

def test_mam_performance():
    
    path='/home/eg01/Data/PBC/pbc2009icdm'    
        
    G,hdr=load_training_set(path)
    for g in [1,2,3,4,5,6,7,8]:
        tracks=G[g]['tracks']

        #tracks, vol_dims=load_tracks(path,1,1)
        #print(pf.zhang(np.random.rand(100)))
        #t1=time.clock()
        #print(pf.zhang(tracks[0:10]))
        #print 'time',time.clock()-t1

        #pf.zhang(tracks[0:100])
        t1=time.clock()
        si,s=pf.most_similar_track_mam(tracks)
        print 'time',time.clock()-t1
        
        G[g]['index_most_similar']=si
        G[g]['similarity']=s

def zhang_performance(G,n=10,metric='avg'):
    ''' Downsample and calculate the most similar tracks in every bundle of the training set    
    '''
    
    for g in [1,2,3,4,5,6,7,8]:
        
        tracks=G[g]['tracks']        
        tracksz=[tm.downsample(t,n) for t in tracks]     
        
        t1=time.clock()
        si,s=pf.most_similar_track_mam(tracksz,'avg')
        print 'time',time.clock()-t1
        
        if metric=='avg':
            G[g]['index_most_similar_avg']=si
            G[g]['similarity_avg']=s
            
        if metric=='min':
            G[g]['index_most_similar_min']=si
            G[g]['similarity_min']=s
            
        if metric=='max':
            G[g]['index_most_similar_max']=si
            G[g]['similarity_max']=s
                

def reference_detection(ref,tracks): 
    ''' Detect the most similar track from tracks with the reference fiber in tracks.    
    '''   
    
    #rt=[tm.zhang_distances(ref,t,'avg') for t in tracks]
    rt=[pf.zhang_distances(ref,t,'avg') for t in tracks]
    
    #print(len(rt))
    
    return rt
    
def all_references_detection(G,tracks):
    '''
    
    Copyied Results
    Br1 S1 - Br1 S1
    >>> pbc.test_reference_detection(G,tracks)
    
    1 : 47148 47148
    
    2 : 15009 15009
    
    3 : 33939 33939
    
    4 : 88045 88045
    
    5 : 118191 118191
    
    6 : 19383 19383
    
    7 : 192033 192033
    
    8 : 187807 187807

    Br1 S1 - Br2 S1    
    >>> pbc.test_reference_detection(G,tracks2)
    
    1 : 47148 105906
    
    2 : 15009 61271
    
    3 : 33939 88274
    
    4 : 88045 91305
    
    5 : 118191 206474
    
    6 : 19383 237434
    
    7 : 192033 184828
    
    8 : 187807 3638

    Br1 S1 - Br3 S1     
    >>> pbc.test_reference_detection(G,tracks3)
    1 : 47148 147694
    
    2 : 15009 55511
    
    3 : 33939 179046
    
    4 : 88045 114472
    
    5 : 118191 58137
    
    6 : 19383 202141
    
    7 : 192033 212647
    
    8 : 187807 12772
    '''
    
    ref2ref=np.zeros((8,3))
    
    for g in [1,2,3,4,5,6,7,8]:
        
        ref=G[g]['tracks'][G[g]['index_most_similar_avg']]
        iref=G[g]['indices'][G[g]['index_most_similar_avg']]
        
        rt=reference_detection(ref,tracks)
        rt=np.array(rt)               
        #print 'Training',g,':',iref, rt.argmin()
        ref2ref[g-1]=np.array([g,iref,rt.argmin()])        
        
    return ref2ref

def angle_threshold(a,b,thr=1):
    ''' Return True if angle between vectors a and b is lower than threshold thr

    Parameters:
    ----------------
    a : array, shape(3,)
        3d vector
    b : array, shape(3,)
        3d vector    
    alpha : float
        angle threshold in radians were alpha=np.pi=3.14... means 180 degrees
    
    Returns :
    -----------
    {True,False}

    '''
    if a == None or b== None:
        return False
    
    an=np.sqrt(np.sum(a**2))
    bn=np.sqrt(np.sum(b**2))
    
    if np.arccos(np.inner(a,b)/(an*bn)) < thr:
        return True
    else:
        return False

def bottleneck_orientation(tracks,ref,alpha=1):
    ''' Find the bottleneck of a bundle of tracks having the same orientation as the reference track        
    
    Parameters :
    ----------------
    
    tracks : sequence
    
    tes : dict    
        The objects at each voxel are a list of integers, where the 
        integers are the indices of the track that passed through the voxel.
    
    ref : int
        index of reference fiber in tracks
        
    
    Returns :
    -----------

    center : sequence, (3,)
            center of sphere
    
    radius : float
            radius of sphere
    

    
    '''
    
    refer=tm.approximate_trajectory_partitioning(tracks[ref])
    print 'No points in refer', refer.shape[0]
    
    radii_set=np.arange(1,21) #np.linspace(2,20,50)
    
    lc=len(refer)
    lr=len(radii_set)
    
    CR=np.zeros((lc,lr))       

    print 'len tracks',len(tracks)
    
    ic=0            
    for c in refer:
        ir=0
        for r in radii_set:  
                        
            print c,r          
                        
            t1=time.clock()
            ref_or=tm.orientation_in_sphere(tracks[ref],c,r)    
            t2=time.clock()
            print 'Time',t2-t1          
            
            orient_tracks=[t for t in tracks if angle_threshold(ref_or,tm.orientation_in_sphere(t,c,r))]
            t3=time.clock()
            print 'Time',t3-t2,'len(orient_tracks)', len(orient_tracks)
            
            CR[ic,ir]=len(orient_tracks)
            ir+=1
          
        ic+=1    
        
    print 'CR.max', CR.max()
    
    iCR=np.where(CR>=alpha*CR.max())

    iminr=iCR[1].argmin()

    bigsph_center=refer[iCR[0][iminr]]
    bigsph_radius=radii_set[iCR[1][iminr]]    
    
    return bigsph_center,bigsph_radius,orient_tracks

def orient_tracks_sphere(tracks,ref,center,radius):
    
    ref_or=tm.orientation_in_sphere(tracks[ref],center,radius)    
    orient_tracks=[t for t in tracks if angle_threshold(ref_or,tm.orientation_in_sphere(t,center,radius))]
    
    return orient_tracks


def cubic_roi_tracks(tes,center,radius):
    ''' Return the track indices inside a cubic 
    region of interest with specific radius where radius 
    here is the 1/2 of the edge of the cube or the radius 
    of inscribed sphere in the cube.
    
    Parameters :
    ----------------
    center : sequence of 3 ints
    
    radius : int
        
    Returns :
    -----------
    tracks : sequence
                list of unique track indices intersecting this cubic region of interest
    
    '''
    
    roi=np.ndindex(radius*2,radius*2,radius*2)    
    
    center=np.array(center)    
    tracks=[]
    st=set(tracks)
    
    for vox in roi:
        
        nvox=np.array(vox)+center        
        
        try:                 
            st=st.union(set(tes[tuple(nvox)]))
        except KeyError:            
            pass
            
    #print len(tracks)
    return [t for t in st]

def show_all_cc(path):
    
    tracks11=load_approximate_tracks(path,1,1)        
    tracks12=load_approximate_tracks(path,1,2)        
    tracks21=load_approximate_tracks(path,2,1)       
    tracks31=load_approximate_tracks(path,3,1)       
    tracks32=load_approximate_tracks(path,3,2)        
    
    cc_indices11,rest_indices11=tl.detect_corpus_callosum(tracks11)
    cc_indices12,rest_indices12=tl.detect_corpus_callosum(tracks12)
    cc_indices21,rest_indices21=tl.detect_corpus_callosum(tracks21)
    cc_indices31,rest_indices31=tl.detect_corpus_callosum(tracks31)
    cc_indices32,rest_indices32=tl.detect_corpus_callosum(tracks32)
    
    '''
    cc_tracks11=[tracks11[i] for i in cc_indices11]
    cc_tracks12=[tracks12[i] for i in cc_indices12]
    cc_tracks21=[tracks21[i] for i in cc_indices21]
    cc_tracks31=[tracks31[i] for i in cc_indices31]
    cc_tracks32=[tracks32[i] for i in cc_indices32]
    '''
    cc_tracks11=[tracks11[i] for i in cc_indices11]
    cc_tracks12=[tracks12[i] + np.array([100,0,0]) for i in cc_indices12]
    cc_tracks21=[tracks21[i] + np.array([200,0,0]) for i in cc_indices21]
    cc_tracks31=[tracks31[i] + np.array([300,0,0]) for i in cc_indices31]
    cc_tracks32=[tracks32[i] + np.array([400,0,0]) for i in cc_indices32]

    
    
    rest_tracks11=[tracks11[i] for i in rest_indices11]
    rest_tracks12=[tracks12[i] + np.array([100,0,0]) for i in rest_indices12]
    rest_tracks21=[tracks21[i] + np.array([200,0,0]) for i in rest_indices21]
    rest_tracks31=[tracks31[i] + np.array([300,0,0]) for i in rest_indices31]
    rest_tracks32=[tracks32[i] + np.array([400,0,0]) for i in rest_indices32]        
    
    
    r=fos.ren()
    
    #show cc    
    fos.add(r,fos.line(cc_tracks11,fos.white,opacity=0.01))
    fos.add(r,fos.line(cc_tracks12,fos.gray,opacity=0.01))
    fos.add(r,fos.line(cc_tracks21,fos.red,opacity=0.01))
    fos.add(r,fos.line(cc_tracks31,fos.green,opacity=0.01))
    fos.add(r,fos.line(cc_tracks32,fos.dark_green,opacity=0.01))
    
    fos.show(r)
    
    #show the rest of the brain
    fos.clear(r)
    fos.add(r,fos.line(rest_tracks11,fos.white,opacity=0.01))
    fos.add(r,fos.line(rest_tracks12,fos.gray,opacity=0.01))
    fos.add(r,fos.line(rest_tracks21,fos.red,opacity=0.01))
    fos.add(r,fos.line(rest_tracks31,fos.green,opacity=0.01))
    fos.add(r,fos.line(rest_tracks32,fos.dark_green,opacity=0.01))
    
    fos.show(r)
    

def show_beauty(path):
    
    G,hdr,R=load_approximate_training_set(path)
    tracks11=load_approximate_tracks(path,1,1)

    #for g in G:
    #    print len(G[g]['indices'])
    
    '''
    1       Arcuate
    2       Cingulum
    3       Corticospinal
    4       Forceps Major
    5       Fornix
    6       Inferior Occipitofrontal Fasciculus
    7       Superior Longitudinal Fasciculus
    8       Uncinate
    '''
    
    '''
    red=np.array([1,0,0])
    green=np.array([0,1,0])
    blue=np.array([0,0,1])
    
    yellow=np.array([1,1,0])    
    golden=np.array([1,0.84,0])        
    
    dark_red=np.array([0.5,0,0])
    dark_green=np.array([0,0.5,0])
    dark_blue=np.array([0,0,0.5])
    gray=np.array([0.5,0.5,0.5])
    cyan=np.array([0,1,1])
    azure=np.array([0,0.49,1])
    white=np.array([1,1,1])
    
    aquamarine=np.array([0.498,1.,0.83])
    indigo=array([ 0.29411765,  0.,  0.50980392])
    lime=array([ 0.74901961,  1.,  0.])
    '''

    colors=[fos.white, fos.dark_red, fos.dark_green, fos.dark_blue, fos.gray,\
        fos.cyan, fos.azure, fos.hot_pink, fos.aquamarine, fos.indigo]
    
        
    r=fos.ren()    
    
    brain_indices=[]
    
    colors_bf=np.array([0,0,0])
    
    for g in G:
        
        indices=G[g]['indices']
        
        color_bf=np.tile(colors[g],(len(indices),1))        
        colors_bf=np.vstack((colors_bf,color_bf))
                        
        brain_indices=brain_indices+indices
        
    colors_bf=colors_bf[1:]

    Rv=R.values()
    
    print 'tracks no:',len(brain_indices)
    print 'colors no:',len(colors_bf)
    
    for (i,ind) in enumerate(brain_indices):
        
        if i in Rv :
                        
            colors_bf[i]=fos.golden
            
        
    brain=[tracks11[i] for i in brain_indices]    
   
        
    #l=fos.line(brain,colors_bf,opacity=0.1)
    
    for g in range(1,9):
        
        fos.add(r,fos.line(tracks11[R[g]],fos.golden))
    
    #fos.add(r,l)
    fos.show(r,track_bf=brain,color_bf=colors_bf)
    

def save_skeletal_tracks(path):
    
    tracks11=load_approximate_tracks(path,1,1)        
    tracks12=load_approximate_tracks(path,1,2)        
    tracks21=load_approximate_tracks(path,2,1)       
    tracks31=load_approximate_tracks(path,3,1)       
    tracks32=load_approximate_tracks(path,3,2)         
    
    no_tracks=5000
    
    print('skeletal11...')
    skeletal11=tl.skeletal_tracks(tracks11,rand_selected=no_tracks,ball_radius=5,neighb_no=50)
    print('saving 11...')
    save_pickle(path+'/skeletal_5000_11.pkl', skeletal11)
    
    print('skeletal12...')
    skeletal12=tl.skeletal_tracks(tracks12,rand_selected=no_tracks,ball_radius=5,neighb_no=50)
    print('saving 12...')
    save_pickle(path+'/skeletal_5000_12.pkl', skeletal12)

    print('skeletal21...')
    skeletal21=tl.skeletal_tracks(tracks21,rand_selected=no_tracks,ball_radius=5,neighb_no=50)    
    print('saving 21...')
    save_pickle(path+'/skeletal_5000_21.pkl', skeletal21)    

    print('skeletal31...')
    skeletal31=tl.skeletal_tracks(tracks31,rand_selected=no_tracks,ball_radius=5,neighb_no=50)    
    print('saving 31...')
    save_pickle(path+'/skeletal_5000_31.pkl', skeletal31)

    print('skeletal32...')
    skeletal32=tl.skeletal_tracks(tracks32,rand_selected=no_tracks,ball_radius=5,neighb_no=50)
    print('saving 32...')
    save_pickle(path+'/skeletal_5000_32.pkl', skeletal32)    


def show_train_skeletal(path):
    
    #skeletal=load_pickle(path+'/skeletal_5000_11.pkl')   
    tracks=load_approximate_tracks(path,1,1)        
    G,hdr,R=load_approximate_training_set(path)        
            
    colors=[fos.white,fos.dark_red, fos.dark_green, fos.dark_blue, fos.gray,\
        fos.cyan, fos.azure, fos.hot_pink, fos.aquamarine]

    labeled_indices=[]
    
    for g in range(1,9):
        
        labeled_indices.append( set(G[g]['indices']) )   
        
    training_set=set([])    
    
    for li in labeled_indices:
        
        training_set=training_set.union(li)    
    
    print('training_set',len(training_set))
    
    track_bf=[]
    color_bf=np.array([0,0,0])
    
    #training set
    for g in range(1,9):
        track_bf=track_bf+[i for i in labeled_indices[g-1]]    
        tmp_color=np.tile(colors[g], (len(labeled_indices[g-1]),1))
        color_bf=np.vstack((color_bf,tmp_color))
    
    color_bf=color_bf[1:]    
        
    #from indices to tracks
    ind_bf=track_bf
    track_bf=[tracks[i] for i in track_bf]
        
    r=fos.ren()    
    fos.show(r,track_bf=track_bf,ind_bf=ind_bf,color_bf=color_bf)

def show_skeletal(path):
    
    skeletal=load_pickle(path+'/skeletal_5000_11.pkl')   
    tracks=load_approximate_tracks(path,1,1)        
    G,hdr,R=load_approximate_training_set(path)        
            
    colors=[fos.white,fos.dark_red, fos.dark_green, fos.dark_blue, fos.gray,\
        fos.cyan, fos.azure, fos.hot_pink, fos.aquamarine]

    skeletal_set=set(skeletal)      
    print('skeletal_set',len(skeletal_set))
    
    golden_set=set(R.values())        
    print('golden_set',len(golden_set))
    
    non_golden_set=skeletal_set.difference(golden_set)
    print('non_golden_set',len(non_golden_set))
    
    if len(non_golden_set)!=len(skeletal_set):
        print('non_golden_set is smaller')
        return None
    
    labeled_indices=[]
    
    for g in range(1,9):
        
        labeled_indices.append( skeletal_set.intersection(set(G[g]['indices'])) )   
        
    training_set=set([])    
    
    for li in labeled_indices:
        
        training_set=training_set.union(li)    
    
    print('training_set',len(training_set))
    
    skeletal_non_training_set=skeletal_set.difference(training_set)
    
    print('skeletal_non_training_set',len(skeletal_non_training_set))
    
    hole_set=(skeletal_non_training_set.union(training_set)).union(golden_set)
    
    print('hole_set',len(hole_set))
    
    
    #golden fibers first
    golden=[i for i in golden_set]    
    
    color_bf=np.tile(fos.golden,(len(golden),1))
    track_bf=golden
    
    #training set
    for g in range(1,9):
        track_bf=track_bf+[i for i in labeled_indices[g-1]]    
        tmp_color=np.tile(colors[g], (len(labeled_indices[g-1]),1))
        color_bf=np.vstack((color_bf,tmp_color))
    
    #the rest
    #'''
    skeletal_non_training=[i for i in skeletal_non_training_set]      
    track_bf=track_bf+skeletal_non_training
    
    tmp_color=np.tile(fos.white, (len(skeletal_non_training),1))
    color_bf=np.vstack((color_bf,tmp_color))
    #'''
    
    #from indices to tracks
    ind_bf=track_bf
    track_bf=[tracks[i] for i in track_bf]
        
    r=fos.ren()    
    #add the warped atlas
    path_atlas=path+'/wICBM_WMPM_1_1.nii'
    
    atlas11,voxsz11,aff11=loadvol(path_atlas)
    
    fos.add(r,fos.volume(vol=atlas11,voxsz=voxsz11,affine=aff11))
    
    fos.show(r,track_bf=track_bf,ind_bf=ind_bf,color_bf=color_bf)
    
    #return color_bf, track_bf

def loadvol(filename):
    '''
    
    Load a volumetric array stored in a Nifti or analyze file with a specific filename. 
    It returns  an array with the data (arr), the voxel size (voxsz) and a transformation matrix (affine).

    This function is using volumeimages.
    
    Example:

    arr,voxsz,aff=loadvol('test.nii')
    
    '''

    if os.path.isfile(filename)==False:
        
        print('File does not exist')
        
        return [],[],[]

    img = vi.load(filename)      

    arr = np.array(img.get_data())
    voxsz = img.get_metadata().get_zooms()
    aff = img.get_affine()

    return arr,voxsz,aff

def make_track_volumes():
    
    '''
    Next stage we used spm with the following way
    
    Normalize-> Normal.(Est&Write)
        Data->Source-> Canonical-> c2single...
                    Images2write->ICBM_WMPM
        Estimation Options->Template-> counts_1_1
    
    Run and rename wICBM_WMPM to wICBM_WMPM_1_1
    
    In summary we transformed the ICBM atlas to the count space using the 
    MNI template as source and the counts as template.    
    
    '''
    path='/home/eg01/Data/PBC/pbc2009icdm'
    ds=pbc1109.PBCData(path)
    
    for subject in (1,2,3):
        for scan in (1,2):
            data = ds.get_data(subject, scan)
            if data.streams is None: # missing scan data
                break
            tracks = [s[0] for s in data.streams]
            vol_dim = data.hdr['dim']
            vox_sizes = data.hdr['voxel_size']
            
            img_hdr = vi.Nifti1Header()
            img_hdr.set_data_shape(vol_dim)
            img_hdr['pixdim'][:len(vox_sizes)] = vox_sizes
            affine = img_hdr.get_best_affine()
            # make affine as if for radiological image - correct?
            affine[0] = affine[0] * -1
            
            counts = tv.track_counts(tracks, vol_dim, vox_sizes, False)
            img = vi.Nifti1Image(counts > 0, affine, img_hdr)
            fname = 'counts_%d_%d.nii' % (subject, scan)
            fname=path+'/'+fname
            vi.save(img, fname)
            

def supervised_learning_test():
    
    path='/home/eg01/Data/PBC/pbc2009icdm'
    
    print('Loading tracks...')
    tracks11=load_approximate_tracks(path,1,1)    
    tracks11d=load_pickle(path+'/tracksd11.pkl')
    
    tracks11full,vol_dims=load_tracks(path,1,1)
    
    G,hdr,R=load_approximate_training_set(path)    
        
    emi=tl.emi_atlas()
    
    t11_indices=[]
    
    r=fos.ren()   
    
    for e in [1,2,3,4,5,6,7,8]:
    #for e in [1,4,8]:
            
        indices=emi[e]['apr_ref']+emi[e]['init_ref']+emi[e]['selected_ref']        
        print indices
        t11_indices.append(np.array(indices))               
    
        t11_t=[tracks11full[i] for i in t11_indices[e-1]]
        
        bundle=tl.bundle_from_refs(tracks11,tracks11d,t11_t,\
                    divergence_threshold=0.25, fibre_weight=0.8,far_thresh=25,zhang_thresh=25,end_thresh=5)
            
        real_bundle=G[e]['indices']
        
        real_bundlet=G[e]['tracks']
        
        print 'Label',e,G[e]['label_name']
        print 'difference real_b to b', len(set(real_bundle).difference(set(bundle)))
        print 'intersection real_b to b', len(set(real_bundle).intersection(set(bundle)))
        print 'size of real_b',len(set(real_bundle))
        print 'size of b',len(set(bundle))

        bundlet=[tracks11[i] for i in bundle]

        fos.add(r,fos.line(bundlet,fos.red))
        fos.add(r,fos.line(real_bundlet,fos.blue))    
        
        fos.add(r,fos.line(t11_t,fos.golden,linewidth=3))
        
        #bundlet2=[tracks11[i]+np.array([100,0,0]) for i in bundle]
        #fos.add(r,fos.line(bundlet2,fos.red))
    
    fos.show(r)


def create_corr():
    
    path='/home/eg01/Data/PBC/pbc2009icdm'
    
    print('Loading tracks...')
    tracks11=load_approximate_tracks(path,1,1)        
    tracks12=load_approximate_tracks(path,1,2)        
    tracks21=load_approximate_tracks(path,2,1)       
    tracks31=load_approximate_tracks(path,3,1)       
    tracks32=load_approximate_tracks(path,3,2)        
    
    emi=tl.emi_atlas()
    
    t11_indices=[]
    t12_indices=[]
    t21_indices=[]
    t31_indices=[]
    t32_indices=[]
    
    #for 8 bundles
    #for e in [1,2,3,4,5,6,7,8]:
    for e in range(1,21):
        
        indices=emi[e]['apr_ref']+emi[e]['init_ref']+emi[e]['selected_ref']        
        print indices
        #t11_indices.append(np.array(indices))
        
        tmp=tl.detect_corresponding_tracks(indices,tracks11,tracks12)
        
        t11_indices.append(tmp[:,1])
        t12_indices.append(tmp[:,2])
        t21_indices.append(tl.detect_corresponding_tracks(indices,tracks11,tracks21)[:,2])
        t31_indices.append(tl.detect_corresponding_tracks(indices,tracks11,tracks31)[:,2])
        t32_indices.append(tl.detect_corresponding_tracks(indices,tracks11,tracks32)[:,2])
    
    corr={0:t11_indices,1:t12_indices,2:t21_indices,3:t31_indices,4:t32_indices}
    
    print corr        
    save_pickle(path+'/corr_20.pkl',corr)

    return corr

def load_atlas_tes_and_tracks(path,brain,scan):
    
    volpath=path+'/ICBM_WMPM_tweaked_'+str(brain) +'_'+str(scan)+'.nii'    
    print volpath    
    template,voxsz,aff=loadvol(volpath)    
    tracks=load_approximate_tracks(path,brain,scan)
    print 'plus 3point tracks...'
    tracksd=load_pickle(path+'/tracksd'+str(brain)+str(scan)+'.pkl')
    
    print 'template shape', template.shape    
    tcs,tes = tv.track_counts(tracks, template.shape, vox_sizes=(1,1,1), return_elements=True)
    print 'tcs shape', tcs.shape    
    return template,tcs,tes,tracks,tracksd

def supervised_learning():
    
    #path='/home/eg01/Data/PBC/pbc2009icdm'
    
    path='/home/eg309/Data/PBC/pbc2009icdm'
        
    atlas,tcs,tes,tracks,tracksd=load_atlas_tes_and_tracks(path,1,1)
        
    atlas_tracks=tracks    
    
    #atlas,tcs,tes,tracks=load_atlas_tes_and_tracks(path,2,1)
    
    brains=[(1,1),(2,1),(3,1)]
    #brains=[(1,1)]
    
    for (b,s) in brains:
        
        atlas,tcs,tes,tracks,tracksd=load_atlas_tes_and_tracks(path,b,s)
    
        #brain=tl.relabel_by_atlas_value_and_mam(atlas_tracks,atlas,tes,tracks,tracksd,zhang_thr=[8,8,8,6,10,6,7,5])
        brain=tl.relabel_by_atlas_value_and_mam(atlas_tracks,atlas,tes,tracks,tracksd,zhang_thr=[9,6,8,5,9,7,6,4])
        
        #G,hdr,R=load_approximate_training_set(path)
        
        #fname='/home/eg01/eg309_brain'+str(b)+'_scan'+str(s)+'_ch1.txt'
        fname='/home/eg309/eg309_brain'+str(b)+'_scan'+str(s)+'_ch1.txt'
        
        f=open(fname,'wt')        
        subm=np.zeros((250000,2)).astype(int)    
        
        r=fos.ren()       
           
        for e in brain.keys():
            
            tracks_tmp=[tracks[ind] for ind in brain[e]['value_indices']]
            
            indices=brain[e]['value_indices']
            
            #indices_ts=G[e]['indices']
            
            #print brain[e]['name'],len(set(tracks_ts).difference(set(tracks_tmp))), len(set(tracks_ts).intersection(set(tracks_tmp)))    
                        
            for ind in indices:                
                subm[ind,0]=ind+1
                subm[ind,1]=e
            
            '''
            print G[e]['label_name'],\
                'diff',len(set(indices_ts).difference(set(indices))), \
                'diff',len(set(indices).difference(set(indices_ts))), \
                'int',len(set(indices_ts).intersection(set(indices))), \
                'ts',len(indices_ts),'ind',len(indices)                        
            '''
            
            fos.add(r,fos.line(tracks_tmp,np.array(brain[e]['color'])))
            fos.label(r,text=brain[e]['bundle_name'],pos=tracks_tmp[0][0],scale=(2,2,2))
        
        for i in range(subm.shape[0]):                        
                print >> f, '%d\t%d' % (i+1,subm[i,1])
            
        f.close()    
        fos.show(r)
    
    
    

def show_corr():
    
    path='/home/eg01/Data/PBC/pbc2009icdm'
    
    print('Loading tracks...')
    tracks11=load_approximate_tracks(path,1,1)        
    #tracks12=load_approximate_tracks(path,1,2) 
    
    tracks21=load_approximate_tracks(path,2,1)       
    
    tracks31=load_approximate_tracks(path,3,1)       
    #tracks32=load_approximate_tracks(path,3,2) 
    
    #all_tracks={0:tracks11,1:tracks12,2:tracks21,3:tracks31,4:tracks32}
    
    all_tracks={0:tracks11,1:tracks21,2:tracks31}
        
    #corr=load_pickle(path+'/corr_20.pkl')
    
    #corr=load_pickle(path+'/corr_8.pkl')
    
    corr=load_pickle(path+'/corr_8_sc1.pkl')
    
    r=fos.ren()      
    
    colors=[fos.red,fos.blue,fos.green,fos.cyan,fos.white,fos.azure,fos.coral,fos.indigo]
    
    for c in corr:        
        
        for (i,ref) in enumerate(corr[c]):
            
            print c,ref
            
            trk=[all_tracks[c][ind]+np.array([c*100,0,0]) for ind in ref]
            
            #print trk
                        
            #fos.add(r,fos.line(trk,fos.golden)) 
            fos.add(r,fos.line(trk,colors[i] )) 
            
    
    fos.show(r)
    

def supervised_learning_skeletal():
    
    path='/home/eg01/Data/PBC/pbc2009icdm'
    
    print('Loading skeletal tracks...')
    #tracks11=load_pickle(path+'/skeletal_5000_11.pkl')
    
    
    tracks11=load_approximate_tracks(path,1,1)            
    
    tracks12=load_approximate_tracks(path,1,2)        
    tracks21=load_approximate_tracks(path,2,1)       
    tracks31=load_approximate_tracks(path,3,1)       
    tracks32=load_approximate_tracks(path,3,2)   
    
    tracks11i=load_pickle(path+'/skeletal_5000_11.pkl')
    tracks12i=load_pickle(path+'/skeletal_5000_12.pkl')
    tracks21i=load_pickle(path+'/skeletal_5000_21.pkl')
    tracks31i=load_pickle(path+'/skeletal_5000_31.pkl')
    tracks32i=load_pickle(path+'/skeletal_5000_32.pkl')
    
    tracks11s=[tracks11[i] for i in tracks11i]
    tracks12s=[tracks12[i] for i in tracks12i]         
    tracks21s=[tracks21[i] for i in tracks21i]
    tracks31s=[tracks31[i] for i in tracks31i]
    tracks32s=[tracks32[i] for i in tracks32i]
    
    emi=tl.emi_atlas()
    
    t11_indices=[]
    t12_indices=[]
    t21_indices=[]
    t31_indices=[]
    t32_indices=[]
    
    #for 8 bundles
    for e in [1,2,3,4,5,6,7,8]:
        
        indices=emi[e]['apr_ref']+emi[e]['init_ref']+emi[e]['selected_ref']        
        print indices
        #t11_indices.append(np.array(indices))    
        
        tmp=tl.detect_corresponding_tracks_extended(indices,tracks11,tracks12i,tracks12s)
        
        t11_indices.append(tmp[:,1])
        t12_indices.append(tmp[:,2])
        t21_indices.append(tl.detect_corresponding_tracks_extended(indices,tracks11,tracks21i,tracks21s)[:,2])
        t31_indices.append(tl.detect_corresponding_tracks_extended(indices,tracks11,tracks31i,tracks31s)[:,2])
        t32_indices.append(tl.detect_corresponding_tracks_extended(indices,tracks11,tracks32i,tracks32s)[:,2])
    
    corr={0:t11_indices,1:t12_indices,2:t21_indices,3:t31_indices,4:t32_indices}
    
    print corr        
    save_pickle(path+'/corr_skeletal_8.pkl',corr)

    return corr

         

def all_bundles_from_refs(divergence_threshold=0.3, fibre_weight=0.7):
    emi=tl.emi_atlas()

    path='/home/eg01/Data/PBC/pbc2009icdm'
    tracks11=load_approximate_tracks(path,1,1)

    bundles=[]
    
    for e in [1,2,3,4,5,6,7,8]:
        
        ref_indices=emi[e]['apr_ref']+emi[e]['init_ref']+emi[e]['selected_ref']        
        t11_refs = [tracks11[i] for i in ref_indices]
        bundle, max_centres, max_indices = tl.bundle_from_refs(tracks11, t11_refs, divergence_threshold=0.3, fibre_weight=0.7)

        bundles.append((bundle, max_centres, max_indices, ref_indices))
        
    return bundles


def show_all_skeletal_brains_together():

    path='/home/eg01/Data/PBC/pbc2009icdm'
    
    print('Loading track indices...')
    tracks11i=load_pickle(path+'/skeletal_5000_11.pkl')
    tracks12i=load_pickle(path+'/skeletal_5000_12.pkl')
    tracks21i=load_pickle(path+'/skeletal_5000_21.pkl')
    tracks31i=load_pickle(path+'/skeletal_5000_31.pkl')
    tracks32i=load_pickle(path+'/skeletal_5000_32.pkl')
    
    print('Loading tracks...')
    tracks11=load_approximate_tracks(path,1,1)        
    tracks12=load_approximate_tracks(path,1,2)        
    tracks21=load_approximate_tracks(path,2,1)       
    tracks31=load_approximate_tracks(path,3,1)       
    tracks32=load_approximate_tracks(path,3,2)       

    print('Skeletal tracks...')
    tracks11=[tracks11[i] for i in tracks11i]
    tracks12=[tracks12[i] for i in tracks12i]
    tracks21=[tracks21[i] for i in tracks21i]
    tracks31=[tracks31[i] for i in tracks31i]
    tracks32=[tracks32[i] for i in tracks32i]
        
    r=fos.ren()    
    
    tracks12=[t+np.array([100,0,0]) for t in tracks12]
    tracks21=[t+np.array([200,0,0]) for t in tracks21]
    tracks31=[t+np.array([300,0,0]) for t in tracks31]
    tracks32=[t+np.array([400,0,0]) for t in tracks32]

    fos.add(r,fos.line(tracks11,fos.red,opacity=1))
    fos.add(r,fos.line(tracks12,fos.red,opacity=1))    
    fos.add(r,fos.line(tracks21,fos.red,opacity=1))
    fos.add(r,fos.line(tracks31,fos.red,opacity=1))
    fos.add(r,fos.line(tracks32,fos.red,opacity=1))
    
    fos.add(r,fos.axes(scale=(100,100,100)))
    fos.show(r)

    

def show_all_brains_together():
    path='/home/eg01/Data/PBC/pbc2009icdm'
    
    print('Loading tracks...')
    tracks11=load_approximate_tracks(path,1,1)        
    tracks12=load_approximate_tracks(path,1,2)        
    tracks21=load_approximate_tracks(path,2,1)       
    tracks31=load_approximate_tracks(path,3,1)       
    tracks32=load_approximate_tracks(path,3,2)       
    
    print('Load correspondance')    
    #corr=load_pickle(path+'/corr_8.pkl')
    corr=load_pickle(path+'/corr_skeletal_8.pkl')
    
    r=fos.ren()    
    
    tracks11_golden=list(corr[0][0])
    tracks12_golden=list(corr[1][0])
    tracks21_golden=list(corr[2][0])
    tracks31_golden=list(corr[3][0])
    tracks32_golden=list(corr[4][0])
    
    tracks11_golden=[tracks11[i] for i in tracks11_golden]
    tracks12_golden=[tracks12[i] +np.array([100,0,0]) for i in tracks12_golden]
    tracks21_golden=[tracks21[i] +np.array([200,0,0]) for i in tracks21_golden]
    tracks31_golden=[tracks31[i] +np.array([300,0,0]) for i in tracks31_golden]
    tracks32_golden=[tracks32[i] +np.array([400,0,0]) for i in tracks32_golden]
    
    tracks12=[t+np.array([100,0,0]) for t in tracks12]
    tracks21=[t+np.array([200,0,0]) for t in tracks21]
    tracks31=[t+np.array([300,0,0]) for t in tracks31]
    tracks32=[t+np.array([400,0,0]) for t in tracks32]
    

    fos.add(r,fos.line(tracks11,fos.red,opacity=0.01))
    fos.add(r,fos.line(tracks12,fos.red,opacity=0.01))    
    fos.add(r,fos.line(tracks21,fos.red,opacity=0.01))
    fos.add(r,fos.line(tracks31,fos.red,opacity=0.01))
    fos.add(r,fos.line(tracks32,fos.red,opacity=0.01))
    
    fos.add(r,fos.line(tracks11_golden,fos.golden,opacity=1,linewidth=3))
    fos.add(r,fos.line(tracks12_golden,fos.golden,opacity=1,linewidth=3))
    fos.add(r,fos.line(tracks21_golden,fos.golden,opacity=1,linewidth=3))
    fos.add(r,fos.line(tracks31_golden,fos.golden,opacity=1,linewidth=3))
    fos.add(r,fos.line(tracks32_golden,fos.golden,opacity=1,linewidth=3))
    
    fos.show(r)

def save_3point_tracks():
    
    path='/home/eg01/Data/PBC/pbc2009icdm'
    
    print('Loading tracks...')
    tracks11=load_approximate_tracks(path,1,1)        
    tracks12=load_approximate_tracks(path,1,2)        
    tracks21=load_approximate_tracks(path,2,1)       
    tracks31=load_approximate_tracks(path,3,1)       
    tracks32=load_approximate_tracks(path,3,2)    
    
    print('Downsampling to 3point tracks')
    
    tracksd11=[tm.downsample(t,3) for t in tracks11]
    tracksd12=[tm.downsample(t,3) for t in tracks12]
    tracksd21=[tm.downsample(t,3) for t in tracks21]
    tracksd31=[tm.downsample(t,3) for t in tracks31]
    tracksd32=[tm.downsample(t,3) for t in tracks32]
    
    print('Saving 3point tracks')
    save_pickle(path+'/tracksd11.pkl',tracksd11)
    save_pickle(path+'/tracksd12.pkl',tracksd12)
    save_pickle(path+'/tracksd21.pkl',tracksd21)
    save_pickle(path+'/tracksd31.pkl',tracksd31)
    save_pickle(path+'/tracksd32.pkl',tracksd32)
    


def analysis_summary_old():
    
    import pbc
    path='/home/eg01/Data/PBC/pbc2009icdm'
    
    print 'Load training set i.e. brain 1 scan 1 in G'
    G,hdr=pbc.load_training_set(path)

    print 'Downsample and calculate the most similar tracks (avg) '
    print 'in every bundle of the training set using zhang metrics.'
    print 'index of most similar track in every bundle is  now kept in G'    
    pbc.zhang_performance(G,n=10,metric='avg')       

    print 'Load tracks'
    tracks,   vol_dims=pbc.load_tracks(path,1,1)
    tracks2, vol_dims=pbc.load_tracks(path,2,1)
    tracks3, vol_dims=pbc.load_tracks(path,3,1)
    
    print 'Detect reference fibers from training set to the test set '
    ref1_ref2=all_references_detection(G,tracks2)    
    ref1_ref3=all_references_detection(G,tracks3)        
    
    ref1=ref1_ref2[:,1]
    ref2=ref1_ref2[:,2]
    ref3=ref1_ref3[:,2]
    
    return G,hdr,tracks,tracks2,tracks3,ref1.astype(int),ref2.astype(int),ref3.astype(int)
    
if __name__ == "__main__":
    
    #import cProfile
        
    #cProfile.run('test_mam_performance()', 'test_mam')
    
    path='/home/eg01/Data/PBC/pbc2009icdm'
    pathatlas='/backup/Data/ICBM_Atlas/ICBM_Wmpm'
    
    G,hdr,R=load_training_set(path)    
    
    arr,voxsz,aff,labels=load_atlas(pathatlas)
    
    show_atlas(arr,voxsz,aff,labels,G)    

    
