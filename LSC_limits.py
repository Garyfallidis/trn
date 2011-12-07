from time import time
import os
import numpy as np
import nibabel as nib
from itertools import combinations
from dipy.reconst.gqi import GeneralizedQSampling
from dipy.tracking.propagation import EuDX
from dipy.tracking.distances import local_skeleton_clustering
from dipy.tracking.metrics import downsample,length,winding
from dipy.tracking.vox2track import track_counts
from nibabel import trackvis as tv
import scipy.linalg as spl
from scipy.stats import describe
from dipy.io.dpy import Dpy
from dipy.io.pickles import load_pickle,save_pickle
import matplotlib.pyplot as plt
from dipy.tracking.distances import bundles_distances_mam,bundles_distances_mdf,most_similar_track_mam
from dipy.viz import fvtk
from fos import Window,World
from fos.actor.curve import InteractiveCurves
from scikits.learn import cluster
from scipy.optimize import fmin as fmin_simplex, fmin_powell, fmin_cg
from scipy.optimize import fmin_l_bfgs_b, anneal, brute, leastsq, fminbound
import matplotlib.pyplot as plt
from hcluster import pdist, linkage, dendrogram, squareform, fcluster
dname='/home/eg309/Data/PROC_MR10032/subj_08/101_32/'
fname='1312211075232351192010121409400346984008131ep2dadvdiffDSI10125x25x25STs004a001'
fbet=dname+fname+'_bet.nii.gz'
fbvals=dname+fname+'.bval'
fbvecs=dname+fname+'.bvec'
#dout='/home/eg309/Data/TMP_LSC_limits2/'
dout='/tmp/'

#seeds=[10**6,2*10**6,3*10**6,4*10**6,5*10**6,6*10**6,7*10**6,8*10**6,9*10**6,10*10**6]
seeds=[10**6]
#seeds=[10*10**6]
outs=[str(s) for s in seeds]

def generate_skeletons():

    img=nib.load(fbet)
    data=img.get_data()
    affine=img.get_affine()
    bvals=np.loadtxt(fbvals)
    bvecs=np.loadtxt(fbvecs).T
    t=time()
    gqs=GeneralizedQSampling(data,bvals,bvecs)
    print 'gqs time',time()-t,'s'
    for (i,sds) in enumerate(seeds):
        print i,sds
        t=time()
        eu=EuDX(gqs.qa(),gqs.ind(),seeds=sds,a_low=0.0239)
        #T=[downsample(e,12) for e in eu]
        T=[e for e in eu]
        np.save(dout+outs[i]+'.npy',np.array(T,dtype=np.object))
        
        #C=local_skeleton_clustering(T,4.)
        #save_pickle(dout+outs[i]+'.skl',C)
        print time()-t
        del T
    print outs
    
def generate_random_tracks(rand_no):

    img=nib.load(fbet)
    data=img.get_data()
    affine=img.get_affine()
    bvals=np.loadtxt(fbvals)
    bvecs=np.loadtxt(fbvecs).T
    t=time()
    gqs=GeneralizedQSampling(data,bvals,bvecs)
    print 'gqs time',time()-t,'s'
    for (i,sds) in enumerate(seeds):
        print i,sds
        t=time()
        eu=EuDX(gqs.qa(),gqs.ind(),seeds=sds,a_low=0.0239)
        T=[downsample(e,12) for e in eu]        
        np.save('/tmp/random_T.npy',np.array(T,dtype=np.object))
        ###################
        
        print time()-t
        del T
    print outs
    
def bring_virtuals(C):
    virtuals=[]
    cinds=[] # count indices
    track_no=0
    for c in C:
        v=C[c]['hidden']/float(C[c]['N'])
        virtuals.append(v)
        cinds.append(len(C[c]['indices']))
        track_no+=len(C[c]['indices'])
    return virtuals,cinds,float(track_no)

def bring_exemplars(C):
    
    exemplars=[]
    cinds=[] # count indices
    track_no=0
    for c in C:
        v=C[c]['most']
        exemplars.append(v.astype('f4'))
        cinds.append(len(C[c]['indices']))
        track_no+=len(C[c]['indices'])
    
    return exemplars,cinds,float(track_no)


def show_2_generations(C,thr=10.,cthr=0.000442707065162):
    r=fvtk.ren()
    vs,cinds,ls=bring_virtuals(C)
    lvs=len(vs)    
    vs=[vs[i] for (i,c) in enumerate(cinds) if c>=cthr] #c/ls>=cthr]    
    #fvtk.add(r,fvtk.line(vs,fvtk.red))
    C2=local_skeleton_clustering(vs,thr)
    vs2,inds2,ls2=bring_virtuals(C2)            
    #fvtk.add(r,fvtk.line(vs2,fvtk.green,opacity=0.5))
    print 'Initial',lvs,'Thresholded',len(vs),'Recalculated',len(vs2),'Total_Tracks',int(ls)
    #fvtk.show(r)
    return np.array([lvs,len(vs),len(vs2),int(ls)])

"""
C1=load_pickle(dout+outs[9]+'.skl')
v0,i0,l0=bring_virtuals(C0)
v1,i1,l1=bring_virtuals(C1)
d01=bundles_distances_mam(v0,v1)
print d01.shape
mins=np.min(d01,axis=1)
amins=np.argmin(d01,axis=1)
print amins.shape
"""
"""
res=[]
for out in outs:  
    C=load_pickle(dout+out+'.skl')
    res.append(show_2_generations(C,thr=4.,cthr=100)) #0.000442707065162))
    del C
res=np.array(res)
print res
"""

def check_bigger_clusters():

    avirtuals={}
    
    for (i,out) in enumerate(outs):
        C=load_pickle(dout+out+'.skl')
        cinds=np.zeros(len(C))
        for c in C:
            cinds[c]=len(C[c]['indices'])
        
        descend=np.argsort(cinds)[::-1]
        desc=descend[:400]
        virtuals=[]
        for c in desc:
            v=C[c]['hidden']/float(C[c]['N'])
            virtuals.append(v)        
        avirtuals[i]=virtuals
    
        
    r=fvtk.ren()
    fvtk.add(r,fvtk.line(avirtuals[0],fvtk.red))
    fvtk.add(r,fvtk.line(avirtuals[9],fvtk.yellow))
    fvtk.add(r,fvtk.line(avirtuals[5],fvtk.green))
    fvtk.show(r)


def converging_lsc(inp):

    C0=load_pickle(dout+outs[inp]+'.skl')
    print len(C0)
    v0,i0,l0=bring_virtuals(C0)
    v=v0
    #print len(v0)
    not_converged=1
    Cs=[]
    while not_converged:
        lv_before=len(v)    
        C=local_skeleton_clustering(v,4.)    
        v,i,l=bring_virtuals(C)
        #print '=',len(v)
        #for (i,v_) in enumerate(v):
        #    if length(v_)<50.:
        #        del v[i]
        #c=[v_ for v_ in v if length(v_)>50.]
        lv=len(v)
        #print lv
        if len(v)==lv_before:
            not_converged=0
        else:
            Cs.append(C)
    return Cs

def test_convergence():
    for i in range(10):
        print i+1,"M"
        C=converging_lsc(i)
        for c in C:
            print len(c)

def bundle_size_table(ls):
    tab = []
    for i in range(max(ls)+1):
        c=ls.count(i)
        if c>0:
            tab.append([i,c])
    return tab


def save_distances(vs1,vs2,filename='d12.npy'):
    print 'making distances ...'
    d12 = bundles_distances_mdf(vs1,vs2)
    #save_pickle('d12.skl',d12)
    np.save(filename,d12)
    print 'distances pickled ...'
    return d12


def cover_fraction(d12, ls1, ls2, dist, bsz1=1, bsz2=1):
    '''
    show what fraction of the tracks in one tractography 'belong' 
    to a skeleton track which is within mdf distance 'dist' of the other tractography  
    '''

    #find bundles of sizes bigger than bsz
    small1=[i for (i,l) in enumerate(ls1) if ls1[i]<bsz1]
    small2=[i for (i,l) in enumerate(ls2) if ls2[i]<bsz2]
    
    #print 'sh',d12.shape
    m12=d12.copy()
    if small2!=[]:
        m12[:,small2]=np.Inf
    if small1!=[]:
        m12[small1,:]=np.Inf
    #print 'sh',m12.shape    
    #find how many taleton-B neighbours nearer than dist
    near12 = np.sum(m12<=dist,axis=1)
    near21 = np.sum(m12<=dist,axis=0)

    #find their sizes    
    #sizes1 = [ls1[t] for t in np.where(np.sum(d12<=dist,axis=1)>0)[0]]
    sizes1 = [ls1[i] for i in range(len(near12)) if near12[i] > 0]
    #sizes2 = [ls2[t] for t in np.where(np.sum(d12<=dist,axis=0)>0)[0]]
    sizes2 = [ls2[i] for i in range(len(near21)) if near21[i] > 0]

    #calculate their coverage
    #total1 = np.float(np.sum(ls1))
    #total2 = np.float(np.sum(ls2))
    total1 = np.sum([ls1[t] for t in range(len(ls1)) if t not in small1])
    #total2 = np.float(np.sum(ls2))
    total2 = np.sum([ls2[t] for t in range(len(ls2)) if t not in small2])
    return np.sum(sizes1)/np.float(total1),np.sum(sizes2)/np.float(total2)

def nearest_fraction(d12,dist,ls1,ls2,bsz1=1, bsz2=1):
    
    great_than=0
    
    #find bundles of sizes bigger than bsz
    small1=[i for (i,l) in enumerate(ls1) if ls1[i]<bsz1]
    small2=[i for (i,l) in enumerate(ls2) if ls2[i]<bsz2]
    
    #print 'sh',d12.shape
    m12=d12.copy()
    if small2!=[]:
        m12[:,small2]=np.Inf
    if small1!=[]:
        m12[small1,:]=np.Inf
    
    near12 = np.sum(m12<=dist,axis=1)
    near21 = np.sum(m12<=dist,axis=0)
    res1=np.sum(near12>great_than)/np.float(len(near12))
    res2=np.sum(near21>great_than)/np.float(len(near21)) 

    return res1,res2

def equal_fraction(d12,dist,eq_to=1):
     
    near12 = np.sum(d12<=dist,axis=1)
    near21 = np.sum(d12<=dist,axis=0)
    res1=np.sum(near12==eq_to)/np.float(len(near12))
    res2=np.sum(near21==eq_to)/np.float(len(near21))
    
    return res1,res2

def taleton(T,dist=4*np.ones(1000)):
    
    Cs=[]
    id=0
    C=local_skeleton_clustering(T,dist[id])
    Cs.append(C)    
    vs=[C[c]['hidden']/float(C[c]['N']) for c in C]    
    ls=[len(C[c]['indices']) for c in C]
        
    vs_change=True
    id+=1
    while vs_change:
        lv_prev=len(vs)
        C=local_skeleton_clustering(vs,dist[id])
        vs=[C[c]['hidden']/float(C[c]['N']) for c in C]
        #ls=[len(C[c]['indices']) for c in C]
        for c in C:
            tmpi=C[c]['indices']
            tmpi2=[]
            
            tmph=np.zeros(C[c]['hidden'].shape)
            
            for i in tmpi:
                tmpi2+=Cs[id-1][i]['indices']
                tmph+=Cs[id-1][i]['hidden']
                
            C[c]['indices']=tmpi2
            C[c]['hidden']=tmph
            C[c]['N']=len(tmpi2)            
        
        if len(vs)!=lv_prev:
            Cs.append(C)            
            id+=1
        else:
            vs_change=False   
    return Cs


    

"""
 
#fn1='/home/ian/Data/LSC_stability/1000000.skl'
fn1='/home/eg309/Data/TMP_LSC_limits2/10000000.skl'
C1=load_pickle(fn1)
vs1,ls1,total1=bring_virtuals(C1)
#fn2='/home/ian/Data/LSC_stability/1000000_2.skl'
fn2='/home/eg309/Data/TMP_LSC_limits2/1000000_2.skl'
C2=load_pickle(fn2)
vs2,ls2,total2=bring_virtuals(C2)
#fn3='/home/ian/Data/LSC_stability/1000000_3.skl'
fn3='/home/eg309/Data/TMP_LSC_limits2/1000000_3.skl'
C3=load_pickle(fn3)
vs3,ls3,total3=bring_virtuals(C3)

d12 = bundles_distances_mdf(vs1,vs2).astype(np.float32)
print '#cf 12',cover_fraction(d12,ls1,ls2,1.,1,1)
print '#nf 12',nearest_fraction(d12,1.)
print '#ef 12',equal_fraction(d12,1.,1)
print d12.shape
del d12

d13 = bundles_distances_mdf(vs1,vs3).astype(np.float32)
print '#cf 13',cover_fraction(d13,ls1,ls3,1.,1,1)
print '#nf 13',nearest_fraction(d13,1.)
print '#ef 13',equal_fraction(d13,1.,1)
print d13.shape
del d13

d23 = bundles_distances_mdf(vs2,vs3).astype(np.float32)
print '#cf 23',cover_fraction(d23,ls2,ls3,1.,1,1)
print '#nf 23',nearest_fraction(d23,1.)
print '#ef 23',equal_fraction(d23,1.,1)
print d23.shape
del d23

fr='/home/eg309/Data/TMP_LSC_limits2/1000000_rand.npy'
rT=np.load(fr)
rv=rT[np.random.randint(0,len(rT),len(vs1))]
d1r = bundles_distances_mdf(vs1,rv).astype(np.float32)
print '#nf 1r',nearest_fraction(d1r,1.)
print '#ef 1r',equal_fraction(d1r,1.,1)

rv=rT[np.random.randint(0,len(rT),len(vs2))]
d2r = bundles_distances_mdf(vs2,rv).astype(np.float32)
print '#nf 2r',nearest_fraction(d2r,1.)
print '#ef 2r',equal_fraction(d2r,1.,1)

rv=rT[np.random.randint(0,len(rT),len(vs3))]
d3r = bundles_distances_mdf(vs3,rv).astype(np.float32)
print '#nf 3r',nearest_fraction(d3r,1.)
print '#ef 3r',equal_fraction(d3r,1.,1)

"""

def show_direct_stability(C1,C2,dist=2.,bt1=1,bt2=1):
    vs1,ls1,tot1=bring_virtuals(C1)
    vs2,ls2,tot2=bring_virtuals(C2)
    d12=bundles_distances_mdf(vs1,vs2)
    lv1=[length(v) for v in vs1]
    lv2=[length(v) for v in vs2]
    print "len vs1", len(vs1)
    print "len vs2", len(vs2)
    print "length1 min", np.min(lv1), " mean ", np.mean(lv1), " max ", np.max(lv1), " sum ", np.sum(lv1)
    print "length2 min", np.min(lv2), " mean ", np.mean(lv2), " max ", np.max(lv2), " sum ", np.sum(lv2)
    print "size1 min", np.min(ls1)," mean ",np.mean(ls1)," max ",np.max(ls1), " sum ", tot1
    print "size2 min", np.min(ls2)," mean ",np.mean(ls2)," max ",np.max(ls2), " sum ", tot2
    print "nf",nearest_fraction(d12,dist,ls1,ls2,bt1,bt2)
    print "ef",equal_fraction(d12,dist)
    print "cf",cover_fraction(d12,ls1,ls2,dist,bt1,bt2)
    #return d12

"""
for dist in [2.,4.]:
    
    bt1=bt2=1
    print '======== First Iteration ==============',dist,bt1,bt2
    show_coverage(C1s[0],C2s[0],dist,bt1,bt2)
    print '======== Last Iteration  ==============',dist,bt1,bt2
    show_coverage(C1s[-1],C2s[-1],dist,bt1,bt2)
    
    bt1=bt2=10
    print '======== First Iteration ==============',dist,bt1,bt2
    show_coverage(C1s[0],C2s[0],dist,bt1,bt2)
    print '======== Last Iteration  ==============',dist,bt1,bt2
    show_coverage(C1s[-1],C2s[-1],dist,bt1,bt2)
    
    bt1=bt2=20
    print '======== First Iteration ==============',dist,bt1,bt2
    show_coverage(C1s[0],C2s[0],dist,bt1,bt2)
    print '======== Last Iteration  ==============',dist,bt1,bt2
    show_coverage(C1s[-1],C2s[-1],dist,bt1,bt2)
"""

def show_one_by_one_ap(vs1,exemplars,labels,bt=20,one=True):

    vs1_=np.array(vs1,dtype=np.object)
    #c=InteractiveCurves(vs1,colors=np.zeros((len(vs1),4)),centered=False)
    w=World()
    #w.add(c)
    wi=Window()
    wi.attach(w)    
    for c in range(len(exemplars)):
        cols=np.random.rand(4,)
        cols[-1]=1
        inds=np.where(labels==c)[0]
        cols_=np.tile(cols,(len(inds),1))    
        if len(inds)>bt:            
            c=InteractiveCurves(list(vs1_[inds]),colors=cols_,centered=False)        
            w.add(c)            
            raw_input("Press Enter to continue...")            
            if one:
                w.delete(c)
                
def show_all_together_ap(vs1,exemplars,labels,bt=20,width=1000,height=1000):
    
    
    #c=InteractiveCurves(vs1,colors=np.zeros((len(vs1),4)),centered=False)
    w=World()
    #w.add(c)
    wi=Window(bgcolor=(1.,1.,1.,1.),width=width,height=height)
    wi.attach(w)    
    
    colors=np.random.rand(len(exemplars),4)
    colors[:,-1]=1           
    colors_=np.ones((len(vs1),4))
    for (i,l) in enumerate(labels):
        colors_[i]=colors[l]
                       
    c=InteractiveCurves(vs1,colors=colors_,centered=True)        
    w.add(c)            
            


def show_tracks(vs,width=1000,height=1000,line_width=3.,cmap='orient',color=None):
    
    from dipy.viz.colormap import orient2rgb,boys2rgb
    from dipy.tracking.metrics import midpoint     
    
    if cmap=='orient':
        cols=np.random.rand(len(vs),4)
        for (i,v) in enumerate(vs):
            cols[i,:3]=orient2rgb(v[0]-v[-1])
            
    if cmap=='boys':
        cols=np.random.rand(len(vs),4)
        for (i,v) in enumerate(vs):
            cols[i,:3]=boys2rgb(v[0]-v[-1])
            
    if cmap=='length':
        ls=[length(v) for v in vs]
        print np.min(ls),np.max(ls),np.mean(ls)
        col=np.interp(ls,[0,100,150],[0,0.5,1])
        cols=np.zeros((len(vs),4))
        cols[:,0]=col
        cols[:,-1]=col
            
    """
    if cmap=='mid':
        print 'Dimensions need correction for general case'
        cols=np.random.rand(len(vs),4)
        for (i,v) in enumerate(vs):
            mv=midpoint(v)
            nx=np.interp(mv[0],[0,95],[0.,1])
            ny=np.interp(mv[1],[0,95],[0.,1])
            nz=np.interp(mv[2],[0,55],[0.,1])
            cols[i,:3]=np.array([nx,ny,nz]).astype('f4')
    """
            
    if cmap=='random':
        cols=np.random.rand(len(vs),4)      
    
    if color!=None:
        cols=np.random.rand(len(vs),4)
        cols[:]=color 
    
    cols[:,-1]=1
    w=World()
    wi=Window(bgcolor=(1.,1.,1.,1.),width=width,height=height)
    wi.attach(w)
    c=InteractiveCurves(vs,colors=cols,centered=True,line_width=line_width)
    w.add(c)
    
def show_2_bundles(b1,b2,width=1000,height=1000,line_width=3.,col1=np.array([1,0,0,1],'f4'),col2=np.array([0,0,1,1],'f4')):    
    w=World()
    wi=Window(bgcolor=(1.,1.,1.,1.),width=width,height=height)
    wi.attach(w)
    cols1=np.zeros((len(b1),4)).astype('f4')
    #cols1[:,0]=1
    #cols1[:,-1]=1
    cols1[:]=col1
    cols2=np.zeros((len(b2),4)).astype('f4')
    #cols2[:,2]=1
    #cols2[:,-1]=1
    cols2[:]=col2
    c=InteractiveCurves(b1,colors=cols1,centered=False,line_width=line_width)
    w.add(c)
    c2=InteractiveCurves(b2,colors=cols2,centered=False,line_width=line_width)
    w.add(c2)
    
def show_tracks3(vs,width=1000,height=1000,line_width=3.):
       
    cols=np.random.rand(len(vs),4)
    
    cols[0,:3]=np.array([1,0,0])
    cols[1,:3]=np.array([0,1,0])
    cols[2,:3]=np.array([0,0,1])
    
    cols[:,-1]=1
    w=World()
    wi=Window(bgcolor=(1.,1.,1.,1.),width=width,height=height)
    wi.attach(w)
    c=InteractiveCurves(vs,colors=cols,centered=True,line_width=line_width)
    w.add(c)
    


    
    
def show_tracks_one_by_one(tracks,width=1000,height=1000):
    
    cols=np.array([[1,0,0,1.]]).astype('f4')    
    w=World()
    wi=Window(bgcolor=(1.,1.,1.,1.),width=width,height=height)
    wi.attach(w)
    for (i,t) in enumerate(tracks):
        c=InteractiveCurves([t],colors=cols,centered=True)
        w.add(c)
        #print i,ls[i]
        print "------>",i
        raw_input("Press Enter to continue...")
        w.delete(c)

def voxel_based_stability(C1,C2,shape=(96,96,55),vox=(2.5,2.5,2.5),absolute=True):

    vs1,ls1,tot1=bring_virtuals(C1)
    print len(vs1)
    vs2,ls2,tot2=bring_virtuals(C2)
    print len(vs2)
       
    tcs1=track_counts(vs1,shape,vox,False)
    tcs2=track_counts(vs2,shape,vox,False)
    if absolute:
        tcs=np.abs(tcs1-tcs2)
    else:
        tcs=tcs1-tcs2
    
    print tcs.min(), tcs1.min(), tcs2.min()
    print tcs.max(), tcs1.max(), tcs2.max()
    print tcs.sum(), tcs1.sum(), tcs2.sum()
    res1=1-np.abs(tcs.sum())/float(tcs1.sum())
    res2=1-np.abs(tcs.sum())/float(tcs2.sum())    
    return res1, res2



def plot_voxel_stability_across_different_distances(T1,T2):
    
    rest0=[]
    rest1=[]
    
    for i in [16.,15.,14.,13.,12.,11.,10.,9.,8.,7.,6.,5.,4.]:
        
        C1s=taleton(T1,10*[i])
        C2s=taleton(T2,10*[i])
        
        print '==========================='
        print i
        print '---------------------------'    
        res1,res2=voxel_based_stability(C1s[0],C2s[0])
        rest0.append(np.array([i,res1,res2]))
        print '---------------------------'
        res1,res2=voxel_based_stability(C1s[-1],C2s[-1])
        rest1.append(np.array([i,res1,res2]))
        
    rest0=np.array(rest0)
    rest1=np.array(rest1)
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    rx1=ax.plot(2.5*rest0[:,0],100*(rest0[:,1]+rest0[:,2])/2.,'b')
    rx2=ax.plot(2.5*rest1[:,0],100*(rest1[:,1]+rest1[:,2])/2.,'r')
    
    ax.set_xlabel('distance threshold (mm)')
    ax.set_ylabel('voxel counts difference %')
    ax.legend( (rx1[0],rx2[0]), ('First Iter', 'Last Iter') )
    
    plt.show()

def lens_weights(lens,low=10,high=100,low2=1,high2=10):
    weights=np.interp(lens,[low,high],[low2,high2])    
    return weights



    
def most_similar_tracks(T,pts=20,dist=4.,len_min=40,mam_type='avg'):
    skel=[]
    Td = [downsample(t,pts) for t in T]
    C=local_skeleton_clustering(Td,dist)
    vs,ls,tot=bring_virtuals(C)    
    nvs=[]
    nls=[]
    nind=[]
    for c in C:
        if C[c]['N']>len_min:
            bundle=[T[i] for i in C[c]['indices']]
            si,s=most_similar_track_mam(bundle,mam_type)    
            skel.append(bundle[si])
            nvs.append(vs[c])
            nls.append(ls[c])
            nind.append(C[c]['indices'])
            
    return skel,nvs,nls,nind



def ap_part(msim,ls):
    
    msim_=[downsample(m,20) for m in msim]
    D=bundles_distances_mam(msim_,msim_,'min')    
    #priors=lens_weights(ls[:-1],low=10,high=100,low2=1,high2=5)    
    S=-D
    centeris,labels=cluster.affinity_propagation(S,1*np.median(S))
    print len(centeris)    
    show_all_together_ap(msim,centeris,labels)    
    show_tracks(list(np.array(msim,dtype=np.object)[centeris]))

    CAP={}
    labels_=np.array(labels)
    
    for c in range(len(centeris)):
        CAP[c]={}
        CAP[c]['exemplar']=msim[centeris[c]]
        CAP[c]['indices']=list(np.where(labels_==c)[0])
    
    return CAP



def show_cap(CAP,C,T,one=False,wait=False,transp=False):
    
    w=World()
    wi=Window(bgcolor=(1.,1.,1.,1.),width=1000,height=1000)
    wi.attach(w)
    
    for ca in CAP:
        
        print "-------------CAP no--------->",ca
        print 'Higher level exemplar AP'
        #color=np.array([[1,0,0,1.]]).astype('f4')
        color=np.random.rand(1,4).astype('f4')  
        color[0,3]=1.
        c=InteractiveCurves([CAP[ca]['most']],colors=color,centered=False,line_width=3.)
        w.add(c)
        if wait:
            raw_input("Press Enter to continue...")
        
        cmostd=[]
        for ind in CAP[ca]['indices']:
            cmostd.append(C[ind]['most'])
                        
        colors=np.tile(color,(len(cmostd),1))
        
        #print 'Print exemplars LSC'
        c2=InteractiveCurves(cmostd,colors=colors,centered=False)
        w.add(c2)
        if wait:
            raw_input("Press Enter to continue...")
        
        #print 'Lower level'
        bundle=[]
        for ind in CAP[ca]['indices']:
            for inds2 in C[ind]['indices']:
                bundle.append(T[inds2]) 
        
        colors2=np.tile(color,(len(bundle),1))
        c3=InteractiveCurves(bundle,colors=colors2,centered=False)
        w.add(c3)
        if wait:
            raw_input("Press Enter to continue...")
        
        if one:
            w.delete(c)
            w.delete(c2)
            w.delete(c3)
            if transp:
                colors2[:,-1]=0.05
                c3=InteractiveCurves(bundle,colors=colors2,centered=False)
                w.add(c3)

            

def show_c(C,T,one=False,delete=True):
    
    w=World()
    wi=Window(bgcolor=(1.,1.,1.,1.),width=1000,height=1000)
    wi.attach(w)
    for c in C:
        
        print('Bundle Number ----> %d' % c)
        
        
        color=np.random.rand(4).astype('f4')
        color[-1]=.1
        
        bundle=[]        
        for ind in C[c]['indices']:
            bundle.append(T[ind])
            
        colors=np.tile(color,(len(bundle),1))
        ci=InteractiveCurves(bundle,colors=colors,line_width=2.,centered=False)
        w.add(ci)
        
        vcolor=np.array([[1,1,0,1.]]).astype('f4')        
        virtual=C[c]['hidden']/C[c]['N']
        virtuals=[virtual]          
        
        ci2=InteractiveCurves(curves=virtuals,colors=vcolor,line_width=6.,centered=False)
        w.add(ci2)
                
        vcolor2=np.array([[1,0,0,1.]]).astype('f4')        
        ci3=InteractiveCurves(curves=[C[c]['most']],colors=vcolor2,line_width=6.,centered=False)
        w.add(ci3)
                
        if one:
            raw_input('Press Enter to continue... ')
            w.delete(ci)
            if delete:
                w.delete(ci2)
                w.delete(ci3)
        
def closest_neighb_tcs(CAP,C,Td,CAP2,C2,Td2,shape=(96,96,55),vox=(2.5,2.5,2.5)):
    
    Td=np.array(Td,np.object)
    Td2=np.array(Td2,np.object)
    
    res=[]
    
    for ca in CAP:
        bundle=[]
        for ind in CAP[ca]['indices']:
            bundle+=list(Td[C[ind]['indices']])
        tcs=track_counts(bundle,shape,vox,False)
        m=tcs[tcs>0].mean()/2.
        tcsb=tcs>m
        
        min_diff=[]
        
        for ca2 in CAP2:
            bundle2=[]
            for ind2 in CAP2[ca2]['indices']:
                bundle2+=list(Td2[C2[ind2]['indices']])
            tcs2=track_counts(bundle2,shape,vox,False)
            m2=tcs2[tcs2>0].mean()/2.
            tcs2b=tcs2>m2
            #print ca, ca2, np.sum(np.abs(tcsb-tcs2b))
            #min_diff+=[ca,ca2,np.sum(np.abs(tcsb-tcs2b))/float(np.sum(np.bitwise_or(tcsb,tcs2b)))    ]
            min_diff+=[ca,ca2, np.sum(np.bitwise_xor(tcsb,tcs2b))/float(np.sum(np.bitwise_or(tcsb,tcs2b))) ]
            
        
        min_diff=np.array(min_diff)
        min_diff=min_diff.reshape(len(min_diff)/3,3)
        
        A=min_diff[np.argsort(min_diff[:,2])[:2]].ravel()
        np.set_printoptions(3,linewidth=150)
        #np.set_printoptions(precision, threshold, edgeitems, linewidth, suppress, nanstr, infstr)
        print A
        res.append(A)
            
        
        #raw_input("Press Enter to continue...")
    return np.array(res)
    
def draw_distance_graph(D):
    
    import networkx as nx

    G=nx.Graph()
    
    D2=np.tril(D.copy())
    
    for i in range(D.shape[0]):
        for j in range(D.shape[1]):
            if i!=j or D2[i,j]!=0:
                if D2[i,j]< 60.:
                    G.add_edge(i,j,weight=D[i,j])
    
    #G.add_edge('a','b',weight=0.6)
    #G.add_edge('a','c',weight=0.2)
    #G.add_edge('c','d',weight=0.1)
    #G.add_edge('c','e',weight=0.7)
    #G.add_edge('c','f',weight=0.9)
    #G.add_edge('a','d',weight=0.3)    
    
    elarge=[(u,v) for (u,v,d) in G.edges(data=True) if d['weight'] >30.]
    esmall=[(u,v) for (u,v,d) in G.edges(data=True) if d['weight'] <=30.]
    
    pos=nx.spring_layout(G) # positions for all nodes
    
    # nodes
    nx.draw_networkx_nodes(G,pos,node_size=len(D))
    #nx.draw_networkx_nodes(G,pos,node_size=700)
    
    # edges
    nx.draw_networkx_edges(G,pos,edgelist=elarge,
                        width=1)
    nx.draw_networkx_edges(G,pos,edgelist=esmall,
                        width=1,alpha=0.5,edge_color='b',style='dashed')
    
    # labels
    #nx.draw_networkx_labels(G,pos,font_size=20,font_family='sans-serif')

    plt.axis('off')
    plt.savefig("weighted_graph.png") # save as png
    plt.show() # display

    
def lsc(T,filt_short=40.,down_pts=12,lsc_dist=4.,min_csize=40):
    
    print('1.Initial tracks')
    print(len(T))
    
    print('2.Remove short tracks')
    T=np.array([t for t in list(T) if length(t)>= filt_short ],dtype=np.object) # 100mm
    print(len(T))
    
    print('3.Change ordering')
    iT=np.random.permutation(len(T))    
    T=list(T[iT])
    
    print('4.Reduce number of track points')
    Td = [downsample(t,down_pts) for t in T]
    print(len(Td))
    
    print('5.Run lsc')
    C=local_skeleton_clustering(Td,lsc_dist)
    print(len(C))
    
    print('6. find the nearest real track (lsc exemplars)')
    for c in C:        
        bundle=[Td[i] for i in C[c]['indices']]        
        D_=bundles_distances_mdf([C[c]['hidden']/float(C[c]['N'])],bundle)        
        ###########Check for degeneracies
        D_=D_.ravel()
        si=np.argmin(D_)
        #print D_.shape,si,D_
        C[c]['most']=bundle[si]
        #si,s=most_similar_track_mam(bundle,mam_type)
        #C[c]['most']=bundle[si]
    
    """    
    print('7.Remove clusters smaller than min_csize ')
    for c in range(len(C)):
        if C[c]['N']<min_csize:
            del C[c]
    print(len(C))
    
    print('8. Renumber dictionary - ascending int keys')
    C2={}
    keys=C.keys()
    for c in range(len(C)):
        C2[c]=C[keys[c]]
    C=C2
    """
    
    #print('8.Downsample lsc exemplars')
    #for c in C:
    #    C[c]['mostd']=downsample(C[c]['most'],down_pts2)  
    
    print('9.Calculate distances between lsc exemplars')
    lsc_exemplars=[C[c]['most'] for c in C]
    D=bundles_distances_mdf(lsc_exemplars,lsc_exemplars)
    
    return D,lsc_exemplars,C,Td

    
def ap(S,lsc_exemplars,prior):

    """
    if S.min()==-np.Inf:
        print '-Inf'
        print lsc_exemplars
        print lsc
        print 'D0'
        print D[0,:]
    """
           
    ap_exemplars,ap_labels=cluster.affinity_propagation(S,prior)
    #print( "-------------------> len ap %d" % len(ap_exemplars))
    
    #print('Create a new higher tree CAP') 
    CAP={}
    labels_=np.array(ap_labels)
    for c in range(len(ap_exemplars)):
        CAP[c]={}
        CAP[c]['most']=lsc_exemplars[ap_exemplars[c]]
        CAP[c]['indices']=list(np.where(labels_==c)[0])
        CAP[c]['N']=len(CAP[c]['indices'])
    #print('Done!')    
    return CAP

def lsc_small(T,lsc_dist):
    #Td = [downsample(t,down_pts) for t in T]
    C=local_skeleton_clustering(T,lsc_dist)
    return C

def brake_T(T,sub=10000):
    """ brake tractography to subsets    
    """
    
    bT=[] 
    step=0    
    while step <= len(T):        
        bT.append(T[step:step+sub])
        step=step+sub
    return bT 

def reorient2light(T,light=np.array([0,0,0])):
    
    T2=[]
    for t in T: 
        if np.sum((t[0]-light)**2) > np.sum((t[-1]-light)**2):
            T2.append(t[::-1])
        else:
            T2.append(t)
    return T2

def positivity_wins(T):
    """
    Nope!!!!
    """
    T2=[]    
    for t in T:
        if np.sum(np.diff(t[::-1],axis=0)) >  np.sum(np.diff(t,axis=0)>0):            
            T2.append(t[::-1])
        else:
            T2.append(t)
    return T2

def merge_Cs(Cs,thr=4.,update_indices=False):
    
    C0=Cs[0]
    
    lenC0=[]    
    lC0=len(C0)
    lenC0.append(lC0)
    
    for C in Cs[1:]:
        
        #print 'hey'        
        
        vs0,ls0,tot0=bring_virtuals(C0)
        vs1,ls1,tot1=bring_virtuals(C)
        
        ds=[]
        inds=[]
        
        #find closest
        for (i,v) in enumerate(vs1):
            d=bundles_distances_mdf([v],vs0)
            #print i, d.shape
            d=d.ravel()
            imd=np.argmin(d)
            
            ds.append(np.min(d))
            inds.append(imd)
        
        #update first cluster
        for i in range(len(ds)):
            
            d=ds[i]
            imd=inds[i]
                        
            if d<=thr:
                #print 'merge'
                if update_indices:
                    C0[imd]['indices']+=list(int(tot0)+np.array(C[i]['indices']))
                else:
                    C0[imd]['indices']+=C[i]['indices']
                C0[imd]['hidden']+=C[i]['hidden']
                C0[imd]['N']+=C[i]['N']
            else:
                #print 'new'                
                C0[lC0]=C[i]
                lC0+=1
    
        lenC0.append(lC0)
    
    return C0,lenC0

def parallel_lsc(Td,lsc_dist=4.,subset=10000):
    
    bT=brake_T(Td,subset)
    Cs=[]
    
    t1=time()
    for (i,b) in enumerate(bT):#[:3]:
        C=local_skeleton_clustering(b,lsc_dist)
        for c in C:
            C[c]['indices']=list(i*subset+np.array(C[c]['indices']))
        Cs.append(C)
    t2=time()
    print t2-t1
    
    print 'Cluster Blocks',len(Cs)
    
    t1=time()
    C0,lenC0=merge_Cs(Cs,thr=lsc_dist)
    t2=time()
    print t2-t1
    
    #print lenC0, len(C0)
    #vs,ls,tot=bring_virtuals(C0)
    #print np.sum(ls), len(Td)
    
    return C0,lenC0

def overlap_matrix(A,iA,B,iB):
    '''
    calculate intersection matrix of a pair
    of clusterings A and B. iA and iB are the labels for 
    the items in the clusterings.
    
    Parameters:
    -----------
    A: LSC holds list of track indices in bundles
    iA: original labels of tracks in A
    B: LSC holds list of (same) track indices in bundles
    iB: original labels of tracks in B
    '''
    print 'calculating overlap matrix ... '
    N = np.zeros((len(A),len(B)))
    print N.shape
    B_dic = {}
    for j in range(len(B)):
        for i in B[j]['indices']:
            B_dic[iB[i]] = j
    # B_dic is a dictionary in which we can look up the B-slass 
    # to which track has been assigned 
    for i in range(len(A)):
        for j in A[i]['indices']:
            #try:
            N[i,B_dic[iA[j]]] += 1
            #except KeyError:
            #    pass
    
    return N

def intersection_matrix(A,iA,B,iB):    
    """ Exactly the same as overlap matrix but slower 
    """
    
    A2={}
    B2={}    
    for a in A:
        A2[a]={}
        A2[a]['indices']=iA[A[a]['indices']]
        
    for b in B:
        B2[b]={}
        B2[b]['indices']=iB[B[b]['indices']]
    
    I=np.zeros((len(A2),len(B2)))
    for i in range(len(A2)):
        for j in range(len(B2)):
            I[i,j]=len(set(A2[i]['indices']).intersection(B2[j]['indices']))        
    return I
    
def compare_classifications(N):
    '''
    calculate various measures for an overlap matrix
    
    Incorporating:  Gini Impurities, Classification Errorsm
    Information, Disagreements

    Parameters:
    ===========
    N: an (m,n) matrix of overlap counts
    
    Output:
    =======
    gini_stats: ...
    class_err_stats: ...
    '''
    
    A_tot = np.sum(N,axis=1,dtype=float)
    B_tot = np.sum(N,axis=0,dtype=float)
    N_tot = np.sum(A_tot)
    
    #print np.min(A_tot), np.max(A_tot), np.min(B_tot), np.max(B_tot)
    
    gini_A = np.sum((1-np.sum(N**2,axis=1)/A_tot**2)*A_tot)/N_tot
    gini_B = np.sum((1-np.sum(N**2,axis=0)/B_tot**2)*B_tot)/N_tot
    
    #info_A_given_B = np.log2(A_tot)-np.sum(N*np.log2(np.maximum(N,1)),axis=1)
    #info_B_given_A = np.log2(B_tot)-np.sum(N*np.log2(np.maximum(N,1)),axis=0)

    random_class_error_A = np.sum(A_tot-np.max(N,axis=1))/N_tot
    random_class_error_B = np.sum(B_tot-np.max(N,axis=0))/N_tot

    print 'calculating disagreements ... '
    n_pairs = N_tot*(N_tot-1)/2.
    #print 'n_pairs', n_pairs
    togetherA = np.sum(A_tot*(A_tot-1)/2.)
    apartA = n_pairs-togetherA
    splitA = np.sum(np.cumsum(N,axis=1)[:,:-1]*N[:,1:])
    joinA = np.sum(np.cumsum(N,axis=0)[:-1,:]*N[1:,:])
    completenessA = 1 - splitA / togetherA
    correctnessA = 1 - joinA / apartA
                                       

    togetherB = np.sum(B_tot*(B_tot-1)/2.)
    apartB = n_pairs-togetherB
    splitB = joinA
    joinB = splitA
    completenessB = 1 - splitB / togetherB
    correctnessB = 1 - joinB / apartB
    
    discord = splitA+joinA
    
#    d=np.sum(np.cumsum(N,axis=1)[:-1,:]*N[1:,:])+np.sum(np.cumsum(N,axis=0)[:,:-1]*N[:,1:])
    #print 'discordant pairs', d
    p_disagree = discord/n_pairs   
                                       

    return gini_A, gini_B, random_class_error_A, random_class_error_B, p_disagree, correctnessA, completenessA, correctnessB, completenessB


def lsc_exemplars(C,Td):

    for c in C:        
        bundle=[Td[i] for i in C[c]['indices']]        
        D_=bundles_distances_mdf([C[c]['hidden']/float(C[c]['N'])],bundle)        
        ###########Check for degeneracies
        D_=D_.ravel()
        si=np.argmin(D_)
        #print D_.shape,si,D_
        C[c]['most']=bundle[si]
        
    lsc_exemplars=[C[c]['most'] for c in C]
    #D=bundles_distances_mdf(lsc_exemplars,lsc_exemplars)
    return lsc_exemplars

def remove_small_bundles(C,se=1):
    for c in range(len(C)):
        if C[c]['N']<=se:
            del C[c]
    #print(len(C))
    C2={}
    keys=C.keys()
    for c in range(len(C)):
        C2[c]=C[keys[c]]
    C=C2
    return C

def remove_bundles_winding(C,degrees=400):
    for c in range(len(C)):
        if winding(C[c]['most'])>degrees:
            del C[c]
    #print(len(C))
    C2={}
    keys=C.keys()
    for c in range(len(C)):
        C2[c]=C[keys[c]]
    C=C2
    return C



def compare(C0,iTd0,C1,iTd1,dist=2.):

    g01, g10, rce1, rce2, dis01, cmpl1,corr1,cmpl2,corr2 = compare_classifications(overlap_matrix(C0,iTd0,C1,iTd1))
    
    print 'Gini impurity :  ', g01, g10    
    print 'Random classification error : ', rce1, rce2    
    print 'Pair classification discordancy: ', dis01 
    print 'Completeness 1 and correctness 1: ', cmpl1, corr1
    print 'Completeness 1 and correctness 2: ', cmpl2, corr2

    #vs0,ls0,tot0 = bring_virtuals(C0)
    #vs1,ls1,tot1 = bring_virtuals(C1)
        
    #print 'Nearest fraction ', nearest_fraction(bundles_distances_mdf(vs0,vs1),dist,ls0,ls1)
    #print 'Cover fraction ', cover_fraction(bundles_distances_mdf(vs0,vs1),ls0,ls1,dist)

def exemplar_segmentation(Td,dist=4.):

    C,lenC=parallel_lsc(Td,lsc_dist=dist,subset=10000)
    print 'plsc',len(C)
    
    C=remove_small_bundles(C,10)
    print 'rsb',len(C)
    
    lscex=lsc_exemplars(C,Td)    

    C=remove_bundles_winding(C,400)
    print 'rbw',len(C)
    
    lscex,lscexls,lscextot=bring_exemplars(C)
    D=bundles_distances_mdf(lscex,lscex)        
    S=-D
    
    med=np.median(S)
    #pbig=2*med
    #psmall=.5*med
    pbig=.5*med
    psmall=2*med
    
    nmin=np.min(lscexls)
    nmax=np.max(lscexls)
    
    p=((lscexls-nmin)/(nmax-nmin))*(psmall - pbig) + pbig 
        
    CAP=ap(S,lscex,prior=p)               
    #ex0,exls0,extot0=bring_exemplars(CAP0)
    return CAP,C,D

def nill_class(C,lenTd):    
    C2=C.copy()
    inds=[]
    for c in C:
        inds+=C[c]['indices']                
    nill=list( set(range(lenTd)).difference(inds) )    
    lc=len(C)    
    C2[lc]={}
    C2[lc]['indices']=nill    
    return C2

def gather_indices(CAP,C):
    CAPF={}
    for ca in CAP:
        CAPF[ca]={}
        CAPF[ca]['indices']=[]
        for c in CAP[ca]['indices']:
            CAPF[ca]['indices']+=C[c]['indices'] 
    return CAPF

def tight_comparison(ex0,ex1,dist):

    d01=bundles_distances_mdf(ex0,ex1)    
    pair12=[]
    solo1=[]
    for i in range(len(ex0)):
        if np.min(d01[i,:]) < dist:
            j=np.argmin(d01[i,:])
            pair12.append((i,j))
        else:            
            solo1.append(ex0[i])
    pair12=np.array(pair12)
    
    pair21=[]
    solo2=[]
    for i in range(len(ex1)):
        if np.min(d01[:,i]) < dist:
            j=np.argmin(d01[:,i])
            pair21.append((i,j))
        else:
            solo2.append(ex1[i])
            
    pair21=np.array(pair21)
    
    b0=[]
    b1=[]
    for p in pair12:
        b0+=[ex0[p[0]]]
        b1+=[ex1[p[1]]]
    
    b2=[]
    b3=[]
    for p in pair21:
        b2+=[ex1[p[0]]]
        b3+=[ex0[p[1]]]
    
    return b0,b1,b2,b3,solo1,solo2

def length_limits(t,low,high):
    lt=length(t)
    if lt >=low and lt<high:
        return 1
    else:
        return 0 
    
def change_dtype(T,type='f4'):
    return [t.astype(type) for t in T]

def compare_2_orderings(T):
    T=reorient2light(T)
    T=[t for t in T if length_limits(t,40.,120.)]
    Td = [downsample(t,12) for t in T]
    Td = np.array(Td,dtype=np.object)
    
    iTd0=np.random.permutation(len(Td))
    Td0=change_dtype(Td[iTd0])
    CAP0,C0,D0=exemplar_segmentation(Td0)
    
    iTd1=np.random.permutation(len(Td))
    Td1=change_dtype(Td[iTd1])
    CAP1,C1,D1=exemplar_segmentation(Td1)
    
    """
    print 'Parallel LSC With Removal'
    C0_=nill_class(C0,len(Td))
    C1_=nill_class(C1,len(Td))
    compare(C0_,iTd0,C1_,iTd1)
    
    print 'Parallel LSC No Removal'
    C0p,lenC0p=parallel_lsc(Td0)
    C1p,lenC1p=parallel_lsc(Td1)
    compare(C0p,iTd0,C1p,iTd1)
    
    print  'Standard LSC No Removal'
    C0l=local_skeleton_clustering(Td0,4.)
    C1l=local_skeleton_clustering(Td1,4.)
    compare(C0l,iTd0,C1l,iTd1)
    
    print 'LSC+AP with Removal'
    CAP0g=gather_indices(CAP0,C0)
    CAP0g2=nill_class(CAP0g,len(Td))
    
    CAP1g=gather_indices(CAP1,C1)
    CAP1g2=nill_class(CAP1g,len(Td))
    """
    
    
    print 'Landmarks'
    
    big =[C0[c]['most'].astype('f4') for c in C0 if C0[c]['N']> 100]
    big2=[C1[c]['most'].astype('f4') for c in C1 if C1[c]['N']> 100]
    
    b0,b1,b2,b3,sol0,sol1=tight_comparison(big,big2,2.)
    
    print len(b0)/float(len(big)),len(b1)/float(len(big)), len(b2)/float(len(big2)), len(b3)/float(len(big2))
    
    
    
    """
    print 'Overlap matrix comparisons'
    
    compare(CAP0g2,iTd0,CAP1g2,iTd1)
    
    print 'Direct comparisons'
    
    ex0,exls0,tot0=bring_exemplars(CAP0)
    ex1,exls1,tot1=bring_exemplars(CAP1)
    
    d01=bundles_distances_mdf(ex0,ex1)
    
    dist=2.
    
    nearest_fraction(d01,dist,exls0,exls1)
    cover_fraction(d01,exls0,exls1,dist)
    near12 = np.sum(d01<=dist,axis=1)
    near21 = np.sum(d01<=dist,axis=0)
    """
    
def plot_timings():
    
    #dir='/home/eg309/Data/LSC_limits/full_1M'
    dir='/tmp/full_1M'
    fs=['.npy','_2.npy','_3.npy','_4.npy','_5.npy']#,'_6.npy','_7.npy','_8.npy','_9.npy','_10.npy']
    
    T=[]
    for f in fs:
        fs1=dir+f
        T+=change_dtype(list(np.load(fs1)))
    #return T
    #T=T[:1000]
    
    print len(T)    
    dists=[4.,6.,8.,10.]
    pts=[3,6,12,18]        
    sub=10**5
    #sub=10**2    
    res={}            
    for p in pts:
        print p
        res[p]={}
        for d in dists:
            print d
            res[p][d]={}
            res[p][d]['time']=[]
            res[p][d]['len']=[]
            step=0
            while step <= len(T):
                print step
                Td=[downsample(t,p) for t in T[0:step+sub]]
                t1=time()
                C=local_skeleton_clustering(Td,d)
                t2=time()
                res[p][d]['time'].append(t2-t1)
                res[p][d]['len'].append(len(C))       
                step=step+sub
    
    #return res
    save_pickle('/tmp/res.pkl',res)
    

       

def compare_tightness():
    
    dir='/home/eg309/Data/LSC_limits/full_1M' 
    fs=['.npy','_2.npy','_3.npy','_4.npy','_5.npy','_6.npy','_7.npy','_8.npy','_9.npy','_10.npy']

    res=[]
    res2=[]
    
    bs=[]
    
    for c in combinations(range(10),2):
    
        fs1=dir+fs[c[0]]
        fs2=dir+fs[c[1]] 

        #fsolid ='/home/eg309/Data/LSC_limits/full_1M.npy'
        #fsolid2='/home/eg309/Data/LSC_limits/full_1M_2.npy'
        #fsolid3='/home/eg309/Data/LSC_limits/full_1M_3.npy'
        #fsolid='/home/ian/Data/LSC_stability/solid_1M.npy'
        print fs1
        print fs2
                
        T=list(np.load(fs1))
        T=reorient2light(T)
        T=[t for t in T if length_limits(t,40.,120.)]
        Td = [downsample(t,12) for t in T]
        Td = np.array(Td,dtype=np.object)
        
        iTd0=np.random.permutation(len(Td))
        Td0=change_dtype(Td[iTd0])
        CAP0,C0,D0=exemplar_segmentation(Td0)
        
        T1=list(np.load(fs2))
        T1=reorient2light(T1)
        T1=[t for t in T1 if length_limits(t,40.,120.)]
        Td1 = [downsample(t,12) for t in T1]
        Td1 = np.array(Td1,dtype=np.object)
        
        iTd1=np.random.permutation(len(Td1))
        Td1=change_dtype(Td1[iTd1])
        CAP1,C1,D1=exemplar_segmentation(Td1)
        
        #print 'Landmarks'
        big =[C0[c]['most'].astype('f4') for c in C0 if C0[c]['N']> 100]
        big2=[C1[c]['most'].astype('f4') for c in C1 if C1[c]['N']> 100]
        
        b0,b1,b2,b3,sol0,sol1=tight_comparison(big,big2,2.)
        print len(b0)/float(len(big)),len(b1)/float(len(big)), len(b2)/float(len(big2)), len(b3)/float(len(big2))
        res.append((len(b0)/float(len(big)),len(b1)/float(len(big)), len(b2)/float(len(big2)), len(b3)/float(len(big2))))

        bs.append((b0,b1,b2,b3,sol0,sol1))
        
        big =[C0[c]['most'].astype('f4') for c in C0 if C0[c]['N']> 0]
        big2=[C1[c]['most'].astype('f4') for c in C1 if C1[c]['N']> 0]
                
        b0,b1,b2,b3,sol0,sol1=tight_comparison(big,big2,2.)
        print len(b0)/float(len(big)),len(b1)/float(len(big)), len(b2)/float(len(big2)), len(b3)/float(len(big2))
        res2.append((len(b0)/float(len(big)),len(b1)/float(len(big)), len(b2)/float(len(big2)), len(b3)/float(len(big2))))



    return res,res2,bs


def compare_tightness_real_data():    
    
    lfiles=[]
    
    dname='/home/eg309/Data/PROC_MR10032/'
    for root, dirs, files in os.walk(dname):        
        if root.endswith('101_32/GQI'):
            #print root
            for file in files:
                if file.endswith('lsc_QA_ref.dpy'):
                    #print file
                    lfiles.append(root+'/'+file)
    
    #for f in lfiles:
    #    print f
    
    res=[]
    res2=[]    
    bs=[]
    lens=[]
    
    cnt=0
    
    for c in combinations(range(10),2):
    
        fs1=lfiles[c[0]]
        fs2=lfiles[c[1]] 
    
        print 'Combination number',cnt
        cnt+=1
        print fs1
        print fs2
              
        
        dpr=Dpy(fs1,'r')        
        T=dpr.read_tracks()
        dpr.close()        
        print 'lenT',len(T)
        
        T=reorient2light(T,(100,100,100))
        T=[t for t in T if length_limits(t,40.*2.5,120.*2.5)]
        Td = [downsample(t,12) for t in T]
        Td = np.array(Td,dtype=np.object)
        
        iTd0=np.random.permutation(len(Td))
        Td0=change_dtype(Td[iTd0])
        print 'lenTd0',len(Td0)
        CAP0,C0,D0=exemplar_segmentation(Td0,4.*2.5)
        print 'lenC0',len(C0)
        
        dpr=Dpy(fs2,'r')        
        T1=dpr.read_tracks()
        dpr.close()
        print 'lenT1',len(T1)
        
        T1=reorient2light(T1,(100,100,100))
        T1=[t for t in T1 if length_limits(t,40.*2.5,120.*2.5)]
        Td1 = [downsample(t,12) for t in T1]
        Td1 = np.array(Td1,dtype=np.object)
        print 'lenTd1',len(Td1)
        
        iTd1=np.random.permutation(len(Td1))
        Td1=change_dtype(Td1[iTd1])
        CAP1,C1,D1=exemplar_segmentation(Td1,4.*2.5)
        print 'lenC1',len(C0)
        
        #"""
        #print 'Landmarks'
        big =[C0[c]['most'].astype('f4') for c in C0 if C0[c]['N']> 10]
        big2=[C1[c]['most'].astype('f4') for c in C1 if C1[c]['N']> 10]
            
        print 'len big',len(big),'len big2',len(big2)
        
        b0,b1,b2,b3,sol0,sol1=tight_comparison(big,big2,4.*2.5)
        print len(b0)/float(len(big)),len(b1)/float(len(big)), len(b2)/float(len(big2)), len(b3)/float(len(big2))
        
        lens.append((len(T),len(T1),len(C0),len(C1),len(big),len(big2)))
        res.append((len(b0)/float(len(big)),len(b1)/float(len(big)), len(b2)/float(len(big2)), len(b3)/float(len(big2))))
        bs.append((b0,b1,b2,b3,sol0,sol1))
        
    Lens=np.array(lens)
    Res=np.array(res)
    
    save_pickle('/tmp/Bs_10mm_bt10_.pkl',bs)
    save_pickle('/tmp/Lens_10mm_bt10_.pkl',Lens)   
    save_pickle('/tmp/Res_10mm_bt10_.pkl',Res)



def atlas_creation():
    
    lfiles=[]
    
    dname='/home/eg309/Data/PROC_MR10032/'
    for root, dirs, files in os.walk(dname):        
        if root.endswith('101_32/GQI'):
            #print root
            for file in files:
                if file.endswith('lsc_QA_ref.dpy'):
                    #print file
                    lfiles.append(root+'/'+file) 
    
    t1=time()
    
    Cs=[]
    for f in lfiles:
        print f
    
        dpr=Dpy(f,'r')        
        T=dpr.read_tracks()
        dpr.close()
                
        T=reorient2light(T,(100,100,100))
        T=[t for t in T if length_limits(t,40.*2.5,200*2.5)]#120.*2.5)]
        Td = [downsample(t,12) for t in T]
        #Td = np.array(Td,dtype=np.object)    
        del T    
        C=local_skeleton_clustering(Td,4*2.5)        
        Cs.append(C)
        
    
    C0,lenC0=merge_Cs(Cs,4*2.5,True)    
    t2=time()
    print 'Done in', t2-t1        
    return C0,lenC0
    """ Ian's way of plotting this result after
    vs,ls,tot=bring_virtuals(C0)
    revlsort=100*(np.arange(len(ls))+1.)/len(ls)
    revcum=100*np.cumsum(np.sort(ls)[::-1])/np.sum(ls)
    figure()
    axis([0.,100.,0.,100.])
    xlabel('largest bundles (%)')
    ylabel('total of tracks (%)')
    grid()
    plot(revlsort,revcum)
    """


def close_to_sketch(sub=0,sdist=25.,method='mdf',type='max',sel=[2,3,8,9,10,11],one=False):
    
    """
    r=fvtk.ren()
    fvtk.add(r,fvtk.line(sketch,fvtk.red))
    for (i,s) in enumerate(sketch):
        fvtk.label(r,text=str(i),pos=s[-1],scale=(2,2,2))
    fvtk.show(r)
    """
    
    lfiles=[]
    
    dname='/home/eg309/Data/PROC_MR10032/'
    for root, dirs, files in os.walk(dname):        
        if root.endswith('101_32/GQI'):
            #print root
            for file in files:
                if file.endswith('lsc_QA_ref.dpy'):
                    #print file
                    lfiles.append(root+'/'+file)
                     
    f=lfiles[sub]
    dpr=Dpy(f,'r')        
    T=dpr.read_tracks()
    dpr.close()
                
    T=reorient2light(T,(100,100,100))
    T=[t for t in T if length_limits(t,40.*2.5,200*2.5)]#120.*2.5)]
    
    if method=='mdf':
        Td = [downsample(t,12) for t in T]
    if method=='mam':
        Td = [downsample(t,20) for t in T]
    
    sketch=load_pickle('/home/eg309/Data/LSC_limits/sketch3.pkl')
    sketch_colors=load_pickle('/home/eg309/Data/LSC_limits/cols_sketch3.pkl')
    dix=load_pickle('/home/eg309/Data/LSC_limits/sketch3_dix.pkl')
    
    """
    for s in dix: print s,dix[s]['name'],dix[s]['side']
    0 arcuate L
    1 arcuate R
    2 splenium CC C
    3 genu CC C
    4 inferior occipitofrontal L
    5 inferior occipitofrontal R
    6 cingulum L
    7 cingulum R
    8 pons + peduncle C
    9 front body CC C
    10 back body CC C
    11 middle body CC C
    12 fornix R
    13 fornix L
    14 cst R
    15 cst L
    16 uncinate R
    17 uncinate L
    18 optic radiation L
    
    close_to_sketch(sub=9,sdist=10.,method='mdf',type='min',sel=[16,17,18],one=False)
    """
    

    #sketch=sketch[-3:]
    #sketch_colors=sketch_colors[-3:,:]
    sketch=[sketch[i] for i in sel]
    sketch_colors=[sketch_colors[i] for i in sel]
    sketch_colors=np.array(sketch_colors)
    
    
    print len(Td[0])==len(sketch[0])
    
    if method=='mdf':        
        D=bundles_distances_mdf(Td,sketch)
    if method=='mam':
        D=bundles_distances_mam(Td,sketch,type)
    
    
    Dm=np.min(D,axis=1)
    I=np.argmin(D,axis=1)
    
    #sketch_colors=np.random.rand(len(sketch),4).astype('f4')
    #sketch_colors[:,-1]=1    
        
    w=World()
    wi=Window(bgcolor=(1.,1.,1.,1.),width=1000,height=1000)
    wi.attach(w)        
    
    for i in range(len(sketch)):
        Ind=np.where(I==i)[0]
        #vs=[Td[ind] for ind in Ind if Dm[ind]<sdist]
        vs=[T[ind] for ind in Ind if Dm[ind]<sdist]
        cols=np.zeros((len(vs),4)).astype('f4')
        cols[:]=sketch_colors[i]
        print dix[sel[i]]['name'],dix[sel[i]]['side'],len(vs)
                       
        c=InteractiveCurves(vs,colors=cols,centered=False,line_width=2.)
        w.add(c)
        if one:
            raw_input('Press Enter to continue ..')
            w.delete(c)
            
    #blacks=np.zeros((len(sketch),4),'f4')
    #blacks[:,-1]=1
    #c=InteractiveCurves(sketch,colors=blacks,centered=False,line_width=10.)
    #w.add(c)
    
def show_sketch(sel=[2,3,8,9,10,11],linewidth=5.):
    
    sketch=load_pickle('/home/eg309/Data/LSC_limits/sketch3.pkl')
    sketch_colors=load_pickle('/home/eg309/Data/LSC_limits/cols_sketch3.pkl')
    dix=load_pickle('/home/eg309/Data/LSC_limits/sketch3_dix.pkl')
    #sketch=sketch[-3:]
    #sketch_colors=sketch_colors[-3:,:]
    sketch=[sketch[i] for i in sel]
    sketch_colors=[sketch_colors[i] for i in sel]
    sketch_colors=np.array(sketch_colors)
    
    w=World()
    wi=Window(bgcolor=(1.,1.,1.,1.),width=1000,height=1000)
    wi.attach(w)
    
    
    c=InteractiveCurves(sketch,colors=sketch_colors,centered=False,line_width=linewidth)
    w.add(c)
    
def atlas_landmarks(bigt=500,t=0.8,p=1):
    
    atlas=load_pickle('/home/eg309/Data/LSC_limits/atlas.pkl')
    vs,ls,tot=bring_virtuals(atlas)
    sketch=load_pickle('/home/eg309/Data/LSC_limits/sketch3.pkl')    
    sketch_dix=load_pickle('/home/eg309/Data/LSC_limits/sketch3_dix.pkl')
    
    #big=[vs[i] for (i,l) in enumerate(ls) if l>50]
    #big2=[vs[i] for (i,l) in enumerate(ls) if l>200]
    big=[vs[i] for (i,l) in enumerate(ls) if l>bigt]
    #big4=[vs[i] for (i,l) in enumerate(ls) if l>1000]
            
    D=bundles_distances_mdf(big,big)
    """
    24: tr=np.tril(np.arange(16).reshape(4,4),-1)
    25: tr[np.where(tr>0)]
    26: mat=np.ones(tr.shape)
    27: mat=np.tril(mat)
    28: tr[np.where(mat>0)]
    29: mat=np.ones(tr.shape)
    30: mat=np.tril(mat,-1)
    31: tr[np.where(mat>0)]
    """
    
    b=[]
    for i in np.arange(0,len(D)-1):
        b+=list(D[i,i+1:])       
    b=np.array(b)
    print b
    print b.min(),b.mean(),b.max()
    Z=linkage(b,'single')
    dendrogram(Z)
    c=fcluster(Z,t,'distance')
    
    print len(set(c)),D.shape
    c_colors=np.random.rand(len(set(c)),4).astype('f4')
    c_colors[:,-1]=1    
    colors=np.zeros((len(c),4),'f4')
    colors[:,-1]=1
        
    w=World()
    wi=Window(caption='LSC+HC',bgcolor=(1.,1.,1.,1.),width=1000,height=1000)
    wi.attach(w)    
    for (i,ci) in enumerate(c):
        #print i,c
        colors[i]=c_colors[ci-1]
               
    c=InteractiveCurves(big,colors=colors,centered=False,line_width=10.)
    w.add(c)
    #"""
    #atlas_landmarks(450,1.1)
    
    
    S=-D
    ap_exemplars,ap_labels=cluster.affinity_propagation(S,p=p*np.median(S))
    print( "-------------------> len ap %d" % len(ap_exemplars))
    
    #return ap_exemplars,ap_labels
    c_colors2=np.random.rand(len(ap_exemplars),4).astype('f4')
    c_colors2[:,-1]=1
    colors2=np.zeros((len(ap_labels),4),'f4')
    colors2[:,-1]=1
    
    w2=World()
    wi2=Window(caption='LSC+AP',bgcolor=(1.,1.,1.,1.),width=1000,height=1000)
    wi2.attach(w2)    
    for (i,ci) in enumerate(ap_labels):
        #print i,c
        colors2[i]=c_colors2[ci]
               
    c=InteractiveCurves(big,colors=colors2,centered=False,line_width=10.)
    w2.add(c)
    #atlas_landmarks(450,1.1)

    """
    
    print('Create a new higher tree CAP') 
    CAP={}
    labels_=np.array(ap_labels)
    for c in range(len(ap_exemplars)):
        CAP[c]={}
        CAP[c]['most']=lsc_exemplars[ap_exemplars[c]]
        CAP[c]['indices']=list(np.where(labels_==c)[0])
        CAP[c]['N']=len(CAP[c]['indices'])
    """
    
def create_sketch():
    
    atlas=load_pickle('/home/eg309/Data/LSC_limits/atlas.pkl')
    vs,ls,tot=bring_virtuals(atlas)
    #sketch=load_pickle('/home/eg309/Data/LSC_limits/sketch.pkl')
    
    big=[vs[i] for (i,l) in enumerate(ls) if l>50]
    big2=[vs[i] for (i,l) in enumerate(ls) if l>200]
    big3=[vs[i] for (i,l) in enumerate(ls) if l>450]
    big4=[vs[i] for (i,l) in enumerate(ls) if l>1000]
    
    sketch=load_pickle('/home/eg309/Data/LSC_limits/sketch.pkl')  
    sketch2=[big4[i] for i in [28,56,29,49,23,21,25,55,11,20,24,14]]    
    fromsketch=[sketch[i] for i in [3,4,7,8,9,10]]
    sketch3=sketch2+fromsketch+[big3[158]]
    save_pickle('/home/eg309/Data/LSC_limits/sketch3.pkl',sketch3)
    
    r=fvtk.ren()
    for (i,v) in enumerate(sketch3):
        fvtk.label(r,text=str(i),pos=v[0],scale=(2,2,2))
        fvtk.add(r,fvtk.line(sketch3,fvtk.red))
    fvtk.show(r)


def pbc_simplifications():
    
    streams,header=tv.read('/home/eg309/Data/PBC/pbc2009icdm/brain1/brain1_scan1_fiber_track_mni.trk')
    T=np.array([s[0] for s in streams],np.object)
    labels=np.loadtxt('/home/eg309/Data/PBC/pbc2009icdm/brain1/brain1_scan1_fiber_labels.txt')[:,1]
    labels=np.array(labels)
    
    """
    1       Arcuate
    2       Cingulum
    3       Corticospinal
    4       Forceps Major
    5       Fornix
    6       Inferior Occipitofrontal Fasciculus
    7       Superior Longitudinal Fasciculus
    8       Uncinate
    """
    
    arc=list(T[np.where(labels==1)[0]])
    cing=list(T[np.where(labels==2)[0]])
    cst=list(T[np.where(labels==3)[0]])
    forc=list(T[np.where(labels==4)[0]])
    fx=list(T[np.where(labels==5)[0]])
    inf_occ=list(T[np.where(labels==6)[0]])
    sup_long=list(T[np.where(labels==7)[0]])
    unc=list(T[np.where(labels==8)[0]])

    #return cst
    
    """
    C=local_skeleton_clustering(cst,4.*2.5)
    cstd=[downsample(t,12) for t in cst]
    C=local_skeleton_clustering(cstd,4*2.5)
    vs,ls,tot=bring_virtuals(C)
    show_tracks(vs)
    show_tracks(cst,color=np.array([1,0,0,1],'f4'))
    show_tracks(vs,cmap='random')
    show_tracks(vs)
    show_tracks(cst,color=np.array([1,0,0,1],'f4'))
    """
    return cst


def check_registration_native_space(type=0,extra=False):
    
    lfiles=[]
    
    dname='/home/eg309/Data/PROC_MR10032/'
    for root, dirs, files in os.walk(dname):        
        if root.endswith('101_32/GQI'):
            #print root
            for file in files:
                if file.endswith('lsc_QA.dpy'):
                    #print file
                    lfiles.append(root+'/'+file) 
    
    #delete problematic files
    del lfiles[3]
    del lfiles[10]
    
    cnt=0
    for c in combinations(range(10),2):
    
        fs1=lfiles[c[0]]
        fs2=lfiles[c[1]] 
    
        print 'Combination number',cnt
        cnt+=1
        print fs1
        print fs2
        
        #f=lfiles[0]
        #print f
        dpr=Dpy(fs1,'r')        
        T0=dpr.read_tracks()
        dpr.close()

        dpr=Dpy(fs2,'r')        
        T1=dpr.read_tracks()
        dpr.close()
        
        #T1=shift(T1,np.array([50,0,0],'f4'))
        
        #First tractography        
        T0=reorient2light(T0,(0,0,0))
        T0=[t for t in T0 if length_limits(t,40.,120.)]
        Td0 = [downsample(t,12) for t in T0]
        del T0
        Td0 = np.array(Td0,dtype=np.object)
            
        iTd0=np.random.permutation(len(Td0))
        Td0=change_dtype(Td0[iTd0])
        print 'lenTd0',len(Td0)
        CAP0,C0,D0=exemplar_segmentation(Td0,4.)
        print 'lenC0',len(C0)
        
        #Second tractography
        T1=reorient2light(T1,(0,0,0))
        T1=[t for t in T1 if length_limits(t,40.,120.)]
        Td1 = [downsample(t,12) for t in T1]
        del T1
        Td1 = np.array(Td1,dtype=np.object)
        
        iTd1=np.random.permutation(len(Td1))
        Td1=change_dtype(Td1[iTd1])
        print 'lenTd1',len(Td1)
        CAP1,C1,D1=exemplar_segmentation(Td1,4.)
        print 'lenC1',len(C1)
        
        #return C0,C1        
        
        big0 = [C0[c]['most'].astype('f4') for c in C0 if C0[c]['N']> 100]
        big1 = [C1[c]['most'].astype('f4') for c in C1 if C1[c]['N']> 100]
       
        b0,b1,b2,b3,sol0,sol1=tight_comparison(big0,big1,4.)        
        BTC=(len(b0)/float(len(big0))+len(b2)/float(len(big1)))/2.
        
        shift0=-np.mean(np.concatenate(big0),axis=0)
        big0=shift(big0,shift0)
        
        shift1=-np.mean(np.concatenate(big1),axis=0)
        big1=shift(big1,shift1)
        
        if extra:
            big1=list(transform_track(big1,matrix44(np.array([0,0,0,0,45,0])) ))
                
        xopt=fmin_powell(call,[0,0,0,0,0,0],(big0,big1,type),xtol=10**(-6),ftol=10**(-6),maxiter=10**6)
        fbig=list(transform_track(big1,matrix44(xopt)))
        
        b0,b1,b2,b3,sol0,sol1=tight_comparison(big0,big1,4.)        
        TC=(len(b0)/float(len(big0))+len(b2)/float(len(big1)))/2.
        
        b0,b1,b2,b3,sol0,sol1=tight_comparison(big0,fbig,4.)
        NTC=(len(b0)/float(len(big0))+len(b2)/float(len(fbig)))/2.
        
        if extra:
            save_pickle('/tmp/'+'extra.pkl',[big0,big1,fbig,shift0,shift1,xopt,BTC,TC,NTC])
            return
        else:
            save_pickle('/tmp/'+str(cnt)+'.pkl',[big0,big1,fbig,shift0,shift1,xopt,BTC,TC,NTC])
        
        
        
def report_registration_check_results():
    
    RES=[]
    dname='/tmp'
    for cnt in range(1,46):
        res=load_pickle('/tmp/'+str(cnt)+'.pkl')
        big0,big1,fbig,shift0,shift1,xopt,BTC,TC,NTC = res
        print BTC,TC,NTC
        RES.append((BTC,TC,NTC))
    return np.array(RES)
        
    
def bring_big_exemplars():    
    
    lfiles=[]
    
    dname='/home/eg309/Data/PROC_MR10032/'
    for root, dirs, files in os.walk(dname):        
        if root.endswith('101_32/GQI'):
            #print root
            for file in files:
                if file.endswith('lsc_QA.dpy'):
                    #print file
                    lfiles.append(root+'/'+file) 

    
    
    f=lfiles[0]
    print f
    dpr=Dpy(f,'r')        
    T0=dpr.read_tracks()
    dpr.close()
    
    T1=list(T0)
    #T1=shift(T1,np.array([50,0,0],'f4'))
        
    T0=reorient2light(T0,(0,0,0))
    T0=[t for t in T0 if length_limits(t,40.,120.)]
    Td0 = [downsample(t,12) for t in T0]
    del T0
    Td0 = np.array(Td0,dtype=np.object)
        
    iTd0=np.random.permutation(len(Td0))
    Td0=change_dtype(Td0[iTd0])
    print 'lenTd0',len(Td0)
    CAP0,C0,D0=exemplar_segmentation(Td0,4.)
    print 'lenC0',len(C0)
    
    T1=reorient2light(T1,(0,0,0))
    T1=[t for t in T1 if length_limits(t,40.,120.)]
    Td1 = [downsample(t,12) for t in T1]
    del T1
    Td1 = np.array(Td1,dtype=np.object)
    
    iTd1=np.random.permutation(len(Td1))
    Td1=change_dtype(Td1[iTd1])
    print 'lenTd1',len(Td1)
    CAP1,C1,D1=exemplar_segmentation(Td1,4.)
    print 'lenC1',len(C1)
    
    #return C0,C1
    
    big0 =[C0[c]['most'].astype('f4') for c in C0 if C0[c]['N']> 100]
    big1=[C1[c]['most'].astype('f4') for c in C1 if C1[c]['N']> 100]
        
    return big0,big1


def rotation_vec2mat(r):
    """
    R = rotation_vec2mat(r)

    The rotation matrix is given by the Rodrigues formula:
    
    R = Id + sin(theta)*Sn + (1-cos(theta))*Sn^2  
    
    with:
    
           0  -nz  ny
    Sn =   nz   0 -nx
          -ny  nx   0
    
    where n = r / ||r||
    
    In case the angle ||r|| is very small, the above formula may lead
    to numerical instabilities. We instead use a Taylor expansion
    around theta=0:
    
    R = I + sin(theta)/tetha Sr + (1-cos(theta))/teta2 Sr^2
    
    leading to:
    
    R = I + (1-theta2/6)*Sr + (1/2-theta2/24)*Sr^2
    """
    theta = spl.norm(r)
    if theta > 1e-30:
        n = r/theta
        Sn = np.array([[0,-n[2],n[1]],[n[2],0,-n[0]],[-n[1],n[0],0]])
        R = np.eye(3) + np.sin(theta)*Sn + (1-np.cos(theta))*np.dot(Sn,Sn)
    else:
        Sr = np.array([[0,-r[2],r[1]],[r[2],0,-r[0]],[-r[1],r[0],0]])
        theta2 = theta*theta
        R = np.eye(3) + (1-theta2/6.)*Sr + (.5-theta2/24.)*np.dot(Sr,Sr)
    return R


def matrix44(t, dtype=np.double):
    """
    T = matrix44(t)

    t is a vector of of affine transformation parameters with size at
    least 6.

    size < 6 ==> error
    size == 6 ==> t is interpreted as translation + rotation
    size == 7 ==> t is interpreted as translation + rotation + isotropic scaling
    7 < size < 12 ==> error
    size >= 12 ==> t is interpreted as translation + rotation + scaling + pre-rotation 
    """
    size = t.size
    T = np.eye(4, dtype=dtype)
    
    #Degrees to radians
    rads=np.deg2rad(t[3:6])
    
        
    R = rotation_vec2mat(rads)
    #R = rotation_vec2mat(t[3:6])
    if size == 6:
        T[0:3,0:3] = R
    elif size == 7:
        T[0:3,0:3] = t[6]*R
    else:
        S = np.diag(np.exp(t[6:9])) 
        Q = rotation_vec2mat(t[9:12]) 
        # Beware: R*s*Q
        T[0:3,0:3] = np.dot(R,np.dot(S,Q))
    T[0:3,3] = t[0:3] 
    return T


def shift(vs,s):    
   return [v+s for v in vs]
   
def transform_track(T,aff):
    return [np.dot(t,aff[:3,:3].T)+aff[:3,3] for t in T]

def call(x0,big0,big,type=1):
    
    #x0[3:6]=np.interp(x0[3:6],[-10**4,10**4],[-np.pi,np.pi])

    aff=matrix44(x0)
    big=transform_track(big,aff)
    d01=bundles_distances_mdf(big0,big)
    if type==1:
        return np.sum(d01)
    if type==0:    
        return np.sum(np.min(d01,axis=0))+np.sum(np.min(d01,axis=1))
    
def linecall(x,i,x0,big0,big,type=1):    
    
    x0[i]=x
    aff=matrix44(x0)
    big=transform_track(big,aff)
    d01=bundles_distances_mdf(big0,big)
    if type==1:
        return np.sum(d01)
    if type==0:    
        return np.sum(np.min(d01,axis=0))+np.sum(np.min(d01,axis=1))

def test_minimize(big0,big1):   
    res=[]
    for x in range(-100,100,10):    
        big=shift(big1,np.array([x,0,0],'f4'))        
        d01=bundles_distances_mdf(big0,big)
        res.append((x,np.sum(d01)))    
    #big=shift(big1,np.array([50,0,0],'f4'))
    big=list(big1)
    xopt=fmin_powell(call,[0,0,0,0,0,0],(big0,big))
    return xopt,res

def registration_experiment(big0,big1,type):
    
    """
    big0,big1=bring_big_exemplars()

    big0=shift(big0,-np.mean(np.concatenate(big0),axis=0))
    #big1=shift(big1,-np.mean(np.concatenate(big1),axis=0))
    big1=list(big0)   
    """
    
    #vec=np.array([0,20,10,np.pi/4.,0,0])
    vecs=90*np.random.rand(1000,6) - 45
    res=[]
    for vec in vecs:
    
        big=list(transform_track(big1,matrix44(vec)))
        big=shift(big,-np.mean(np.concatenate(big),axis=0))        
        #show_2_bundles(big0,big)
        #xopt, fopt, direc, iter, funcalls, warnflag,allvecs=fmin_powell(call,[0,0,0,0,0,0],(big0,big),full_output=True,retall=True)#,xtol=1e-10,ftol=1e-10,maxiter=10**3,maxfun=10**3)
        xopt=fmin_powell(call,[0,0,0,0,0,0],(big0,big,type),xtol=10**(-6),ftol=10**(-6),maxiter=10**6)
        #xopt=fmin_simplex(call,[0,0,0,0,0,0],(big0,big))
        #xopt,retval=anneal(call,[0,0,0,0,0,0],(big0,big),schedule='fast')
        #xopt=fmin_simplex(call,[0,0,0,0,0,0],(big0,big))
        #xopt,fopt,dix=fmin_l_bfgs_b(call,[0,0,0,0,0,0],None,(big0,big),True,[(-100,100),(-100,100),(-100,100),(0,2*np.pi),(0,2*np.pi),(0,2*np.pi)])
        #print 'res',xopt
        fbig=list(transform_track(big,matrix44(xopt)))
        #show_2_bundles(big0,fbig)        
        #raw_input('Press Enter to continue...')
        D=bundles_distances_mdf(big0,fbig)
        res.append(np.sum(np.diag(D)))
    
    print xopt#,retval
    print res
    
    return np.array(res)#,big,fbig


def plot_global_minimum(big0,big1):
    res=[]
    res2=[]
    for x in range(-100,100,1):
        big=shift(big1,np.array([x,0,0],'f4'))
        d01=bundles_distances_mdf(big0,big)
        res.append((x,np.sum(d01)))
        minm=np.sum(np.min(d01,axis=0))+np.sum(np.min(d01,axis=1))
        #minm=minm**2/10**3
        res2.append((x,minm))
        
    res=np.array(res)
    res2=np.array(res2)
    
    plt.figure(1)
    plt.plot(res[:,0],res[:,1])
    plt.xlabel('translation along x axis in native units (2.5*mm)')
    plt.ylabel('optimizer value')
    plt.title('Smooth global minimum found - convex')
    plt.show()
    
    plt.figure(2)
    plt.plot(res2[:,0],res2[:,1])
    plt.xlabel('translation along x axis in native units (2.5*mm)')
    plt.ylabel('optimizer value')
    plt.title('Smooth global minimum found')
    plt.show()

def plot_global_minimum_angle(big0,big1):
    #big0,big1=bring_big_exemplars()
    vecs=np.zeros((360,6))
    k=3
    vecs[:,k]=np.arange(-180,180)
    
    #vecs[:,k]=np.interp(vecs[:,k],[-10**2,10**2],[-np.pi,np.pi])
    #vecs[:,k]=np.interp(vecs[:,k],[-10**4,10**4],[-180,180])
    plt.figure(1)
    plt.plot(vecs[:,k])
    plt.show()
    plt.figure(2)
    res=[]
    res2=[]
    for vec in vecs:
        big=list(transform_track(big1,matrix44(vec)))
        d01=bundles_distances_mdf(big0,big)
        res.append(np.sum(d01))
        res2.append(np.sum(np.min(d01,axis=0))+np.sum(np.min(d01,axis=1)))
        #res2.append(vec[k])
    plt.plot(np.arange(-180,180),res)
    plt.xlabel('rotation of first polar angle in degrees')
    plt.ylabel('optimizer value')
    plt.title('Smooth global minimum found - non-convex')
    plt.show()
    plt.figure(3)    
    plt.plot(np.arange(-180,180),res2)
    plt.xlabel('rotation of first polar angle in degrees')
    plt.ylabel('optimizer value')
    plt.title('Smooth global minimum found - non-convex')
    
    plt.show()
    
def plot_very_winding():
    
    f='/home/eg309/Data/LSC_limits/full_1M.npy'
    T=list(np.load(f))
    T=[t for t in T if length_limits(t,40.,120.)]
    Td = [downsample(t,12) for t in T]
    C,lenC=parallel_lsc(Td,lsc_dist=4.,subset=10000)
    lscex=lsc_exemplars(C,Td)   
    wind=[]

    for c in range(len(C)):
        if winding(C[c]['most'])>500:
            wind.append(c)
        
    wex=[]    
    for w in wind:
        t=C[w]['most']                    
        U,s,V=np.linalg.svd(t-np.mean(t,axis=0),0)
        proj=np.dot(U[:,0:2],np.diag(s[0:2]))
        #W=[Td[i] for i in C[w]['indices']]
        W=[T[i] for i in C[w]['indices']] 
        wex.append((t,proj,W))

    return wex
    
def show_very_winding(wex):    
    w=World()
    wi=Window(bgcolor=(1.,1.,1.,1.),width=2000,height=1000)
    wi.attach(w)    
    for we in wex:        
        vs=we[2]
        col=np.random.rand(4)
        col[-1]=1.        
        cols=np.random.rand(len(vs),4)      
        cols[:]=col    
        c=InteractiveCurves(vs,colors=cols,centered=False,line_width=3.)
        w.add(c)
        #raw_input('Press Enter to continue ...')
        
def vec2vecrotmat(u,v):
    """ 2 unit vectors rotation matrix
    
    u,v being unit 3d vectors return the 3x3 rotation matrix R than aligns u to v
    The transpose of R will align v to u
    """
    
    w=np.cross(u,v)
    w=w/np.linalg.norm(w)
    
    # vp is in plane of u,v,  perpendicular to u
    vp=(v-(np.dot(u,v)*u)) 
    vp=vp/np.linalg.norm(vp)
    
    # (u vp w) is an orthonormal basis   
    
    P=np.array([u,vp,w])
    Pt=P.T
    cosa=np.dot(u,v)
    sina=np.sqrt(1-cosa**2)
    R=np.array([[cosa,-sina,0],[sina,cosa,0],[0,0,1]])
    Rp=np.dot(Pt,np.dot(R,P))
    return Rp


def arcuate_small_bundles(sub=0,sdist=25.,arcdist=2.,method='mdf',type='max',sel=[0,1],one=False):
    
    """
    r=fvtk.ren()
    fvtk.add(r,fvtk.line(sketch,fvtk.red))
    for (i,s) in enumerate(sketch):
        fvtk.label(r,text=str(i),pos=s[-1],scale=(2,2,2))
    fvtk.show(r)
    """
    
    lfiles=[]
    
    dname='/home/eg309/Data/PROC_MR10032/'
    for root, dirs, files in os.walk(dname):        
        if root.endswith('101_32/GQI'):
            #print root
            for file in files:
                if file.endswith('lsc_QA_ref.dpy'):
                    #print file
                    lfiles.append(root+'/'+file)
                     
    f=lfiles[sub]
    dpr=Dpy(f,'r')        
    T=dpr.read_tracks()
    dpr.close()
                
    T=reorient2light(T,(100,100,100))
    Tall=list(T)
    T=[t for t in T if length_limits(t,40.*2.5,200*2.5)]#120.*2.5)]
    
    if method=='mdf':
        Td = [downsample(t,12) for t in T]
    if method=='mam':
        Td = [downsample(t,20) for t in T]
    
    sketch=load_pickle('/home/eg309/Data/LSC_limits/sketch3.pkl')
    sketch_colors=load_pickle('/home/eg309/Data/LSC_limits/cols_sketch3.pkl')
    dix=load_pickle('/home/eg309/Data/LSC_limits/sketch3_dix.pkl')

    #sketch=sketch[-3:]
    #sketch_colors=sketch_colors[-3:,:]
    sketch=[sketch[i] for i in sel]
    sketch_colors=[sketch_colors[i] for i in sel]
    sketch_colors=np.array(sketch_colors)    
    
    print len(Td[0])==len(sketch[0])    
    if method=='mdf':
        D=bundles_distances_mdf(Td,sketch)
    if method=='mam':
        D=bundles_distances_mam(Td,sketch,type)
    
    Dm=np.min(D,axis=1)
    I=np.argmin(D,axis=1)
    #sketch_colors=np.random.rand(len(sketch),4).astype('f4')
    #sketch_colors[:,-1]=1    
    w=World()
    wi=Window(bgcolor=(1.,1.,1.,1.),width=1000,height=1000)
    wi.attach(w)    
    
    Ind=np.where(I==i)[0]
    vs=[T[ind] for ind in Ind if Dm[ind]<sdist]
    cols=np.zeros((len(vs),4)).astype('f4')
    cols[:]=sketch_colors[0]
    c=InteractiveCurves(vs,colors=cols,centered=False,line_width=2.)
    w.add(c)   
    
    #return vs,Tall
    vsd=[downsample(t,12) for t in vs]
    C=local_skeleton_clustering(vsd,2*2.5)
    vs2,ls2,tot2=bring_virtuals(C)
    Tall20 = [downsample(t,20) for t in Tall]
    Darc=bundles_distances_mam(vs2,Tall20,'min')
    x,y=np.where(Darc<arcdist)
    yset=set(y)
    arcuateclose=[Tall20[i] for i in yset]
    
    return vs,vs2,arcuateclose

def barchart_QB():    

    
    N = 2
    Means = (0.4690955208544213,0.52516460626048289)
    Stds =   (0.02635549381315298,0.049120754359623708)

    ind = np.array([0.5,1.5])#np.arange(N)  # the x locations for the groups
    width = 0.35       # the width of the bars

    fig = plt.figure()
    ax = fig.add_subplot(111)
    rects1 = ax.bar(ind, Means, width, color='r', yerr=Stds,linewidth=10)
        
    # add some
    ax.set_ylabel('mean')
    ax.set_title('TC comparisons between all the combinations of 10 real subjects')
    ax.set_xticks(ind+width/2)
    ax.set_xticklabels( ('TC10', 'TC100') )
    
    #ax.legend( (rects1[0], rects2[0]), ('Men', 'Women') )    
    def autolabel(rects):
        # attach some text labels
        for rect in rects:
            height = rect.get_height()
            ax.text(rect.get_x()+rect.get_width()/2., 1.05*height, '%d'%int(height),
                    ha='center', va='bottom')
    
    #autolabel(rects1)
    #autolabel(rects2)
    
    plt.show()
    
def barchart_multiple_orderings():
    
    dname='/home/eg309/Data/LSC_limits/multiple_comparisons/results_full_'    
    metrics = ['Purity', 'Random Accuracy', 'Pairs Concordancy', 'Completeness', 'Correctness', 'Matched Agreement', 'Matched Kappa']
    N=10
    Means=np.zeros((N,7))
    Std=np.zeros((N,7))        
    for i in range(N):
        fname=dname+str(i)+'.pkl'        
        means,sds,table=analyze_multiple_comparisons(fname)
        Means[i,:]=means
        Std[i,:]=sds        

    fig = plt.figure()
    ind = np.arange(N)+0.2  # the x locations for the groups
    width = 0.3       # the width of the bars
    
    colors=['r','g','b','c','m','y',(1,0.5,0.2)]
        
    for i in range(7):
        if i==6:    
            ax = fig.add_subplot(3,3,8)
            ax.set_xlabel('Subjects')
        else:
            ax = fig.add_subplot(3,3,i+1)
        rects1 = ax.bar(ind, Means[:,i], width, color=colors[i], yerr=Std[:,i])
        #rects1 = ax.bar(ind, Means[:,i], width, yerr=Std[:,i])        
        ax.set_ylabel('Scores')
        ax.set_ylim((0,110))
        ax.set_title(metrics[i])
        ax.set_xticks(ind+width/2.)
        ax.set_xticklabels(('1', '2', '3', '4', '5', '6','7','8','9','10'))
            
    plt.show()
    
    return

def sizes_orderings_tractographies():
    dname='/home/eg309/Data/LSC_limits/multiple_comparisons/'    
    fnames=['C_size35750_0.pkl','C_size36889_1.pkl','C_size33612_2.pkl',\
     'C_size46358_3.pkl','C_size40400_4.pkl','C_size39967_5.pkl',\
     'C_size56256_6.pkl','C_size48743_7.pkl','C_size37098_8.pkl','C_size36244_9.pkl']
    
    Means=np.zeros(10)
    Std=np.zeros(10)
    
    for (i,f) in enumerate(fnames):
        filename=dname+f
        dix=load_pickle(filename)
        sz=[]
        for d in dix:
            sz.append(dix[d])
        Means[i]=np.mean(sz)
        Std[i]=np.std(sz)
        
    fig = plt.figure()
    ind = np.arange(10)+0.2  # the x locations for the groups
    width = 0.5       # the width of the bars
    
    colors=['r','g','b','c','m','y',(1,0.5,0.2)]
        
    ax = fig.add_subplot(1,1,1)
    ax.set_xlabel('Subjects')
    rects1 = ax.bar(ind, Means, width, color=(24/255.,159/255.,222/255.), yerr=Std)
            
    ax.set_ylabel('Number of Clusters')
    #ax.set_ylim((0,110))
    ax.set_title('Number of clusters for 20 different orderings')
    ax.set_xticks(ind+width/2.)
    ax.set_xticklabels(('1', '2', '3', '4', '5', '6','7','8','9','10'))
            
    plt.show()
    

def analyze_multiple_comparisons(filename):

    metrics = ['Purity', 'RandomAccuracy', 'PairsConcordancy', 'Completeness', 'Correctness', 'MatchedAgreement', 'MatchedKappa']
    
    alldata = []   
    
    results=load_pickle(filename)        
    keys = results.keys()        
    table = np.zeros((len(results), len(metrics)))        
    for i,k in enumerate(keys):
        r = results[k]
        for j,m in enumerate(metrics):
            table[i,j] = r[m]
    alldata = alldata + [table]                
    d = describe(table,axis=0)        
    #size = d[0]
    type = ['Min','Max','Mean','s.d.']
    #print >> f, size,'", Min,"'
    mins = (None,type[0])+tuple(d[1][0])
    maxs = (None,type[1])+tuple(d[1][0])
    means = (None,type[2])+tuple(d[2])
    sds = (None,type[3])+tuple(np.sqrt(d[3]))
    tab = np.vstack((means,sds))        
    
    return means[2:], sds[2:], tab[:,2:]

#def run_multiple_comparisions():
if __name__=='__main__':    
    
    repls=20
    
    from LSC_stats import multiple_comparisons
            
    dname='/home/eg309/Data/PROC_MR10032/'
    lfiles=[]
    for root, dirs, files in os.walk(dname):        
        if root.endswith('101_32/GQI'):
            #print root
            for file in files:
                if file.endswith('lsc_QA_ref.dpy'):
                    #print file
                    lfiles.append(root+'/'+file)
    
    for (i,f) in enumerate(lfiles):
        #f=lfiles[0]
        print f    
        #break    
        dpr=Dpy(f,'r')        
        T=dpr.read_tracks()
        dpr.close()                
        #T=np.load(fsolid)
        T=[downsample(t,12) for t in T]
        print 'Before',len(T)
        T=np.array([t for t in list(T) if length(t)>= 40.*2.5 and length(t)< 120.*2.5],dtype=np.object) # 100mm - 200mm
        print 'After',len(T)    
        results = multiple_comparisons(T,samplesize=len(T),lscdist = 4.*2.5, replications=repls,subj=str(i))
        fres='results_full_'+str(i)+'.pkl'
        save_pickle(fres, results)        
        #break



"""
print 'Overlap matrix comparisons'

compare(CAP0g2,iTd0,CAP1g2,iTd1)

print 'Direct comparisons'

ex0,exls0,tot0=bring_exemplars(CAP0)
ex1,exls1,tot1=bring_exemplars(CAP1)

d01=bundles_distances_mdf(ex0,ex1)

dist=2.

nearest_fraction(d01,dist,exls0,exls1)
cover_fraction(d01,exls0,exls1,dist)
near12 = np.sum(d01<=dist,axis=1)
near21 = np.sum(d01<=dist,axis=0)
"""
 
#t1=time()
#CAP0,C0,ex0=exemplar_segmentation(Td0,.5)
#t2=time()
#print t2-t1

#t1=time()
#CAP1,C1,ex1=exemplar_segmentation(Td1,.5)
#t2=time()
#print t2-t1




"""
iTd1=np.random.permutation(len(Td))    
Td1=list(Td[iTd1])

C1,vs1,ls1,lenC1=parallel_lsc(Td1,lsc_dist=6.,subset=10000)

iTd2=np.random.permutation(len(Td))
Td2=list(Td[iTd2])

C2=local_skeleton_clustering(Td2,6.)
vs2,ls2,tot2=bring_virtuals(C2)

iTd3=np.random.permutation(len(Td))
Td3=list(Td[iTd3])

C3=local_skeleton_clustering(Td3,6.)
vs3,ls3,tot3=bring_virtuals(C3)
"""


"""
print 'C0 vs C1'
compare(C0,iTd0,C1,iTd1)
print 'C0 vs C2'
compare(C0,iTd0,C2,iTd2)
print 'C1 vs C2'
compare(C1,iTd1,C2,iTd2)
print 'C2 vs C3'
compare(C2,iTd2,C3,iTd3)
"""


"""
vstmp=[v[5:7] for v in vs]
from dipy.tracking.metrics import midpoint
pts=[midpoint(v) for v in vs]
PTS=np.zeros((len(pts),3))
for i in range(len(PTS)):
    PTS[i]=pts[i]    

r=fvtk.ren()
fvtk.add(r,fvtk.point(PTS,np.ones((len(pts),3)),point_radius=.8,theta=6,phi=6))
fvtk.show(r)
fvtk.clear(r)
fvtk.add(r,fvtk.point(PTS,np.ones((len(pts),3)),point_radius=.2,theta=6,phi=6))
fvtk.show(r)
fvtk.clear(r)
fvtk.add(r,fvtk.point(PTS,np.ones((len(pts),3)),point_radius=.2,theta=6,phi=6))
fvtk.add(r,fvtk.line(vs,fvtk.red))
fvtk.show(r)


"""

"""
for h in range(-20,0):
    print h, np.sort(ls)[h],np.sort(ls1)[h],np.sort(ls2)[h]
    show_tracks3([vs[np.argsort(ls)[h]],vs1[np.argsort(ls1)[h]],vs2[np.argsort(ls2)[h]]])
    raw_input('Press Enter ..')
"""
#print nearest_fraction(bundles_distances_mdf(vs,vs1),1.,ls,ls1)
#print nearest_fraction(bundles_distances_mdf(vs,vs2),1.,ls,ls2)
#print nearest_fraction(bundles_distances_mdf(vs1,vs2),1.,ls1,ls2)
    
#print cover_fraction(bundles_distances_mdf(vs,vs1),ls,ls1,1.)
#print cover_fraction(bundles_distances_mdf(vs,vs2),ls,ls2,1.)
#print cover_fraction(bundles_distances_mdf(vs1,vs2),ls1,ls2,1.)

"""
collect=[]
for h in range(-200,0):
    collect.append(np.sort(bundles_distances_mdf([vs[np.argsort(ls)[h]]],vs).ravel())[1])
    
plot(np.sort(collect))
collect2=[]
for h in range(0,200):
    collect2.append(np.sort(bundles_distances_mdf([vs[np.argsort(ls)[h]]],vs).ravel())[1])
plot(np.sort(collect2))
"""



""" show light
r=fvtk.ren()
fvtk.add(r,fvtk.line(vs[:100],fvtk.red))
for v in vs[:100]:
    fvtk.label(r,text='0',pos=v[0],scale=(.5,.5,.5))
fvtk.show(r)

"""







"""
small2=reorient2light(arc)
#small2=positivity_wins(arc)
pts=[v[0] for v in small2]
PTS=np.zeros((len(pts),3))
for i in range(len(PTS)):
    PTS[i]=pts[i]
    
r=fvtk.ren()
fvtk.add(r,fvtk.line(small2,fvtk.red))
fvtk.add(r,fvtk.point(PTS,np.ones((len(pts),3)),point_radius=.2,theta=6,phi=6))
fvtk.show(r)




"""

#"""
#T=arc+cing+cst+forc+fx+inf_occ+sup_long+unc

"""
lsc_results=[]
ap_results=[]

priors=[.2,.3,.4,.5,.6,.7,.8,.9,1.,1.1,1.2,1.3,1.4,1.5,1.6]

for i in range(10):
    #priors=[.6] 
    D,lscex,C,Td=lsc(T,filt_short=4.*2.5,lsc_dist=4.*2.5,min_csize=0)
    print len(lscex[0]),len(lscex[1]),len(lscex[2]),len(lscex[3])
    cap_lens=[]
    for prior in priors:        
        print '>>>>>>>>>',i,prior
        CAP=ap(D,lscex,prior)
        cap_lens.append(len(CAP))
    lsc_results.append(len(C))
    ap_results.append(cap_lens)
    #show_cap(CAP,C,Td,one=False,wait=False)
    #raw_input('Press Enter to continue... ')
"""

#D,lscex,C,Td=lsc(T,filt_short=2*4.*2.5,lsc_dist=4.*2.5,min_csize=0)
#D,lscex,C,Td=lsc(T,filt_short=2*.4,lsc_dist=6.,min_csize=0)#tracks smaller than 20mm are removed
#vs,ls,tot=bring_virtuals(C)
#tcs=track_counts(vs,(96/2.,96/2.,55/2.),(2.,2.,2.),False)
#



"""
10 permutations with 9 distances - 90 runs same brain
 
dists=[2.,3.,4.,5.,6.,7.,8.,9.,10.]
lsc_res=[]
for i in range(10):
    tmp=[]
    for d in dists:
        D,lscex,C,Td=lsc(T,filt_short=4.*2.5,lsc_dist=d*2.5,min_csize=0)
        tmp.append(len(C))
    lsc_res.append(tmp)
    
"""


#CAP,C,Td=lscap(T,40.,10,4.,40,20,'avg',.2)#1 was very compact
#show_cap(CAP,C,Td)

#CAP2,C2,Td2=lscap(T,40.,10,4.,40,20,'avg',.2)
#show_cap(CAP2,C2,Td2)

#Y1=load_pickle('/tmp/it1.pkl')
#Y2=load_pickle('/tmp/it2.pkl')
#CAP=Y1['CAP']
#C=Y1['C']
#Td=Y1['Td']

#CAP2=Y2['CAP']
#C2=Y2['C']
#Td2=Y2['Td']
            
#res12=closest_neighb_tcs(CAP,C,Td,CAP2,C2,Td2)
#len(set(res12[:,1]))

#res11=closest_neighb_tcs(CAP,C,Td,CAP,C,Td)

#res21=closest_neighb_tcs(CAP2,C2,Td2,CAP,C,Td)
#len(set(res21[:,1]))

"""
print len(T)
T=np.array([t for t in list(T) if length(t)>= 40.],dtype=np.object) # 100mm
print len(T)
i1=np.random.permutation(len(T))
i2=np.random.permutation(len(T))
T1=list(T[i1])
T2=list(T[i2])

msim1,virts1,ls1,inds1=most_similar_tracks(T1,20,4.,40,'avg')
#msim2,virts2,ls2=most_similar_tracks(T2,20,4.,5,'avg')
"""
#ap_part(msim1,ls1)
#ap_part(msim2,ls2)






