from time import time
import numpy as np
import nibabel as nib
from dipy.reconst.gqi import GeneralizedQSampling
from dipy.tracking.propagation import EuDX
from dipy.tracking.distances import local_skeleton_clustering
from dipy.tracking.metrics import downsample,length
from dipy.io.dpy import Dpy
from dipy.io.pickles import load_pickle,save_pickle
import matplotlib.pyplot as plt
from dipy.tracking.distances import bundles_distances_mam
from dipy.tracking.distances import bundles_distances_mdf
from dipy.viz import fvtk
import hung_APC

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
        T=[downsample(e,12) for e in eu]
        #np.save(dout+outs[i]+'.npy',np.array(T,dtype=np.object))
        C=local_skeleton_clustering(T,4.)
        save_pickle(dout+outs[i]+'.skl',C)
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

def nearest_fraction(d12,dist,great_than=0):
    
    near12 = np.sum(d12<=dist,axis=1)
    near21 = np.sum(d12<=dist,axis=0)
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

#os.system('gcc -O2 -c -fPIC -l/usr/include/python2.6/ maxkapS1.c')
#os.system('gcc -shared maxkapS1.o -o maxkapS1.so')

import ctypes
max_kappa=ctypes.cdll.LoadLibrary('./maxkapS1.so')

def maxKappa(x,y):
    newdata = np.row_stack((x,y))
    newdata = newdata.astype('f4')
    newdata=newdata+1 #add one to labels ... so they start at 1 in C-fashion
    print newdata.shape
    maxclasses = np.max([np.max(x),np.max(y)])+1
    nclasses = np.array(maxclasses).astype('i4')
    print 'nclasses ', nclasses
    nunits = np.array(newdata.shape[1]).astype('i4')
    print 'units ', nunits
    nmeths = np.array(newdata.shape[0]).astype('i4')
    print 'meths', nmeths
    print (nclasses+1)*nmeths*(nmeths-1)/2
    ans=np.zeros((nclasses+1)*nmeths*(nmeths-1)/2).astype('f4')
    ncl=np.array([nclasses],'i4')
    order = np.array(np.arange(nclasses),'i4')
    max_kappa.clustComp(ans.ctypes.data,newdata.ctypes.data,nunits.ctypes.data,nmeths.ctypes.data,ncl.ctypes.data,order.ctypes.data)
    maxkappa = ans[0]
    maxorder = np.argsort(ans[1:(nclasses+1)])
    return maxkappa, maxorder

'''
fn1='/home/ian/Data/LSC_stability/1000000.skl'
#fn1='/home/eg309/Data/TMP_LSC_limits2/10000000.skl'
C1=load_pickle(fn1)
vs1,ls1,total1=bring_virtuals(C1)
fn2='/home/ian/Data/LSC_stability/1000000_2.skl'
#fn2='/home/eg309/Data/TMP_LSC_limits2/1000000_2.skl'
C2=load_pickle(fn2)
vs2,ls2,total2=bring_virtuals(C2)
'''

'''
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

'''

#fsolid='/home/eg309/Data/LSC_limits/solid_1M.npy'
#fsolid='/home/ian/Data/LSC_stability/solid_1M.npy'

#T=np.load(fsolid)
#print 'Before',len(T)
#T=np.array([t for t in list(T) if length(t)>= 40. and length(t)< 120.],dtype=np.object) # 100mm - 200mm
#print 'After',len(T)

#i11=np.random.permutation(1*10**3)#len(T))
#i12=np.random.permutation(1*10**3)#len(T))
#i21=1*10**4+np.random.permutation(1*10**4)#len(T))
#i22=1*10**4+np.random.permutation(1*10**4)#len(T))

#T11=list(T[i11])
#T12=list(T[i12])
#T21=list(T[i21])
#T22=list(T[i22])

#dw1=bundles_distances_mdf(T11,T12)
#print "initial nf",nearest_fraction(dw1,4.)
#dw2=bundles_distances_mdf(T21,T22)
#print "initial nf",nearest_fraction(dw2,4.)

#C11=taleton(T11,10*[4.])
#C12=taleton(T12,10*[4.])
#save_pickle('/home/eg309/Data/LSC_limits/solid_1M_1_new.tal',C1s)
#C21=taleton(T21,10*[4.])
#C22=taleton(T22,10*[4.])
#save_pickle('/home/eg309/Data/LSC_limits/solid_1M_2_new.tal',C2s)

#"""

def show_coverage(C1,C2):
    vs1,ls1,tot1=bring_virtuals(C1)
    vs2,ls2,tot2=bring_virtuals(C2)
    d12=bundles_distances_mdf(vs1,vs2)
    nfa,nfb = nearest_fraction(d12,4.)
    cfa,cfb = cover_fraction(d12,ls1,ls2,4.,1,1)
    print "nf %6.1f %6.1f cf %6.1f %6.1f" % (100*nfa,100*nfb,100*cfa,100*cfb)    
    return d12

"""
print 'initial w1'
show_coverage(C11[0],C12[0])
print 'initial w2'
show_coverage(C21[0],C22[0])
print 'initial b1'
show_coverage(C11[0],C21[0])
print 'initial b2'
show_coverage(C12[0],C22[0])
print 'initial b3'
show_coverage(C11[0],C22[0])
print 'initial b4'
show_coverage(C12[0],C21[0])
print 'last w1'
show_coverage(C11[-1],C12[-1])
print 'last w2'
show_coverage(C21[-1],C22[-1])
print 'last b1'
show_coverage(C11[-1],C21[-1])
print 'last b2'
show_coverage(C12[-1],C22[-1])
print 'last b3'
show_coverage(C11[-1],C22[-1])
print 'last b4'
show_coverage(C12[-1],C21[-1])
"""

"""
r=fvtk.ren()
vs0,ls0,tot0=bring_virtuals(C1s[0])
fvtk.add(r,fvtk.line(vs0,fvtk.blue))
#fvtk.show(r)

print 'lenvs0',len(vs0)

#r=fvtk.ren()
vs1,ls1,tot1=bring_virtuals(C1s[-1])
fvtk.add(r,fvtk.line(vs1,fvtk.red))
fvtk.show(r)

print 'lenvs1',len(vs1)

#C1=local_skeleton_clustering(T1,4.)
#vs1,ls1,tot1=bring_virtuals(C1)
#C2=local_skeleton_clustering(T2,4.)
#vs2,ls2,tot2=bring_virtuals(C2)
#d12 = bundles_distances_mdf(vs1,vs2).astype(np.float32)
"""
 
#from scipy import sparse
def overlap_matrix(A,iA,B,iB):
    '''
    calculate intersection matrix of a pair
    of clusterings A and B. iA and iB are the labels for 
    the items in the clusterings.
    
    Parameters:
    ===========
    A: LSC holds list of track indices in bundles
    iA: original labels of tracks in A
    B: LSC holds list of (same) track indices in bundles
    iB: original labels of tracks in B
    '''
    #print 'calculating [square] overlap matrix ... '
    M = np.max([len(A),len(B)])
    N = np.zeros((M,M))
    B_dic = {}
    for j in range(len(B)):
        for i in B[j]['indices']:
            B_dic[iB[i]] = j
    labs=set([B_dic[k] for k in B_dic.keys()])
    # B_dic is a dictionary in which we can look up the B-class 
    # to which an indexed track has been assigned 
    for i in range(len(A)):
        for j in A[i]['indices']:
            N[i,B_dic[iA[j]]] += 1
    return N

def tracks2classes(A,iA):
    '''
    calculate labelling vector of a clustering A,
    where iA are the global indices for the items 
    in the clustering.
    
    Parameters:
    ===========
    A: LSC holds list of track indices in bundles
    iA: original labels of tracks in A
    '''
    #print 'calculating labelling vector ... '
    x = {}
    for i in range(len(A)):
        for j in A[i]['indices']:
            x[iA[j]] = i    
    return x

def compare_classifications(N):
    '''
    calculate various measures for an overlap matrix
    
    Incorporating:  Gini Impurities, Classification Errorsm
    Information, Disagreements

    Parameters:
    ===========
    N: a padded (M,M) matrix of overlap counts
    
    Output:
    =======
    gini_stats: ...
    class_err_stats: ...
    '''

    A_tot = np.sum(N,axis=1,dtype=float)
    B_tot = np.sum(N,axis=0,dtype=float)
    N_tot = np.sum(A_tot)
    
    #print 'NaN counts:', np.sum(np.isnan(N)), np.sum(np.isnan(A_tot)), np.sum(np.isnan(B_tot))
    
    #print np.min(A_tot), np.max(A_tot), np.min(B_tot), np.max(B_tot)
    
    #A_tmp = (1-np.sum(N**2,axis=1)/A_tot**2)*A_tot)
    gini_A = np.sum(((1-np.sum(N**2,axis=1)/A_tot**2)*A_tot)[A_tot>0])/N_tot
    gini_B = np.sum(((1-np.sum(N**2,axis=0)/B_tot**2)*B_tot)[B_tot>0])/N_tot
    
    #print 'ginis', gini_A, gini_B
    
    #info_A_given_B = np.log2(A_tot)-np.sum(N*np.log2(np.maximum(N,1)),axis=1)
    #info_B_given_A = np.log2(B_tot)-np.sum(N*np.log2(np.maximum(N,1)),axis=0)

    random_class_error_A = np.sum(A_tot-np.max(N,axis=1))/N_tot
    random_class_error_B = np.sum(B_tot-np.max(N,axis=0))/N_tot

    #print 'calculating disagreements ... '
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
    p_discord = discord/n_pairs
    
    #hung_agree, hung_kappa, map = hungarian_matching(N)
    #print '... starting APC hungarian matching'
    hung_agree, hung_kappa, map = hungarian_APC(N)    
    
    return {'gini12':gini_A, 'gini21': gini_B, \
            'rce1': random_class_error_A, 'rce2': random_class_error_B, \
            'raw_concord': 1.-p_discord, \
            'correctness1': correctnessA, 'completeness1': completenessA, \
            'correctness2': correctnessB, 'completeness2': completenessB, \
            'matched_agree': hung_agree, 'matched_kappa': hung_kappa, \
            'matching': map}

def kappa(N):
    rowsum = np.sum(N,axis=1)
    colsum = np.sum(N,axis=0)
    total = np.sum(N)
    agree = np.sum([N[i,i] for i in range(min(N.shape))])/total
    chance = np.sum(colsum*rowsum)/total**2
    kappa = (agree-chance)/(1-chance)
    return kappa

from munkres import Munkres

def hungarian_matching(N):
    hungarian = Munkres()
    mapping = hungarian.compute(-N)
    total = 0
    map = np.zeros((len(mapping)),'i4')
    for row, column in mapping:
        map[row]=column
        value = N[row,column]
        total += value
        #print '(%d, %d) -> %d' % (row, column, value)
    return np.sum(np.diag(N[:,map]))/np.sum(N), kappa(N[:,map]), map

def hungarian_APC(N):
    '''
    Hungarian matching (APC: Lawler - implemented by G. CARPANETO, S. MARTELLO, P. TOTH)
    '''
    #print '... entering fortran binary'
    mapping, cost, errorcode  = hung_APC.apc(-N)
    if errorcode != 0:
        print 'APC error code %d: need to increase MAXSIZE in APC.f to handle this problem' % (errorcode)
    total=np.sum(np.diag(N[:,mapping-1]))
    if total != -cost:
        print 'cost %d and total %d unequal!' % (-cost,total)
    #total = -cost    
    #print 'mapping length', len(mapping), 'cost', cost, 'total', total
    #print 'percent agreements: ', 100*total/np.sum(N)
    
    return 100*total/np.sum(N), kappa(N[:,mapping-1]), mapping-1

def return_comparisons(c1,list1,c2,list2):
    #print '... creating %d by %d overlap matrix' % (len(c1),len(c2))
    N = overlap_matrix(c1,list1,c2,list2)
    #print '... calculating clustering comparison metrics'
    metrics = compare_classifications(N)

    results = {'Purity': (100.*(1.-(metrics['gini12']+metrics['gini21'])/2.)), \
               'RandomAccuracy': (100.*(1-(metrics['rce1']+metrics['rce2'])/2.)), \
               'PairsConcordancy': (100.*metrics['raw_concord']), \
               'Completeness': (100.*(metrics['completeness1']+metrics['completeness2'])/2.), \
               'Correctness': (100.*(metrics['correctness1']+metrics['correctness2'])/2.), \
               'MatchedAgreement':(metrics['matched_agree']), \
               'MatchedKappa': (100.*metrics['matched_kappa']),
               'MatchedMapping': metrics['matching']}
    return results
    
    #print 'Combine Gini Purity:                    %5.1f' % (100.*(1.-(metrics['gini12']+metrics['gini21'])/2.))
    #print 'Combined Random Classification Accuracy:%5.1f' % (100.*(1-(metrics['rce1']+metrics['rce2'])/2.))
    #print 'Pairs Classification Concordancy:       %5.1f' % (100.*metrics['raw_concord'])
    #print 'Completeness 1:                         ', cmpl1
    #print 'Correctness 1:                          ', corr1
    #print 'Completeness 2:                         ', cmpl2
    #print 'Combined Completeness:                  %5.1f' % (100.*(metrics['completeness1']+metrics['completeness2'])/2.)
    #print 'Correctness 2:                          ', corr2
    #print 'Combined Correctness:                   %5.1f' % (100.*(metrics['correctness1']+metrics['correctness2'])/2.)
    #print 'Hungarian Agreement:                    %5.1f' % (metrics['matched_agree'])
    #print 'Hungarian Kappa:                        %5.1f' % (100.*metrics['matched_kappa'])
    #print 'Mapping from C2 to C1', metrics['matching']

'''
'A= {0: {'indices': [0,1,2,3]}, 
    1: {'indices': [4,5,6,7]}, 
    2: {'indices': [8,9,10,11]}, 
    3: {'indices': [12,13,14,15]}
   }

B= {0: {'indices': [0,1,2]},
    1: {'indices': [3,4,5,6]},
    2: {'indices': [7,8,9]},
    3: {'indices': [10,11,12]},
    4: {'indices': [13,14,15]}
   }

M = np.array([[0,0],[0,0],[0,0],[0,1],[1,1],[1,1],[1,1],[1,2],
              [2,2],[2,2],[2,3],[2,3],[3,3],[3,4],[3,4],[3,4]])

s = 0
n = 0
for i in range(15):
    for j in range(i+1,16):
        if M[i,0]==M[j,0]:
            i1 = 1
        else:
            i1 = 0
        if M[i,1]==M[j,1]:
            i2 = 1
        else:
            i2 = 0
        s += np.abs(i1-i2)
        n += 1

print 's, n', s, n

pairs = 0

join0=0
split0=0
together0=0
apart0=0

join1=0
split1=0
together1=0
apart1=0

for i in range(15):
    for j in range(i+1,16):
        pairs += 1
        if M[i,0] == M[j,0]:
            together0 += 1
            if M[i,1] != M[j,1]:
                split0 += 1
        else:
            apart0 += 1
            if M[i,1] == M[j,1]:
                join0 += 1

        if M[i,1] == M[j,1]:
            together1 += 1
            if M[i,0] != M[j,0]:
                split1 += 1
        else:
            apart1 += 1
            if M[i,0] == M[j,0]:
                join1 += 1
            
print pairs, together0, split0, apart0, join0, together1, split1, apart1, join1
'''

'''
N = overlap_matrix(A,range(16),B,range(16))

print N

print compare_classifications(N)
'''

def multiple_comparisons(T,samplesize = 10**4, lscdist = 4., replications = 2,subj='1'):
    labels1 = np.random.permutation(np.arange(len(T)))[:samplesize]
    #labels1 = np.arange(0*samplesize,1*samplesize)
    T1=T[labels1]
    print 'Number of tracks is %d' % (len(T1))
    lists = {}
    C = {}
    results = {}
    C_size = {}

    for replication in np.arange(replications):
        print '... preparing LSC(%d)' % (replication)
        rearrangement = np.random.permutation(np.arange(samplesize))
        #print '... min %d, max %d, len %d' % (np.min(rearrangement), np.max(rearrangement), len(rearrangement))
        rearrangement = list(rearrangement)
        C[replication] = local_skeleton_clustering(T1[rearrangement],lscdist)
        print '... skeleton size %d' % (len(C[replication]))
        lists[replication] = rearrangement
        C_size[replication] = len(C[replication])

    save_pickle('labels'+str(samplesize)+'_'+subj+'.pkl',labels1)
    save_pickle('C'+str(samplesize)+'_'+subj+'.pkl',C)
    save_pickle('lists'+str(samplesize)+'_'+subj+'.pkl',lists)
    save_pickle('C_size'+str(samplesize)+'_'+subj+'.pkl',C_size)

    for rep1 in np.arange(replications-1):
        for rep2 in np.arange(rep1+1,replications):
            #report_comparisons(C[rep1],lists[rep1],C[rep2],lists[rep2])
            print 'comparing %d and %d' % (rep1,rep2)
            results[(rep1,rep2)] = return_comparisons(C[rep1],lists[rep1],C[rep2],lists[rep2])

    '''
    labels2 = np.arange(1*samplesize,2*samplesize)
    T2=T[labels2]
    list21 = np.random.permutation(np.arange(samplesize))
    C21=local_skeleton_clustering(T2[list21],lscdist)
    list22 = np.random.permutation(np.arange(samplesize))
    C22=local_skeleton_clustering(T2[list22],lscdist)
    '''
    #print 'C11 vs C12'
    '''
    print 'C21 vs C22'
    report_comparisons(C21,list21,C22,list22)
    '''
    return results

def find_matches():
    
    fsolid='/home/ian/Data/LSC_stability/solid_1M.npy'
    T=np.load(fsolid)

    samplesize = 10**3
    lscdist = 4.
    labels1 = np.arange(0*samplesize,1*samplesize)
    T1=T[labels1]
    C1=local_skeleton_clustering(T1,lscdist)

    labels2 = np.arange(1*samplesize,2*samplesize)
    T2=T[labels2]
    C2=local_skeleton_clustering(T2,lscdist)

    v1,l1,tot1 = bring_virtuals(C1)
    v2,l2,tot2 = bring_virtuals(C2)

    d12 = bundles_distances_mdf(v1,v2)        

    mv21 = np.argmin(d12,axis=0)
    
    print mv21[0], C2[0]['indices'], C1[mv21[0]]['indices']

'''
C11=taleton(T11,10*[4.])
C12=taleton(T12,10*[4.])

print len(C11), [len(b) for b in C11] 
print len(C12), [len(b) for b in C12] 
'''



'''
x = tracks2classes(C11[-1],i11)
y = tracks2classes(C12[-1],i12)

#print len(set(x.keys()).difference(y.keys()))

xdata = np.zeros(len(x.keys()))
ydata = np.zeros(len(y.keys()))

for i,k in enumerate(x.keys()):
    xdata[i]=x[k]
    ydata[i]=y[k]

#maxKappa(xdata,ydata)

#maxkappa, maxorder = maxKappa(xdata,ydata)
#print 'maxkappa   ', maxkappa
#print 'max order', maxorder
'''

"""
fsolid='/home/eg309/Data/LSC_limits/solid_1M.npy'
#fsolid='/home/ian/Data/LSC_stability/solid_1M.npy'

T=np.load(fsolid)
print 'Before',len(T)
T=np.array([t for t in list(T) if length(t)>= 40. and length(t)< 120.],dtype=np.object) # 100mm - 200mm
print 'After',len(T)

results = multiple_comparisons(samplesize=len(T), replications=3)
save_pickle('results_full.pkl', results)
"""

"""
results = multiple_comparisons(samplesize=25*10**3, replications=12)
save_pickle('results25k.pkl', results)

results = multiple_comparisons(samplesize=5*10**4, replications=12)
save_pickle('results50k.pkl', results)

results = multiple_comparisons(samplesize=10*10**4, replications=12)
save_pickle('results100k.pkl', results)
"""

