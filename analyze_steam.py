import os
from glob import glob
from time import time
import numpy as np
import dipy as dp
from dipy.reconst.dti import Tensor
from dipy.reconst.gqi import GeneralizedQSampling
from dipy.external.fsl import flirt2aff_files,warp_displacements
from itertools import chain

from subprocess import Popen,PIPE
from os.path import join as pjoin



from dipy.viz import fvtk
from dipy.viz import colormap

from dipy.io.pickles import save_pickle,load_pickle
import nibabel as nib
from dipy.tracking.propagation import EuDX
from dipy.tracking.vox2track import track_counts
from dipy.tracking.distances import track_roi_intersection_check
from dipy.io.dpy import Dpy
from dipy.tracking.metrics import length, downsample,intersect_sphere
from dipy.tracking.distances import local_skeleton_clustering, most_similar_track_mam
from scipy.ndimage import affine_transform,map_coordinates

import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from scipy.ndimage.morphology import generate_binary_structure, binary_erosion, distance_transform_cdt


dname='/home/eg309/Data/PROC_MR10032/'
fref = '/usr/share/fsl/data/standard/FMRIB58_FA_1mm.nii.gz'
fatlas='/home/eg309/Data/ICBM_Wmpm/ICBM_WMPM_eleftherios_padded.nii'


def pipe(cmd):    
    p = Popen(cmd, shell=True,stdout=PIPE,stderr=PIPE)
    sto=p.stdout.readlines()
    ste=p.stderr.readlines()
    print(sto)
    print(ste)

def dcm2nii(dname,outdir,filt='*.dcm',options='-d n -g n -i n -o'):
    cmd='dcm2nii '+options +' ' + outdir +' ' + dname + '/' + filt
    print(cmd)
    pipe(cmd)


def bet(in_nii,out_nii,options=' -F -f .2 -g 0'):
    cmd='bet '+in_nii+' '+ out_nii + options
    print(cmd)
    pipe(cmd)

    
def apply_warp(in_nii,affine_mat,nonlin_nii,out_nii):
    cmd='applywarp --ref=${FSLDIR}/data/standard/FMRIB58_FA_1mm --in='+in_nii+' --warp='+nonlin_nii+' --out='+out_nii
    print(cmd)
    pipe(cmd)
    
def load_img(fname):
    imgdata=nib.load(fname)
    data=imgdata.get_data()
    affine=imgdata.get_affine()
    return data,affine

def save_img(data,affine,fname):
    img=nib.Nifti1Image(data,affine)
    nib.save(img,fname)

def create_displacements(in_nii,affine_mat,nonlin_nii,invw_nii,disp_nii,dispa_nii):
    commands=[]    
    commands.append('flirt -ref ${FSLDIR}/data/standard/FMRIB58_FA_1mm -in '+in_nii+' -omat ' + affine_mat)
    commands.append('fnirt --in='+in_nii+' --aff='+affine_mat+' --cout='+nonlin_nii+' --config=FA_2_FMRIB58_1mm')
    commands.append('invwarp --ref='+in_nii+' --warp='+nonlin_nii+' --out='+invw_nii)
    commands.append('fnirtfileutils --in='+nonlin_nii+' --ref=${FSLDIR}/data/standard/FMRIB58_FA_1mm --out='+disp_nii)
    commands.append('fnirtfileutils --in='+nonlin_nii+' --ref=${FSLDIR}/data/standard/FMRIB58_FA_1mm --out='+dispa_nii + ' --withaff')
    for c in commands:
        print(c)
        pipe(c)

def write_tracks(fdpy,scalar,indices,seed_no=10**6,a_thr=.2,compression=1):
    eudx=EuDX(scalar,indices,seed_no=seed_no,a_low=a_thr)
    #exi=iter(eudx)       
    dpw=Dpy(fdpy,'w',compression=1)    
    #for (i,track) in enumerate(exi):
    for track in eudx:       
        dpw.write_track(track.astype(np.float32))
    dpw.close()

def read_warp_save_tracks(fdpy,ffa,fmat,finv,fdis,fdisa,fref,fdpyw):   
    
    #read the tracks from the image space 
    dpr=Dpy(fdpy,'r')
    T=dpr.read_tracks()
    dpr.close()    
    
    #copy them in a new file
    dpw=Dpy(fdpyw,'w',compression=1)
    dpw.write_tracks(T)
    dpw.close()
    
    #from fa index to ref index
    res=flirt2aff_files(fmat,ffa,fref)
    
    #load the reference img    
    imgref=nib.load(fref)
    refaff=imgref.get_affine()
    
    #load the invwarp displacements
    imginvw=nib.load(finv)
    invwdata=imginvw.get_data()
    invwaff = imginvw.get_affine()
    
    #load the forward displacements
    imgdis=nib.load(fdis)
    disdata=imgdis.get_data()
    
    #load the forward displacements + affine
    imgdis2=nib.load(fdisa)
    disdata2=imgdis2.get_data()
    
    #from their difference create the affine
    disaff=disdata2-disdata
    
    del disdata
    del disdata2
    
    shape=nib.load(ffa).get_data().shape
    
    #transform the displacements affine back to image space
    disaff0=affine_transform(disaff[...,0],res[:3,:3],res[:3,3],shape,order=1)
    disaff1=affine_transform(disaff[...,1],res[:3,:3],res[:3,3],shape,order=1)
    disaff2=affine_transform(disaff[...,2],res[:3,:3],res[:3,3],shape,order=1)
    
    #remove the transformed affine from the invwarp displacements
    di=invwdata[:,:,:,0] + disaff0
    dj=invwdata[:,:,:,1] + disaff1
    dk=invwdata[:,:,:,2] + disaff2    
    
    dprw=Dpy(fdpyw,'r+')
    rows=len(dprw.f.root.streamlines.tracks)   
    blocks=np.round(np.linspace(0,rows,10)).astype(int)#lets work in blocks
    print rows
    for i in range(len(blocks)-1):        
        print blocks[i],blocks[i+1]   
        #copy a lot of tracks together
        caboodle=dprw.f.root.streamlines.tracks[blocks[i]:blocks[i+1]]
        mci=map_coordinates(di,caboodle.T,order=1) #interpolations for i displacement
        mcj=map_coordinates(dj,caboodle.T,order=1) #interpolations for j displacement
        mck=map_coordinates(dk,caboodle.T,order=1) #interpolations for k displacement            
        D=np.vstack((mci,mcj,mck)).T
        #go back to mni image space                        
        WI2=np.dot(caboodle,res[:3,:3].T)+res[:3,3]+D
        #and then to mni world space
        caboodlew=np.dot(WI2,refaff[:3,:3].T)+refaff[:3,3]
        #write back       
        dprw.f.root.streamlines.tracks[blocks[i]:blocks[i+1]]=caboodlew.astype('f4')
    dprw.close()

def see_tracks(fdpy,N=2000):
    
    
    dpr=Dpy(fdpy,'r')
    #T=dpr.read_tracksi(range(N))
    T=dpr.read_tracks()
    dpr.close()    
    
    T=[downsample(t,5) for t in T]    

    r=fvtk.ren()
    colors=np.ones((len(T),3)).astype('f4')
    for (i,c) in enumerate(T):        
        orient=c[0]-c[-1]
        orient=np.abs(orient/np.linalg.norm(orient))
        colors[i,:3]=orient    
    fvtk.add(r,fvtk.line(T,colors,opacity=0.5))
    #fos.add(r,fos.sphere((0,0,0),10))
    fvtk.show(r)
    
def generate_lengths(fdpy,fnpy):
    dpr=Dpy(fdpy,'r')
    T=dpr.read_tracks()
    dpr.close()
    lenT=[length(t) for t in T]
    np.save(fnpy,np.array(lenT))    

def save_histogram(fnpy,fpng):        
    lengths=np.load(fnpy)
    binss=np.round(np.linspace(0, 400, 50)).astype(np.int)
    n, bins, patches = plt.hist(lengths, bins=binss, normed=True, facecolor='green', alpha=0.75)
    plt.ylim( (0, 0.025) )
    plt.xlabel('length(mm)')
    plt.ylabel('counts')
    plt.title('Histogram of lengths')
    #plt.axis([40, 160, 0, 0.03])
    #plt.grid(True)     
    plt.savefig(fpng)    
    plt.clf()
    
def get_roi(froi,no,erosion_level=1):
    imgroi=nib.load(froi)    
    roidata=imgroi.get_data()
    roiaff=imgroi.get_affine()            
    cross=generate_binary_structure(3,1)
    roidata2=roidata.copy()
    roidata2[roidata2!=no]=0    
    if erosion_level>0:
        roidata2=binary_erosion(roidata2,cross,erosion_level)                 
        I=np.array(np.where(roidata2==True)).T
    else:
        I=np.array(np.where(roidata2==no)).T    
    wI=np.dot(roiaff[:3,:3],I.T).T+roiaff[:3,3]
    wI=wI.astype('f4')
    return wI

def roi_intersection(fdpy,fatlas,roi_no,froidpy):    
    dpr=Dpy(fdpy,'r')
    T=dpr.read_tracksi(range(10000))    
    dpr.close()    
    Troi=[]
    wI=get_roi(fatlas,roi_no,0)
    for (i,t) in enumerate(T):
        if i%1000==0:
            print i
        if track_roi_intersection_check(t,wI,.5):
            Troi.append(t)
    print(len(Troi))    
    dpw=Dpy(froidpy,'w')
    dpw.write_tracks(Troi)
    dpw.close()
    
    '''
    from dipy.viz import fvtk
    r=fvtk.ren()
    fvtk.add(r,fvtk.line(Troi,fvtk.red))
    fvtk.add(r,fvtk.point(wI,fvtk.green))
    fvtk.show(r)
    '''
    
def roi_track_counts(fdpy,fref,fatlas,roi_no,dist_transf=True,fres=None):
    
    dpr=Dpy(fdpy,'r')
    T=dpr.read_tracks()
    dpr.close()
    
    img=nib.load(fref)
    affine=img.get_affine()
    zooms = img.get_header().get_zooms()
    iaffine=np.linalg.inv(affine)
    T2=[]
    #go back to volume space
    for t in T:
        T2.append(np.dot(t,iaffine[:3,:3].T)+iaffine[:3,3])
    del T
    
    tcs,tes=track_counts(T2,img.get_shape(),zooms,True)
    
    atlas_img=nib.load(fatlas)
    atlas=atlas_img.get_data()
    roi=atlas.copy()
    roi[atlas!=roi_no]=0
    
    if dist_transf:
        roi2=distance_transform_cdt(roi)
        roi[roi2!=roi2.max()]=0
        I=np.array(np.where(roi==roi_no)).T
    else:
        I=np.array(np.where(roi==roi_no)).T    
    
    """    
    if erosion_level>0:
        roi2=binary_erosion(roi,cross,erosion_level)                 
        I=np.array(np.where(roi2==True)).T
    else:        
        roi2=distance_transform_cdt(roi)        
        I=np.array(np.where(roi==roi_no)).T        
    """
    
    #print I.shape    
    #nib.save(nib.Nifti1Image(roi2,affine),'/tmp/test.nii.gz')
    
    Ttes=[]
    for iroi in I:
        try:
            Ttes.append(tes[tuple(iroi)])
        except KeyError:
            pass
    
    Ttes=list(set(list(chain.from_iterable(Ttes))))    
    T2n=np.array(T2,dtype=np.object)
    res=list(T2n[Ttes])
    
    #back to world space
    res2=[]
    for t in res:
        res2.append(np.dot(t,affine[:3,:3].T)+affine[:3,3])
    np.save(fres,np.array(res2,dtype=np.object))
    
"""
Find common rows
-----------------

voxels=np.random.rand(10,3)
Z=voxels.view([('a',int),('b',int),('c',int)])
Z.shape
Zs = np.unique(Z).view((int,3))
"""
    
def skeletonize(fdpy,flsc,points=3):

    dpr=Dpy(fdpy,'r')    
    T=dpr.read_tracks()
    dpr.close()    
    print len(T)
    Td=[downsample(t,points) for t in T]
    C=local_skeleton_clustering(Td,d_thr=10.,points=points)    
    #Tobject=np.array(T,dtype=np.object)
    

    #'''
    #r=fvtk.ren()    
    skeleton=[]    
    for c in C:
        #color=np.random.rand(3)
        if C[c]['N']>0:
            Ttmp=[]
            for i in C[c]['indices']:
                Ttmp.append(T[i])
            si,s=most_similar_track_mam(Ttmp,'avg')
            print si,C[c]['N']    
            C[c]['most']=Ttmp[si]            
            #fvtk.add(r,fvtk.line(Ttmp[si],color))            
    print len(skeleton)
    #r=fos.ren()
    #fos.add(r,fos.line(skeleton,color))    
    #fos.add(r,fos.line(T,fos.red))    
    #fvtk.show(r)
    #'''
    
    save_pickle(flsc,C)
    


def track_counts_all(fdpyw,fref,fatlas,local):
    
    print('-- Calculate BCC tracks')    
    ffaBCC = pjoin(local,'FAW_BCC.npy')
    roi_track_counts(fdpyw,fref,fatlas,4,False,ffaBCC)
    
    print('-- Calculate GCC tracks')
    ffaGCC = pjoin(local,'FAW_GCC.npy')
    roi_track_counts(fdpyw,fref,fatlas,3,False,ffaGCC)
    
    print('-- Calculate SCC tracks')
    ffaSCC = pjoin(local,'FAW_SCC.npy')
    roi_track_counts(fdpyw,fref,fatlas,5,False,ffaSCC)
    
    print('-- Calculate CST-R tracks')
    ffaCSTR = pjoin(local,'FAW_CST-R.npy')
    roi_track_counts(fdpyw,fref,fatlas,8,False,ffaCSTR) #it is 7 in the atlas docs
    
    print('-- Calculate CST-L tracks')
    ffaCSTL = pjoin(local,'FAW_CST-L.npy')
    roi_track_counts(fdpyw,fref,fatlas,9,False,ffaCSTL) #it is 8 in the atlas docs
    
    print('-- Calculate UNC-R tracks')
    ffaUNCR = pjoin(local,'FAW_UNC-R.npy')
    roi_track_counts(fdpyw,fref,fatlas,47,False,ffaUNCR) #it is 8 in the atlas docs
    
    print('-- Calculate UNC-L tracks')
    ffaUNCL = pjoin(local,'FAW_UNC-L.npy')
    roi_track_counts(fdpyw,fref,fatlas,48,False,ffaUNCL) #it is 8 in the atlas docs
        
    print('All done.')

def skeleton2tracks(fskel):
    
    C=load_pickle(fskel)
    tracks=[C[c]['most'] for c in C]
    
    return tracks


def get_roi_new(roi_no,dist_transf=False): 

    atlas_img=nib.load(fatlas)
    atlas=atlas_img.get_data()
    roiaff=atlas_img.get_affine()
    roi=atlas.copy()
    roi[atlas!=roi_no]=0
    
    if dist_transf:
        roi2=distance_transform_cdt(roi)
        roi[roi2!=roi2.max()]=0
        I=np.array(np.where(roi==roi_no)).T
    else:
        I=np.array(np.where(roi==roi_no)).T
        
    wI=np.dot(roiaff[:3,:3],I.T).T+roiaff[:3,3]
    wI=wI.astype('f4')
    
    return wI

def see_skeletons(fskel):
    
    C=load_pickle(fskel)
    tracks=[C[c]['most'] for c in C if C[c]['N'] > 10 ]
    
    r=fvtk.ren()    
    colors=np.array([t[0]-t[-1] for t in tracks])
    colors=colormap.orient2rgb(colors)
    fvtk.add(r,fvtk.line(tracks,colors))
    
    fvtk.show(r)
    

def create_FA_displacements_warp_FAs(fname,dname):

    fbvals=fname+'.bval'
    fbvecs=fname+'.bvec'       
    fdata=fname+'.nii.gz'
    
    if os.path.isfile(fdata):
        pass
    else:
        fdata=fname+'.nii'
        if os.path.isfile(fdata)==False:
            print('Data do not exist')
            return
     
    dti_dname=os.path.join(dname,'DTI')    
    if os.path.isdir(dti_dname):
        pass
    else:
        os.mkdir(dti_dname)
        
    local=dti_dname
    print('-------------------------')
    print('----Working with DTI-----')
    print('-------------------------')
    print local
    
    print('1.Remove the sculp using bet')
    fdatabet=fname+'_bet.nii.gz'
    bet(fdata,fdatabet)
    
    print('2.Load data and save S0')
    data,affine=load_img(fdatabet)
    fs0=pjoin(local,'S0_bet.nii.gz')
    #print fs0
    save_img(data[...,0],affine,fs0)
    
    print('3.Create Tensors and save FAs')
    bvals=np.loadtxt(fbvals)
    gradients=np.loadtxt(fbvecs).T
    ten=Tensor(data,bvals,gradients,thresh=50)
    ffa=pjoin(local,'FA_bet.nii.gz')
    print ffa
    save_img(ten.fa(),affine,ffa)
    fmd=pjoin(local,'MD_bet.nii.gz')
    #print fmd
    save_img(ten.md(),affine,fmd)
    
    print('4.Create the displacements using fnirt')
    fmat=pjoin(local,'flirt.mat')
    fnon=pjoin(local,'fnirt.nii.gz')
    finv=pjoin(local,'invw.nii.gz')
    fdis=pjoin(local,'dis.nii.gz')
    fdisa=pjoin(local,'disa.nii.gz')
    create_displacements(ffa,fmat,fnon,finv,fdis,fdisa)
    
    print('5.Warp FA')
    ffaw=pjoin(local,'FAW_bet.nii.gz')
    apply_warp(ffa,fmat,fnon,ffaw)
    ##warp_displacements(ffa,fmat,fdis,fref,ffaw2,order=1)
    
    print('6.Warp S0')
    fs0w=pjoin(local,'S0W_bet.nii.gz')
    apply_warp(fs0,fmat,fnon,fs0w)
    ##warp_displacements(fs0,fmat,fdis,fref,fs0w2,order=1)
    
    print('7.Warp MD')
    fmdw=pjoin(local,'MDW_bet.nii.gz')
    apply_warp(fmd,fmat,fnon,fmdw)
    ##warp_displacements(fmd,fmat,fdis,fref,fmdw2,order=1)

def generate_gqi_tracks_and_warp_in_MNI_space(fname,dname):

    fbvals=fname+'.bval'
    fbvecs=fname+'.bvec'       
    fdata=fname+'.nii.gz'
    
    if os.path.isfile(fdata):
        pass
    else:
        fdata=fname+'.nii'
        if os.path.isfile(fdata)==False:
            print('Data do not exist')
            return
     
    dti_dname=os.path.join(dname,'DTI')    
    if os.path.isdir(dti_dname):
        pass
    else:
        os.mkdir(dti_dname)    
    print dti_dname
    
    gqi_dname=os.path.join(dname,'GQI')    
    if os.path.isdir(gqi_dname):
        pass
    else:
        os.mkdir(gqi_dname)    
    print gqi_dname
    
    fdatabet=fname+'_bet.nii.gz'
    
    if os.path.isfile(fdatabet):
        pass
    else:
        print('fdatabet does not exist')
    
    img=nib.load(fdatabet)
    data=img.get_data()
    affine=img.get_affine()
    bvals=np.loadtxt(fbvals)
    bvecs=np.loadtxt(fbvecs).T
    
    gqs=GeneralizedQSampling(data,bvals,bvecs)
    eu=EuDX(gqs.qa(),gqs.ind(),seeds=10**6,a_low=0.0239)
    
    fdpy=pjoin(gqi_dname,'lsc_QA.dpy')
    dpw=Dpy(fdpy,'w',compression=1)
    for track in eu:
        dpw.write_track(track.astype(np.float32))
    dpw.close()
    
    local=dti_dname
    
    fmat=pjoin(local,'flirt.mat')
    fnon=pjoin(local,'fnirt.nii.gz')
    finv=pjoin(local,'invw.nii.gz')
    fdis=pjoin(local,'dis.nii.gz')
    fdisa=pjoin(local,'disa.nii.gz')
    ffa=pjoin(local,'FA_bet.nii.gz')
    
    fdpyw=pjoin(gqi_dname,'lsc_QA_ref.dpy')
    
    #print fdatabet
    #print fmat
    #print fnon
    print fdpy
    print fdpyw
        
    read_warp_save_tracks(fdpy,ffa,fmat,finv,fdis,fdisa,fref,fdpyw)
    
    
    

    
def dti_tracking_analysis():
    
    """    
    print('8.Calculate FA tracks')
    ffadpy=pjoin(local,'FAW_img.dpy')
    write_tracks(ffadpy,ten.fa(),ten.ind(),seed_no=10**6,a_thr=.2,compression=1)
      
    print('9.Read warp and write tracks')
    ffadpyw=pjoin(local,'FAW_ref.dpy')
    read_warp_save_tracks(ffadpy,ffa,fmat,finv,fdis,fdisa,fref,ffadpyw)
    
    print('10.See result')
    #see_tracks(ffadpyw)
    
    print('11.Calculate lengths')
    ffalenw = pjoin(local,'FAW_len.npy')
    generate_lengths(ffadpyw,ffalenw)
    
    print('12.Save histogram')
    ffalenwpng = pjoin(local,'FAW_len.png')
    save_histogram(ffalenw,ffalenwpng)
    
    print('13.Create skeleton')
    ffalscw=pjoin(local,'FAW_LSC_ref_3.pkl')
    skeletonize(ffadpyw,ffalscw,3)
    
    ffalscw6=pjoin(local,'FAW_LSC_ref_6.pkl')
    skeletonize(ffadpyw,ffalscw6,6)
    
    ffalscw9=pjoin(local,'FAW_LSC_ref_9.pkl')
    skeletonize(ffadpyw,ffalscw9,9)
    
    ffalscw12=pjoin(local,'FAW_LSC_ref_12.pkl')
    skeletonize(ffadpyw,ffalscw12,12)
    
    print('14.Intersections with spherical rois')    
    ##track_counts_all(ffadpyw,fref,fatlas,local)
    fsr=pjoin(local,'FAW_SR.pkl')
    spherical_rois(ffadpyw,fsr,sq_radius=4)
    """
    
    
    
    
def spherical_rois(fdpy,fsr,sq_radius=4):    
    
    
    R=atlantic_points()    
    dpr=Dpy(fdpy,'r')
    T=dpr.read_tracks()
    dpr.close()
    
    center=R['BCC']
    
    refimg=nib.load(fref)
    aff=refimg.get_affine()
    
    SR={}
    
    for key in R:
        
        center=R[key]
        #back to world space
        centerw=np.dot(aff,np.array(center+(1,)))[:3]        
        centerw.shape=(1,)+centerw.shape   
        centerw=centerw.astype(np.float32)
    
        res= [track_roi_intersection_check(t,centerw,sq_radius) for t in T]
        res= np.array(res,dtype=np.int)
        ind=np.where(res>0)[0]
        
        SR[key]={}
        SR[key]['center']=center
        SR[key]['centerw']=tuple(np.squeeze(centerw))
        SR[key]['radiusw']=np.sqrt(sq_radius)
        SR[key]['indices']=ind
        
    
    save_pickle(fsr,SR)
    
def see_spherical_intersections(fdpy,fsr):
    
    dpr=Dpy(fdpy,'r')
    T=dpr.read_tracks()
    dpr.close()
    
    SR=load_pickle(fsr)
    
    r=fvtk.ren()
    
    for key in SR:
        ind=SR[key]['indices']
        intersT=[T[i] for i in ind]
        fvtk.add(r,fvtk.line(intersT,np.random.rand(3)))    
        centerw=SR[key]['centerw']
        radius=SR[key]['radiusw']
        fvtk.add(r,fvtk.sphere(position=centerw,radius=radius))
        
    fvtk.show(r)
    

def atlantic_points(dic=True):
    
    f=open('/home/eg309/Devel/tractarian/devel/scripts/Atlantic_Points.txt','r')
    lines=f.readlines()
    
    if dic==False:
        
        S=[]
        for l in lines:
            s=l.split()[2:5]
            for sp in s:            
                S.append(float(sp))
        
        f.close()
        return np.array(S).reshape(len(S)/3,3)
    
    if dic==True:
        
        R={}
        for l in lines:
            
            s=l.split()
            if len(s)>0:
                print s
                s0=float(s[2])
                s1=float(s[3])
                s2=float(s[4])
                R[s[1]]=(s0,s1,s2)
                
        return R

def dcm2nii_all(type):
    dname2='/home/eg309/Data/MR10032_32ch'
    cnt =0
    for root, dirs, files in os.walk(dname2):
        if root.endswith(type):
            #print cnt, root
            #cnt+=1
            #print root
            for file in files:
                if file.endswith('.dcm'):
                    #print root
                    #print file                    
                    dcm2nii(dname=root,outdir=root)
                    break


def prepare():
    for root, dirs, files in os.walk(dname):
        #if root.endswith('101_32'):
        for file in files:
            if file.endswith('.bval'):       
                fname=os.path.join(root,file)
                print fname
                #dti_analyze(fname.split('.bval')[0],root)
                create_FA_displacements_warp_FAs(fname.split('.bval')[0],root)
                #return
                
def prepare_gqi_101_32():        
    for root, dirs, files in os.walk(dname):
        
        #if root.endswith('subj_07'): #problem found with this subject reasons not known yet
        #    continue
        
        if root.endswith('subj_07/101_32')==False and root.endswith('subj_01/101_32')==False:
            if root.endswith('101_32'): 
                print root
                #"""
                for file in files:
                    if file.endswith('.bval'):
                        fname=os.path.join(root,file)
                        #print fname
                        #dti_analyze(fname.split('.bval')[0],root)
                        #create_FA_displacements_warp_FAs(fname.split('.bval')[0],root)
                        print fname
                        generate_gqi_tracks_and_warp_in_MNI_space(fname.split('.bval')[0],root)
                #"""
        

def generate_cumulatives():
    subjs=['subj_01','subj_02','subj_03','subj_04','subj_05','subj_06','subj_07','subj_08','subj_09','subj_10','subj_11','subj_12']
    bins=np.round(np.linspace(0, 400, 50)).astype(np.int)
    
    for sub in subjs:
        lengthsFA={}
        lengthsQA={}    
        for root, dirs, files in os.walk(dname+sub):
            #print root
            if root.endswith('64'):
                print root
                for file in files:
                    if file.endswith('_FA_warp_lengths.npy'):
                        fname=os.path.join(root,file)
                        lengthsFA['64']=np.load(fname)
                    if file.endswith('_QA_warp_lengths.npy'):
                        fname=os.path.join(root,file)
                        lengthsQA['64']=np.load(fname)                       
            
            if root.endswith('101'):
                print root
                for file in files:
                    if file.endswith('_FA_warp_lengths.npy'):
                        fname=os.path.join(root,file)
                        lengthsFA['101']=np.load(fname)
                    if file.endswith('_QA_warp_lengths.npy'):
                        fname=os.path.join(root,file)
                        lengthsQA['101']=np.load(fname)

            if root.endswith('118'):
                print root
                for file in files:
                    if file.endswith('_FA_warp_lengths.npy'):
                        fname=os.path.join(root,file)
                        lengthsFA['118']=np.load(fname)
                    if file.endswith('_QA_warp_lengths.npy'):
                        fname=os.path.join(root,file)
                        lengthsQA['118']=np.load(fname)                
        
        dcols=[['64','red'],['101','blue'],['118','green']]
        for d in dcols:
            n, bins, patches = plt.hist(lengthsFA[d[0]], bins=bins, normed=True, cumulative=True, facecolor='none', edgecolor=d[1], alpha=0.75,label=d[0])
        
        plt.ylim( (0, 1) )
        plt.xlabel('length(mm)')
        plt.ylabel('cumulative')
        plt.legend()
        plt.title(sub)        
        plt.grid(True)
        fname2=dname+'sumFA/'+sub+'_FA_cumul.png'
        print fname2
        plt.savefig(fname2)
        plt.clf()
        
        dcols=[['64','red'],['101','blue'],['118','green']]
        for d in dcols:
            n, bins, patches = plt.hist(lengthsQA[d[0]], bins=bins, normed=True, cumulative=True, facecolor='none', edgecolor=d[1], alpha=0.75,label=d[0])
        
        plt.ylim( (0, 1) )
        plt.xlabel('length(mm)')
        plt.ylabel('cumulative')
        plt.legend()
        plt.title(sub)        
        plt.grid(True)
        fname2=dname+'sumQA/'+sub+'_QA_cumul.png'
        print fname2
        plt.savefig(fname2)
        plt.clf()

            
def combine_results():
    
    subs=['subj_01','subj_02','subj_03','subj_04','subj_05','subj_06','subj_07','subj_08','subj_09','subj_10','subj_11','subj_12']
    categ=['64','64_32','101','101_32','118','118_32']
    methods=['DTI','GQI','SDI','NPA']
    
    RES={}
    for sub in subs:
        RES[sub]={}
        for cat in categ:
            RES[sub][cat]={}
            for meth in methods:
                RES[sub][cat][meth]={}                                
                for root, dirs, files in os.walk(dname+sub+'/'+cat+'/'+meth):
                    for file in files:
                        if file.endswith('FAW_len.npy'):
                            print pjoin(root,file)
                            track_lengths=np.load(pjoin(root,file))
                            RES[sub][cat][meth]['track_no']=len(track_lengths)
                            RES[sub][cat][meth]['total_length']=np.sum(track_lengths)
                            #RES[sub][cat][meth]['lenghts']=track_lengths
    
                        if file.startswith('FAW_LSC_ref_'):                                
                            if file.endswith('3.pkl'):
                                RES[sub][cat][meth]['len_lsc_3']=len(load_pickle(pjoin(root,file)))                                    
                            if file.endswith('6.pkl'):
                                RES[sub][cat][meth]['len_lsc_6']=len(load_pickle(pjoin(root,file)))                                    
                            if file.endswith('9.pkl'):
                                RES[sub][cat][meth]['len_lsc_9']=len(load_pickle(pjoin(root,file)))                                    
                            if file.endswith('12.pkl'):
                                RES[sub][cat][meth]['len_lsc_12']=len(load_pickle(pjoin(root,file)))
                                
                        if file.endswith('_SR.pkl'):
                            
                            RES[sub][cat][meth]['sr']=load_pickle(pjoin(root,file))
    
    return RES

def see_combined_results_A():
    
    subs=['subj_01','subj_02','subj_03','subj_04','subj_05','subj_06','subj_07','subj_08','subj_09','subj_10','subj_11','subj_12']
    categ=['64','64_32','101','101_32','118','118_32']
    methods=['DTI','GQI','SDI','NPA']
    
    RES=load_pickle('/home/eg309/Data/PROC_MR10032/results/res_tmp.pkl')
    
    print RES['subj_03']['64_32']['DTI']
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    counts={}
    tlengths={}
    skel3={}
    skel6={}
    skel9={}
    skel12={}
    for cat in categ:        
        
        counts[cat]=[]
        tlengths[cat]=[]
        
        skel3[cat]=[]
        skel6[cat]=[]
        skel9[cat]=[]
        skel12[cat]=[]
                                
        for sub in subs:
            try:        
                counts[cat].append(RES[sub][cat]['DTI']['track_no'])
                tlengths[cat].append(RES[sub][cat]['DTI']['total_length'])
                skel3[cat].append(RES[sub][cat]['DTI']['len_lsc_3'])
                skel6[cat].append(RES[sub][cat]['DTI']['len_lsc_6'])
                skel9[cat].append(RES[sub][cat]['DTI']['len_lsc_9'])
                skel12[cat].append(RES[sub][cat]['DTI']['len_lsc_12'])
                
            except:
                pass
    
    
    mean_track_no=[]
    std_track_no=[]
    
    mean_tlengths=[]
    std_tlengths=[]
    
    mean_skel3=[]
    std_skel3=[]

    mean_skel6=[]
    std_skel6=[]
    
    mean_skel9=[]
    std_skel9=[]
    
    mean_skel12=[]
    std_skel12=[]
     
    category=[]                       
    for cat in categ:
        mean_track_no.append(np.array(counts[cat]).mean())
        std_track_no.append(np.array(counts[cat]).std())
        
        mean_tlengths.append(np.array(tlengths[cat]).mean())
        std_tlengths.append(np.array(tlengths[cat]).std())
        
        mean_skel3.append(np.array(skel3[cat]).mean())
        std_skel3.append(np.array(skel3[cat]).std())

        mean_skel6.append(np.array(skel6[cat]).mean())
        std_skel6.append(np.array(skel6[cat]).std())
        
        mean_skel9.append(np.array(skel9[cat]).mean())
        std_skel9.append(np.array(skel9[cat]).std())
        
        mean_skel12.append(np.array(skel12[cat]).mean())
        std_skel12.append(np.array(skel12[cat]).std())
        
        
        category.append(cat)
    
    print mean_track_no
    print std_track_no
    print category
    
    width = 0.35
    ind=np.arange(6)
    
    rects1 = ax.bar(ind+width/2., mean_track_no, width, color='r', yerr= std_track_no)
    #rects2 = ax.bar(ind+width, mean_track_no, width, color='y', yerr= std_track_no)
           
    ax.set_xticks(ind+width)    
    ax.set_xticklabels(categ)    
    ax.legend( (rects1[0],), ('Track No',) )    
    plt.show()
    
    fig = plt.figure()
    ax2 = fig.add_subplot(111)    
    rects2 = ax2.bar(ind+width/2., mean_tlengths, width, color='y', yerr= std_tlengths)    
    ax2.set_xticks(ind+width)    
    ax2.set_xticklabels(categ)    
    ax2.legend( (rects2[0],), ('Total length',) )    
    plt.show()
    
    fig = plt.figure()
    
    width=0.2
    
    ax3 = fig.add_subplot(111)    
    rects3 = ax3.bar(ind, mean_skel3, width, color='y', yerr= std_skel3)
    rects4 = ax3.bar(ind+width, mean_skel6, width, color='r', yerr= std_skel6)
    rects5 = ax3.bar(ind+2*width, mean_skel9, width, color='b', yerr= std_skel9)
    rects6 = ax3.bar(ind+3*width, mean_skel12, width, color='g', yerr= std_skel12)
            
    ax3.set_xticks(ind+2*width)    
    ax3.set_xticklabels(categ)    
    ax3.legend( (rects3[0],rects4[0],rects5[0],rects6[0]), ('3','6','9','12') )    
    plt.show()
    

def see_combined_spherical_intersections():
    
    subs=['subj_01','subj_02','subj_03','subj_04','subj_05','subj_06','subj_07','subj_08','subj_09','subj_10','subj_11','subj_12']
    categ=['64','64_32','101','101_32','118','118_32']
    methods=['DTI','GQI','SDI','NPA']
    
    centers=['GCC', 'CSTL', 'FX', 'CGCR', 'SCC', 'BCC', 'CGCL', 'UNCL?', 'CSTR', 'UNCR?']
    
    
    RES=load_pickle('/home/eg309/Data/PROC_MR10032/results/res_tmp.pkl')    
    #print RES['subj_03']['64_32']['DTI'] 
    
    SR_combined={}
    for cent in centers:
        SR_combined[cent]={}
        for cat in categ:
            SR_combined[cent][cat]=[]
            for sub in subs:
                try:
                    SR_combined[cent][cat].append(len(RES[sub][cat]['DTI']['sr'][cent]['indices']))
                except KeyError:
                    pass
    
    #return SR_combined
    
    #RES['subj_03']['64_32']['DTI']['sr']['GCC']['indices']
    width=0.2
    for cent in SR_combined:
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_title(cent)
        
        mean_=[]
        std_=[]
        ind=np.arange(6)
        for cat in categ:            
            mean_.append(np.array(SR_combined[cent][cat]).mean())
            std_.append(np.array(SR_combined[cent][cat]).std())
            
        rects = ax.bar(ind+width, mean_ , width, color='m', yerr= std_)
        ax.set_xticks(ind+width)    
        ax.set_xticklabels(categ)
        plt.show()
        
    """
    fig = plt.figure()
    
    width=0.2
    
    ax = fig.add_subplot(111)    
    rects3 = ax.bar(ind, mean_skel3, width, color='y', yerr= std_skel3)
    rects4 = ax.bar(ind+width, mean_skel6, width, color='r', yerr= std_skel6)
    rects5 = ax.bar(ind+2*width, mean_skel9, width, color='b', yerr= std_skel9)
    rects6 = ax.bar(ind+3*width, mean_skel12, width, color='g', yerr= std_skel12)
            
    ax.set_xticks(ind+2*width)
    ax.set_xticklabels(categ)
    ax.legend( (rects3[0],rects4[0],rects5[0],rects6[0]), ('3','6','9','12') )
    plt.show()
    """
    
    


    
def rename_S0():
    
    for root, dirs, files in os.walk(dname):        
        for file in files:
            if file.startswith('_S0_bet.nii.gz'):
                current=pjoin(root,file)
                new=pjoin(root,'S0_bet.nii.gz')
                #print current
                #print new                
                #os.rename(current, new)
                        
    #return counts
    
    

def barchart():    

    
    N = 5
    menMeans = (20, 35, 30, 35, 27)
    menStd =   (2, 3, 4, 1, 2)
    
    ind = np.arange(N)  # the x locations for the groups
    width = 0.35       # the width of the bars
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    rects1 = ax.bar(ind, menMeans, width, color='r', yerr=menStd)
    
    womenMeans = (25, 32, 34, 20, 25)
    womenStd =   (3, 5, 2, 3, 3)
    rects2 = ax.bar(ind+width, womenMeans, width, color='y', yerr=womenStd)
    
    # add some
    ax.set_ylabel('Scores')
    ax.set_title('Scores by group and gender')
    ax.set_xticks(ind+width)
    ax.set_xticklabels( ('G1', 'G2', 'G3', 'G4', 'G5') )
    
    ax.legend( (rects1[0], rects2[0]), ('Men', 'Women') )
    
    def autolabel(rects):
        # attach some text labels
        for rect in rects:
            height = rect.get_height()
            ax.text(rect.get_x()+rect.get_width()/2., 1.05*height, '%d'%int(height),
                    ha='center', va='bottom')
    
    autolabel(rects1)
    autolabel(rects2)
    
    plt.show()

    #plt.savefig('/tmp/test.png')    
    #plt.clf()





            
"""
TODO

1. Draw spherical rois in the FA template

2. Picking some tracks

3. Matching skeletons

4. Spherical Diffusivity

5. Boundary problems with EuDX

6. Increase number of seeds

"""

if __name__ == "__main__":
    pass
    #t1=time()
    #prepare()
    #t2=time()
    #print("Done in %.2f s." % (t2-t1))
    #analyze2('/home/eg309/Data/PROC_MR10032/subj_01/64/1312211075232351192010091419353265314888418CBUDTI64InLea2x2x2s006a001')
    #new_data('CBU_DTI_64InLea_2x2x2')
    #new_data('advdiff_DSI_101_25x25x25_STEAM')
    #new_data('advdiff_DTI_25x25x25_STEAM_118dr')
    #new_data('MPRAGE')
    #combine_results()
    #see_combined_results()
