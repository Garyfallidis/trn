import sys
import numpy as np
import nibabel as nib
from time import time
from dipy.reconst.dti import Tensor
from dipy.reconst.dni import EquatorialInversion
from dipy.reconst.gqi import GeneralizedQSampling
from dipy.reconst.dsi import DiffusionSpectrum
from dipy.segment.quickbundles import QuickBundles
from dipy.tracking.eudx import EuDX
from dipy.io.dpy import Dpy
from dipy.tracking.metrics import length
from dipy.external.fsl import pipe,flirt2aff
from fos.actor.slicer import Slicer
from fos.actor.point import Point
from fos import Actor,World, Window, WindowManager
from labeler import TrackLabeler
from dipy.tracking.vox2track import track_counts
from dipy.io.pickles import load_pickle, save_pickle
from dipy.tracking.distances import bundles_distances_mdf, bundles_distances_mam
from fos.actor.line import Line
from eudx_results import show_tracks



def expand_seeds(seeds,no=5,width=1):
    lseeds=len(seeds)
    seeds2=np.zeros((no*lseeds,3))
    for i in range(no):
        seeds2[i*lseeds:(i+1)*lseeds]=seeds + width*np.random.rand(lseeds,3) - width*0.5
    return seeds2

def transform_tracks(tracks,affine):
    return [(np.dot(affine[:3,:3],t.T).T + affine[:3,3]) for t in tracks]

def get_seeds(mask,no=50,width=1):
    mask[mask>0]=1
    mask=mask.astype(np.uint8)
    seeds=np.where(mask>0)
    seeds=np.array(seeds).T
    seeds=seeds.astype(np.float64)
    seeds=expand_seeds(seeds,no,width)
    return seeds

def track_range(t,a,b):
    lt=length(t)
    if lt>a and lt<b:
        return True
    return False

def rotate_camera(w):
    cam=w.get_cameras()[0]
    cam.cam_rot.rotate(-90,1,0,0.)
    cam.cam_rot.rotate(-90,0,1,0.)

def is_close(t,lT,thr):
    res=bundles_distances_mam([t.astype('f4')],lT,'min')
    if np.sum(res<thr)>=1:
        return True
    return False




if __name__=='__main__':


    dname='/home/eg309/Data/orbital_phantoms/dwi_dir/subject1/'
    fsl_ref = '/usr/share/fsl/data/standard/FMRIB58_FA_1mm.nii.gz'
    img_ref =nib.load(fsl_ref)
    ffa='data/subj_01/101_32/DTI/fa.nii.gz'
    fmat='data/subj_01/101_32/DTI/flirt.mat'
    img_fa =nib.load(ffa)
    img_ref =nib.load(fsl_ref)
    ref_shape=img_ref.get_data().shape
    mat=flirt2aff(np.loadtxt(fmat),img_fa,img_ref)

    ftracks_dti=dname+'dti_tracks.dpy'
    ftracks_dsi=dname+'dsi_tracks.dpy'
    ftracks_gqi=dname+'gqi_tracks.dpy'
    ftracks_eit=dname+'eit_tracks.dpy'

    ftracks=[ftracks_dti,ftracks_dsi,ftracks_gqi,ftracks_eit]

    #load data
    fraw=dname+'data.nii.gz'
    fbval=dname+'bvals'
    fbvec=dname+'bvecs'
    img = nib.load(fraw)
    data = img.get_data()
    affine = img.get_affine()
    bvals = np.loadtxt(fbval)
    gradients = np.loadtxt(fbvec).T

    print 'Data shape',data.shape
    t=time()
    #calculate FA
    tensors = Tensor(data, bvals, gradients, thresh=50)
    FA = tensors.fa()
    print 'ten',time()-t
    famask=FA>=.2

    maskimg=nib.Nifti1Image(famask.astype(np.uint8),affine)
    nib.save(maskimg,dname+'mask.nii.gz')

    #create seeds for first mask
    seeds=get_seeds(famask,3,1)
    print 'no of seeds', len(seeds)

    t=time()
    #GQI
    gqs=GeneralizedQSampling(data,bvals,gradients,1.2,
                odf_sphere='symmetric642',
                mask=famask,
                squared=False,
                save_odfs=False)
    print 'GQS',time()-t

    t=time()
    #EIT
    ei=EquatorialInversion(data,bvals,gradients,odf_sphere='symmetric642',\
            mask=famask,\
            half_sphere_grads=True,\
            auto=False,\
            save_odfs=False,\
            fast=True)
    ei.radius=np.arange(0,5,0.4)
    ei.gaussian_weight=0.05
    ei.set_operator('laplacian')
    ei.update()
    ei.fit()

    print 'EITL', time()-t    
    t=time()
    #DSI
    ds=DiffusionSpectrum(data,bvals,gradients,\
                odf_sphere='symmetric642',\
                mask=famask,\
                half_sphere_grads=True,\
                auto=True,\
                save_odfs=False)
    print 'DSI', time()-t
    #ds.PK[FA<.2]=np.zeros(5)
    
    t=time()
    euler=EuDX(a=FA,ind=tensors.ind(),seeds=seeds,a_low=.2)
    T=[track for track in euler] 
    print 'Eudx ten',time()-t,len(T)

    dpr_linear = Dpy(ftracks[0], 'w')
    dpr_linear.write_tracks(T)
    dpr_linear.close()
    del T


    for i,qgrid in enumerate([ds,gqs,ei]):
        t=time()
        euler=EuDX(a=qgrid.PK,ind=qgrid.IN,seeds=seeds,odf_vertices=qgrid.odf_vertices,a_low=.2)
        T=[track for track in euler]     
        print 'Eudx ',time()-t, len(T)
        dpr_linear = Dpy(ftracks[i+1], 'w')
        dpr_linear.write_tracks(T)
        dpr_linear.close()
        del T





















