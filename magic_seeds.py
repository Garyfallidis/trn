import sys
import numpy as np
import nibabel as nib
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

fmask1='/home/eg309/Data/John_Seg/_RH_premotor.nii.gz'
fmask2='/home/eg309/Data/John_Seg/_RH_parietal.nii.gz'
fmask3='/home/eg309/Data/John_Seg/_LH_premotor.nii.gz'
fmask4='/home/eg309/Data/John_Seg/_LH_parietal.nii.gz'

fmasks=[fmask1,fmask2,fmask3,fmask4]

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

def tracks_mask(tracks,shape,mask,value):
    tcs,tes=track_counts(tracks,shape,return_elements=True)    
    inds=np.array(np.where(mask>value)).T
    tracks_mask=[]
    for p in inds:
        try:
            tracks_mask+=tes[tuple(p)]
        except KeyError:
            pass
    return [tracks[i] for i in list(set(tracks_mask))]   


def tracks_double_mask(seeds,seeds2,gqs,shape,mask,mask2,a_low=.0239):

    euler =EuDX(a=gqs.qa(),ind=gqs.ind(),seeds=seeds,  odf_vertices=gqs.odf_vertices,\
               a_low=a_low,step_sz=0.25,ang_thr=80.)
    euler2=EuDX(a=gqs.qa(),ind=gqs.ind(),seeds=seeds2, odf_vertices=gqs.odf_vertices,\
               a_low=a_low,step_sz=0.25,ang_thr=80.)
    T=[t for t in euler]
    T2=[t for t in euler2]
    Tn=tracks_mask(T,shape,mask2,0)
    Tn2=tracks_mask(T2,shape,mask,0)
    return Tn+Tn2

def get_luigi_SLFI():
    fseg1='/home/eg309/Devel/segmented_bundles/luigi_s1_for_eleftherios/S1_SLFI'
    #fseg2='/home/eg309/Devel/segmented_bundles/luigi_s1_for_eleftherios/S1_SLFII'
    subject='01'
    fdpyw ='data/subj_'+subject+'/101_32/DTI/tracks_gqi_3M_linear.dpy'
    dpr = Dpy(fdpyw, 'r')
    T = dpr.read_tracks()
    dpr.close()
    seg_inds=load_pickle(fseg1)
    T1=[T[i] for i in seg_inds]
    #seg_inds=load_pickle(fseg2)
    #T2=[T[i] for i in seg_inds]
    return T1


def is_close(t,lT,thr):
    res=bundles_distances_mam([t.astype('f4')],lT,'min')
    if np.sum(res<thr)>=1:
        return True
    return False




def load_PX_tracks():

    roi='LH_premotor'

    dn='/home/hadron/from_John_mon12thmarch'
    dname='/extra_probtrackX_analyses/_subject_id_subj05_101_32/particle2trackvis_'+roi+'_native/'
    fname=dn+dname+'tract_samples.trk'
    from nibabel import trackvis as tv
    points_space=[None,'voxel','rasmm']
    streamlines,hdr=tv.read(fname,as_generator=True,points_space='voxel')
    tracks=[s[0] for s in streamlines]
    del streamlines
    #return tracks

    qb=QuickBundles(tracks,25./2.5,18)
    #tl=Line(qb.exemplars()[0],line_width=1)
    del tracks
    qb.remove_small_clusters(20)

    tl = TrackLabeler(qb,\
                qb.downsampled_tracks(),\
                vol_shape=None,\
                tracks_line_width=3.,\
                tracks_alpha=1)

    #put the seeds together
    #seeds=np.vstack((seeds,seeds2))
    #shif the seeds
    #seeds=np.dot(mat[:3,:3],seeds.T).T + mat[:3,3]
    #seeds=seeds-shift
    #seeds2=np.dot(mat[:3,:3],seeds2.T).T + mat[:3,3]
    #seeds2=seeds2-shift    
    #msk = Point(seeds,colors=(1,0,0,1.),pointsize=2.)
    #msk2 = Point(seeds2,colors=(1,0,.ppppp2,1.),pointsize=2.)
    w=World()
    w.add(tl)
    #w.add(msk)
    #w.add(msk2)
    #w.add(sl)    
    #create window
    wi = Window(caption='Fos',\
                bgcolor=(.3,.3,.6,1.),\
                width=1600,\
                height=900)
    wi.attach(w)
    #create window manager
    wm = WindowManager()
    wm.add(wi)
    wm.run()

def load_tracks(method='pmt'):
    from nibabel import trackvis as tv
    dname='/home/eg309/Data/orbital_phantoms/dwi_dir/subject1/'

    if method=='pmt':
        fname='/home/eg309/Data/orbital_phantoms/dwi_dir/workflow/tractography/_subject_id_subject1/cam2trk_pico_twoten/data_fit_pdfs_tracked.trk'
        streams,hdr=tv.read(fname,points_space='voxel')
        tracks=[s[0] for s in streams]
    if method=='dti':  fname=dname+'dti_tracks.dpy'    
    if method=='dsi':  fname=dname+'dsi_tracks.dpy'     
    if method=='gqs':  fname=dname+'gqi_tracks.dpy'
    if method=='eit':  fname=dname+'eit_tracks.dpy'
    if method in ['dti','dsi','gqs','eit']:
        dpr_linear = Dpy(fname, 'r')
        tracks=dpr_linear.read_tracks()
        dpr_linear.close()

    if method!='pmt':
        tracks = [t-np.array([96/2.,96/2.,55/2.]) for t in tracks if track_range(t,100/2.5,150/2.5)]
    tracks = [t for t in tracks if track_range(t,100/2.5,150/2.5)]

    print 'final no of tracks ',len(tracks)
    qb=QuickBundles(tracks,25./2.5,18)
    #from dipy.viz import fvtk
    #r=fvtk.ren()
    #fvtk.add(r,fvtk.line(qb.virtuals(),fvtk.red))
    #fvtk.show(r)
    #show_tracks(tracks)#qb.exemplars()[0])
    #qb.remove_small_clusters(40)
    del tracks
    #load 
    tl = TrackLabeler(qb,\
                qb.downsampled_tracks(),\
                vol_shape=None,\
                tracks_line_width=3.,\
                tracks_alpha=1)

    #return tracks
    w=World()
    w.add(tl)
    #create window
    wi = Window(caption='Fos',\
                bgcolor=(1.,1.,1.,1.),\
                width=1600,\
                height=900)
    wi.attach(w)
    #create window manager
    wm = WindowManager()
    wm.add(wi)
    wm.run()


if __name__=='__main__':

    #tracks=load_PX_tracks()
    #tracks=load_tracks('dti')
    #tracks=load_tracks('dsi')
    #tracks=load_tracks('gqs')
    #tracks=load_tracks('eit')
    tracks=load_tracks('pmt')






    stop

    subject=sys.argv[1]
    standard2native=False
    fsl_ref = '/usr/share/fsl/data/standard/FMRIB58_FA_1mm.nii.gz'
    img_ref =nib.load(fsl_ref)
    ffa='data/subj_'+subject+'/101_32/DTI/fa.nii.gz'
    fmat='data/subj_'+subject+'/101_32/DTI/flirt.mat'
    img_fa =nib.load(ffa)
    img_ref =nib.load(fsl_ref)
    ref_shape=img_ref.get_data().shape
    mat=flirt2aff(np.loadtxt(fmat),img_fa,img_ref)

    if standard2native:
        fimat='data/subj_'+subject+'/101_32/DTI/iflirt.mat'
        cmd='convert_xfm -omat '+ fimat + ' -inverse '+ fmat
        pipe(cmd)
        fmasknative='data/subj_'+subject+'/101_32/DTI/LH_premotor_native.nii.gz'
        #fmasknative='/tmp/test.nii.gz'
        cmd='flirt -in '+fmask3+ ' -ref '+ffa + ' -out '+ fmasknative + ' -init ' + fimat + ' -applyxfm'  
        pipe(cmd)
        fmasknative2='data/subj_'+subject+'/101_32/DTI/LH_parietal_native.nii.gz'
        cmd='flirt -in '+fmask4+ ' -ref '+ffa + ' -out '+ fmasknative2 + ' -init ' + fimat + ' -applyxfm'
        pipe(cmd)
    else:
        dmask='/home/eg309/Data/John_Seg/probtrackx_medial_SLF_analysis/seeds/'
        fmasknative =dmask+'_LH_parietal_native.nii.gz'
        fmasknative2=dmask+'_LH_premotor_native.nii.gz'

    ftracks='data/subj_'+subject+'/101_32/DTI/gqi_tracks_3M_ms.dpy'
    #create seeds for first mask
    img = nib.load(fmasknative)
    mask = img.get_data()
    seeds=get_seeds(mask,5,1)
    #create seeds for second mask
    img2 = nib.load(fmasknative2)
    mask2 = img2.get_data()
    seeds2=get_seeds(mask2,5,1)
    #load data
    fraw='data/subj_'+subject+'/101_32/rawbet.nii.gz'
    fbval='data/subj_'+subject+'/101_32/raw.bval'
    fbvec='data/subj_'+subject+'/101_32/raw.bvec'
    img = nib.load(fraw)
    data = img.get_data()
    affine = img.get_affine()
    bvals = np.loadtxt(fbval)
    gradients = np.loadtxt(fbvec).T
    #calculate FA
    tensors = Tensor(data, bvals, gradients, thresh=50)
    FA = tensors.fa()
    famask=FA>=.2
    #GQI
    gqs=GeneralizedQSampling(data,bvals,gradients,1.2,
                odf_sphere='symmetric642',
                mask=famask,
                squared=False,
                save_odfs=False)
    """
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
    ""
    #DSI
    ds=DiffusionSpectrum(data,bvals,gradients,\
                odf_sphere='symmetric642',\
                mask=famask,\
                half_sphere_grads=True,\
                auto=True,\
                save_odfs=False)
    """
    #ds.PK[FA<.2]=np.zeros(5)
    #euler=EuDX(a=FA,ind=tensors.ind(),seeds=seeds,a_low=.2)
    #euler=EuDX(a=ds.PK,ind=ds.IN,seeds=seeds,odf_vertices=ds.odf_vertices,a_low=.2)
    #euler=EuDX(a=ei.PK,ind=ei.IN,seeds=seeds,odf_vertices=ei.odf_vertices,\
    #            a_low=.2,step_sz=0.25,ang_thr=80.)
    #euler=EuDX(a=gqs.qa(),ind=gqs.ind(),seeds=seeds,odf_vertices=gqs.odf_vertices,a_low=.0239)
    euler=EuDX(a=gqs.qa(),ind=gqs.ind(),seeds=3*10**6, odf_vertices=gqs.odf_vertices,\
               a_low=.0239,step_sz=0.25,ang_thr=80.)
    #euler=EuDX(a=ds.pk(),ind=ds.ind(),seeds=5*10**6, odf_vertices=gqs.odf_vertices,\
    #           a_low=.1,step_sz=0.25,ang_thr=80.)


    lT=get_luigi_SLFI()

    T=[track for track in euler]
    print len(T)
            
    #T=[track for track in euler if track_range(track,100/2.5,200/2.5)]
    #T=tracks_double_mask(seeds,seeds2,gqs,mask.shape,mask,mask2,)
    T=transform_tracks(T,mat)
    print len(T)
    T=[track for track in T if is_close(track,lT,5)]

    shift=(np.array(ref_shape)-1)/2.
    T=[t-shift for t in T]
    print len(T)

    #save tracks
    dpr_linear = Dpy(ftracks, 'w')
    dpr_linear.write_tracks(T)
    dpr_linear.close()
    #cluster tracks
    qb=QuickBundles(T,25.,25)    
    #qb.remove_small_clusters(40)
    del T
    #load 
    tl = TrackLabeler(qb,\
                qb.downsampled_tracks(),\
                vol_shape=ref_shape,\
                tracks_line_width=3.,\
                tracks_alpha=1)
    fT1 = 'data/subj_'+subject+'/MPRAGE_32/T1_flirt_out.nii.gz'
    #fT1_ref = '/usr/share/fsl/data/standard/MNI152_T1_1mm_brain.nii.gz'
    img = nib.load(fT1)
    #img = nib.load(fT1)
    sl = Slicer(img.get_affine(),img.get_data())
    tl.slicer=sl

    luigi=Line([t-shift for t in lT],line_width=2)

    #put the seeds together
    seeds=np.vstack((seeds,seeds2))
    #shif the seeds
    seeds=np.dot(mat[:3,:3],seeds.T).T + mat[:3,3]
    seeds=seeds-shift
    #seeds2=np.dot(mat[:3,:3],seeds2.T).T + mat[:3,3]
    #seeds2=seeds2-shift    
    msk = Point(seeds,colors=(1,0,0,1.),pointsize=2.)
    #msk2 = Point(seeds2,colors=(1,0,.ppppp2,1.),pointsize=2.)
    w=World()
    w.add(tl)
    w.add(msk)
    #w.add(msk2)
    w.add(sl)
    w.add(luigi)
    #create window
    wi = Window(caption='subj_'+subject+'.png',\
                bgcolor=(.3,.3,.6,1.),\
                width=1600,\
                height=900)
    wi.attach(w)
    #create window manager
    wm = WindowManager()
    wm.add(wi)
    wm.run()



















