import sys
import numpy as np
import nibabel as nib
import os.path as op

import pyglet

#fos modules
from fos.actor.axes import Axes
from fos import World, Window, WindowManager
from fos.actor.line import Line
from labeler import TrackLabeler
from fos.actor.slicer import Slicer
#dipy modules
from dipy.segment.quickbundles import QuickBundles
from dipy.io.dpy import Dpy
from dipy.io.pickles import load_pickle,save_pickle
from dipy.viz.colormap import orient2rgb
from dipy.tracking.metrics import length

import copy

 
def track_range(a,b):
    lt=length(t) 
    if lt>a and lt<b:
        return True
    return False
if __name__ == '__main__':
    
    subject=sys.argv[1]#'05'    
    #print subject
   
    use_seg=False
    reduce_length=False
    rotate_snap=True
    
    dseg='/home/eg309/Devel/eleftherios2011/code/'
    #fseg='arcuate_s5m3_Nivedita_newdef.pkl'
    fseg='corticospinal_s1m3_LuigiNivedita.pkl'
    
    #fatlas='/home/eg309/Data/ICBM_Wmpm/ICBM_WMPM_eleftherios_padded.nii'
    #imga=nib.load(fatlas)
    #dataa=imga.get_data()    
        
    #load T1 volume registered in MNI space    
    #img = nib.load('data/subj_05/MPRAGE_32/T1_flirt_out.nii.gz')
    #img = nib.load('data/subj_'+subject+'/MPRAGE_32/T1_flirt_out.nii.gz')
    img = nib.load('/home/eg309/Data/John_Seg/_RH_premotor.nii.gz')
    data = img.get_data()
    affine = img.get_affine()    

    img = nib.load('/home/eg309/Data/John_Seg/_RH_parietal.nii.gz')
    data2 = img.get_data()
    affine2 = img.get_affine()

    img = nib.load('/home/eg309/Data/John_Seg/_LH_premotor.nii.gz')
    data3 = img.get_data()
    affine3 = img.get_affine()

    img = nib.load('/home/eg309/Data/John_Seg/_LH_parietal.nii.gz')
    data4 = img.get_data()
    affine4 = img.get_affine()

    data=200*(data+data2)+100*(data3+data4)
    data=data.astype(np.uint8)
    
    #load the tracks registered in MNI space
    #fdpyw = 'data/subj_'+subject+'/101_32/DTI/ei_linear.dpy' 
    fdpyw ='data/subj_'+subject+'/101_32/DTI/tracks_gqi_3M_linear.dpy'
        
    #fdpyw = 'data/subj_05/101_32/DTI/tracks_gqi_1M_linear.dpy'    
    dpr = Dpy(fdpyw, 'r')
    T = dpr.read_tracks()
    dpr.close()
    
    if use_seg:
        fseg=dseg+fseg
        seg_inds=load_pickle(fseg)
        T=[T[i] for i in seg_inds]
    
    if reduce_length:
        T=[t for t in T if track_range(120,150)]
        iT=np.random.randint(0,len(T),5000)
        T=[T[i] for i in iT]
    #stop
    
    #center
    shift=(np.array(data.shape)-1)/2.    
    T=[t-shift for t in T]
    
    #load initial QuickBundles with threshold 30mm
    #fpkl = 'data/subj_05/101_32/DTI/qb_gqi_1M_linear_30.pkl'
    qb=QuickBundles(T,25.,30)    
    #qb=load_pickle(fpkl)
        
    #create the interaction system for tracks 
    tl = TrackLabeler(qb,qb.downsampled_tracks(),vol_shape=data.shape,tracks_line_width=3.,tracks_alpha=1)   
    #add a interactive slicing/masking tool
    sl = Slicer(affine,data)    
    #add one way communication between tl and sl
    tl.slicer=sl
    #OpenGL coordinate system axes    
    ax = Axes(100)
    x,y,z=data.shape
    #add the actors to the world    
    w=World()
    w.add(tl)
    w.add(sl)
    #w.add(ax)
    #create a window
    #wi = Window(caption="Interactive bundle selection using fos and QB",\
    #            bgcolor=(0.3,0.3,0.6,1),width=1600,height=1000)    
    wi = Window(caption="Fos",bgcolor=(1.,1.,1.,1.),width=1600,height=900)
    #attach the world to the window
    wi.attach(w)
    #create a manager which can handle multiple windows
    wm = WindowManager()
    wm.add(wi)
    wm.run()
    print('Everything is running ;-)')

    #rotate camera
    if rotate_snap==True:
        cam=w.get_cameras()[0]
        cam.cam_rot.rotate(-90,1,0,0.)
        cam.cam_rot.rotate(-90,0,1,0.)











    
