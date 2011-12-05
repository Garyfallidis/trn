import numpy as np
import nibabel as nib
from dipy.reconst.dti import Tensor
from dipy.tracking.propagation import EuDX
from dipy.tracking.metrics import length
from dipy.tracking.vox2track import track_counts
from os.path import join as pjoin
from dipy.viz import fvtk
from dipy.viz.colormap import orient2rgb,boys2rgb

dname='/home/eg309/Data/Virginia'#/11125'
fbvals='12387_0005.bvals'#'11125_0005.bvals'
fbvecs='12387_0005.bvecs'#'11125_0005.bvecs'
fnii='12387_0005.nii'#11125_0005_bet.nii.gz'
froi1='dti_FA_ant.nii'
froi2='dti_FA_ant2.nii'

#img=nib.load(dname+'/'+fnii)
img=nib.load(pjoin(dname,fnii))
data=img.get_data()
aff =img.get_affine()
bvals=np.loadtxt(dname+'/'+fbvals)
bvecs=np.loadtxt(dname+'/'+fbvecs).T
print 'dimensions ',data.shape,'voxel size ', img.get_header().get_zooms()[:3]
print 'check affine',np.round(aff,2)
print 'initial data structure', bvals.shape,bvecs.shape

#find b0s
ind=np.where(bvals==0)[0]
print ind

#create one S0 from all b0 volumes
S0=np.zeros(data.shape[:3],data.dtype)
S0=data[:,:,:,0]+data[:,:,:,13]+data[:,:,:,26]+data[:,:,:,39]+data[:,:,:,52]
S0=S0/5.

#update data
x,y,z,w=data.shape
data2=np.zeros((x,y,z,w-len(ind)+1),data.dtype)
bvals2=np.zeros(w-len(ind)+1)
bvecs2=np.zeros((w-len(ind)+1,3))

#copy data
cnt=0
for i in range(w):
    if i not in ind[1:]:        
        data2[:,:,:,cnt]=data[:,:,:,i]
        bvals2[cnt]=bvals[i]
        bvecs2[cnt,:]=bvecs[i,:]
        cnt+=1

print 'after',bvals2.shape,bvecs2.shape


#always check the threshold but 50 should be okay
ten=Tensor(data2,bvals2,bvecs2,thresh=50)

#ten.ind is indices of the eigen directions projected on a sphere
#stopping criteria is FA of .2
eu=EuDX(a=ten.fa(),ind=ten.ind(),seeds=5000,a_low=0.2)

#generate tracks
ten_tracks=[track for track in eu]
print 'No tracks ',len(ten_tracks)

#remove short tracks smaller than 40mm i.e. 20 in native units
ten_tracks=[t for t in ten_tracks if length(t)>20]
print 'No reduced tracks ',len(ten_tracks)

raw_input('Press enter...')


#load the rois
imsk1=nib.load(dname+'/'+froi1)
roi1=imsk1.get_data()
imsk2=nib.load(dname+'/'+froi2)
roi2=imsk2.get_data()

print 'roi dimensions', roi1.shape,roi2.shape
print 'roi voxels', np.sum(roi1==255),np.sum(roi2==255)
#tcs track counts volume
#tes dictionary of tracks passing from voxels
tcs,tes=track_counts(ten_tracks,data.shape[:3],(1,1,1),True)

#find volume indices of mask's voxels
roiinds1=np.where(roi1==255)
roiinds2=np.where(roi2==255)

#make it a nice 2d numpy array (Nx3)
roiinds1=np.array(roiinds1).T
roiinds2=np.array(roiinds2).T

def bring_roi_tracks(tracks,roiinds,tes):
    """
    bring the tracks from the roi region and their indices
    """
    cnt=0    
    sinds=[]
    for vox in roiinds:
        try:
            sinds+=tes[tuple(vox)]        
            cnt+=1
        except:
            pass
    return [tracks[i] for i in list(set(sinds))], list(set(sinds))
    
roi1_tracks,roi1_tracks_inds=bring_roi_tracks(ten_tracks,roiinds1,tes)
roi2_tracks,roi2_tracks_inds=bring_roi_tracks(ten_tracks,roiinds2,tes)

#use sets to get the intersections of track indices passsing roi1 and roi2
indices_tracks_roi1and2=list(set(roi1_tracks_inds).intersection(set(roi2_tracks_inds)))
tracks_roi1and2=[ten_tracks[i] for i in indices_tracks_roi1and2]
print 'Number of tracks passing from both rois', len(tracks_roi1and2)


#save results

ten_tracks=np.array(ten_tracks,dtype=np.object)
np.save(pjoin(dname,'ten_tracks.npy'),ten_tracks)
roi1_tracks=np.array(roi1_tracks,dtype=np.object)
np.save(dname+'/'+'roi1_tracks.npy',roi1_tracks)
roi2_tracks=np.array(roi2_tracks,dtype=np.object)
np.save(dname+'/'+'roi2_tracks.npy',roi2_tracks)

tracks_roi1and2=np.array(tracks_roi1and2,dtype=np.object)
np.save(dname+'/'+'roi1and2_tracks.npy',tracks_roi1and2)

#show them
r=fvtk.ren()

#change back to the correct dtype for tracks 
ten_tracks=[t.astype('f4') for t in ten_tracks]
tracks_roi1and2=[t.astype('f4') for t in tracks_roi1and2]

fvtk.add(r,fvtk.line(ten_tracks[:500],fvtk.green,opacity=0.5))
fvtk.add(r,fvtk.line(tracks_roi1and2,fvtk.red,opacity=0.8))

"""
#play with orientation colormap or many colours
cols=np.zeros((len(T),3))
for (i,v) in enumerate(T):
    cols[i,:]=orient2rgb(v[0]-v[-1])
fvtk.add(r,fvtk.line(T,cols,opacity=0.5))           
"""

fvtk.show(r,size=(1024,768))
fvtk.record(r,size=(1024,768),n_frames=100,az_ang=10,magnification=1)



