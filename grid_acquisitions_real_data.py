
import numpy as np
import nibabel as nib
from time import time
from dipy.data import get_data
from dipy.reconst.gqi import GeneralizedQSampling
from dipy.reconst.dti import Tensor
from dipy.reconst.dni import DiffusionNabla, EquatorialInversion
from dipy.reconst.dsi import DiffusionSpectrum, project_hemisph_bvecs
from dipy.tracking.propagation import EuDX
from dipy.align.aniso2iso import resample
from dipy.viz import fvtk

from visualize_dsi import show_blobs
#you need to switch to matthew's branch dicom-sorting for nibabel
from nibabel.nicom.dicomreaders import read_mosaic_dir

if __name__ == '__main__':

    
    dirname='/home/eg309/Data/project01_dsi/connectome_0001/tp1/RAWDATA/DSI'
    data,affine,bvals,bvecs = read_mosaic_dir(dirname,globber='mr*',sort_func='instance number')
           
    bvecs[np.isnan(bvecs)]=0
    #project identical b-vectors to the other hemisphere
    bvecs2,pairs=project_hemisph_bvecs(bvals,bvecs)
    #get voxel size
    zooms=np.sqrt(np.sum(affine[:3,:3]**2,axis=0))
    nzooms=(zooms[0],zooms[0],zooms[0])
    #resample datasets
    print(data.shape)
    data,affine=resample(data,affine,zooms,nzooms)
       
    print(data.shape)
    #mask a bit the background or load the mask
    mask=data[:,:,:,0]>20
    #img=nib.load('mask.nii.gz')
    #mask.nii.gz')
 
    #mask=img.get_data()>0

    data_part=data[50:80,30:70,27,:]
    dat=data_part[0:15,0:20].reshape(15,20,1,data_part.shape[-1])    
    np.save('/tmp/bvals.npz',bvals)
    np.save('/tmp/bvecs.npy',bvecs)
    np.save('/tmp/bvecs2.npy',bvecs2)
    np.save('/tmp/data.npy',dat)        
    
    #bvecs=np.load('/tmp/bvecs.npy')
    #bvecs2=np.load('/tmp/bvecs2.npy')
    #dat=np.load('/tmp/data.npy')
    #dat=data_part2[6,15,0,:].reshape(1,1,1,515)
    dat=dat.astype('f8')
    
    """
    dn=DiffusionNabla(dat,bvals,bvecs2,
                      odf_sphere='symmetric642',
                      mask=None,auto=False,
                      save_odfs=True,fast=True)    
    dn.peak_thr=.4
    dn.iso_thr=.05
    dn.radius=np.arange(0,5,0.1)
    dn.radiusn=len(dn.radius)
    dn.origin=8#16
    dn.precompute_fast_coords()    
    dn.precompute_equator_indices(5.)
    dn.precompute_angular(0.01)    
    dn.fit()
    """
    
    """
    ei=EquatorialInversion(dat,bvals,bvecs2,odf_sphere='symmetric642',auto=False,save_odfs=True,fast=True)
    ei.radius=np.arange(0,5,0.1)
    ei.gaussian_weight=0.05
    ei.set_operator('signal')
    ei.update()
    ei.fit()    
    """
    
    #ds=DiffusionSpectrum(dat,bvals,bvecs2,odf_sphere='symmetric642',mask=None,auto=True,save_odfs=True)    
    ten=Tensor(dat,bvals,bvecs)
    FA=ten.fa()
   

    #gq=GeneralizedQSampling(dat,bvals,bvecs,1.2,odf_sphere='symmetric642',squared=False,save_odfs=True)
    gq2=GeneralizedQSampling(dat,bvals,bvecs,3.,odf_sphere='symmetric642',squared=True,save_odfs=True)
    #remove the minimum ODF
    #res=dn.ODF-np.min(dn.ODF,axis=-1)[:,:,:,None]
    
    #show_blobs(dn.ODF,dn.odf_vertices,dn.odf_faces,FA.squeeze(),size=1.5,scale=1.)
    #show_blobs(ei.ODF,ei.odf_vertices,ei.odf_faces,FA.squeeze(),size=1.5,scale=1.)
    #show_blobs(ds.ODF,ds.odf_vertices,ds.odf_faces,FA.squeeze(),size=1.5,scale=1.)
    #show_blobs(gq.ODF,gq.odf_vertices,gq.odf_faces,FA.squeeze(),size=1.5,scale=1.)
    show_blobs(gq2.ODF,gq2.odf_vertices,gq2.odf_faces,FA.squeeze(),size=1.5,scale=1.)
    #show_blobs(dn2.ODF,dn.odf_vertices,dn.odf_faces)
    #show_blobs(ds.ODF,dn.odf_vertices,dn.odf_faces)
    #show_blobs(gq.ODF,dn.odf_vertices,dn.odf_faces)
    #sodfs=botox_weighting(dn.ODF,0.1,0.1,dn.odf_vertices)
    """
    #dti reconstruction
    ten=Tensor(data,bvals,bvecs2,mask=mask)
    #generate tensor tracks with random seeds
    eu=EuDX(ten.fa(),ten.ind(),seeds=10000,a_low=0.2)
    tenT=[e for e in eu]

    #dsi reconstruction
    ds=DiffusionSpectrum(data,bvals,bvecs2,mask=mask)
    #generate dsi tracks with random seeds
    eu_fa=EuDX(ten.fa(),ds.ind()[...,0],seeds=10000,a_low=0.2)    
    dsTfa=[e for e in eu_fa]
    eu_gfa=EuDX(ds.gfa(),ds.ind()[...,0],seeds=10000,a_low=0.2)
    dsTgfa=[e for e in eu_gfa]
    eu_qa=EuDX(ds.qa(),ds.ind(),seeds=10000,a_low=0.001)
    dsTqa=[e for e in eu_qa]
    """
    
    #r=fvtk.ren()
    #fvtk.add(r,fvtk.line(dsTfa,fvtk.green))
    #fvtk.add(r,fvtk.line(dsTqa,fvtk.blue))
    #fvtk.add(r,fvtk.line(tenT,fvtk.red))
    #fvtk.show(r)
    
    def show_pks(PK):
        x,y,z,w=PK.shape
        sPK=np.sum(PK.reshape(x*y*z,w)>0,axis=1)
        return sPK.reshape(x,y,z)

    

