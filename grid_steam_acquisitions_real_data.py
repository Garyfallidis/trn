
import numpy as np
import nibabel as nib
from time import time
from dipy.data import get_data
from dipy.reconst.gqi import GeneralizedQSampling
from dipy.reconst.dti import Tensor
from dipy.reconst.dni import DiffusionNabla,EquatorialInversion
from dipy.reconst.dsi import DiffusionSpectrum, project_hemisph_bvecs
from dipy.tracking.propagation import EuDX
from dipy.align.aniso2iso import resample
from dipy.viz import fvtk
from visualize_dsi import show_blobs


if __name__ == '__main__':

    path='/home/eg309/Data/PROC_MR10032/subj_03/101_32/1312211075232351192010121313490254679236085ep2dadvdiffDSI10125x25x25STs004a001'    
    #path='/home/eg309/Data/101_32/1312211075232351192010121313490254679236085ep2dadvdiffDSI10125x25x25STs004a001'
   
    img=nib.load(path+'_bet.nii.gz')
    data=img.get_data()
    affine=img.get_affine()
    bvals=np.loadtxt(path+'.bval')
    bvecs=np.loadtxt(path+'.bvec').T
        
    data_part=data[48-20:48+20,48-20:48+20,35,:]
    #dat=data_part[0:40,0:40].reshape(40,40,1,data_part.shape[-1])
    #data_part=data[:48,:,35,:]    
    dat=data_part[:,:,None,:]       
    dat=dat.astype('f8')
    

    """
    dn=DiffusionNabla(dat,bvals,bvecs,
                      odf_sphere='symmetric642',
                      mask=None,
                      half_sphere_grads=True,
                      auto=False,
                      save_odfs=True,fast=True)
    
    dn.peak_thr=.4
    dn.iso_thr=.05
    dn.radius=np.arange(0,6,0.2)
    #dn.radius=np.arange(1.,6,0.2)
    #dn.radius=np.arange(0,1.,0.2)
    dn.radiusn=len(dn.radius)
    dn.origin=8#16
    dn.precompute_fast_coords()    
    dn.precompute_equator_indices(5.)
    dn.precompute_angular(0.1)    
    dn.fit()
    """
    
    ei=EquatorialInversion(dat,bvals,bvecs,odf_sphere='symmetric642',
            half_sphere_grads=True,auto=False,save_odfs=True,fast=True)
    ei.radius=np.arange(0,5,0.1)
    ei.gaussian_weight=0.05
    ei.set_operator('laplacian')
    ei.update()
    ei.fit()    

    ds=DiffusionSpectrum(dat,bvals,bvecs,
                         odf_sphere='symmetric642',
                         mask=None,
                         half_sphere_grads=True,
                         auto=True,
                         save_odfs=True)
    
    ten=Tensor(dat,bvals,bvecs)
    FA=ten.fa()

    #gq=GeneralizedQSampling(dat,bvals,bvecs,1.2,odf_sphere='symmetric642',squared=False,save_odfs=True)
    #gq2=GeneralizedQSampling(dat,bvals,bvecs,3.,odf_sphere='symmetric642',squared=True,save_odfs=True)    
    #show_blobs(ds.ODF,ds.odf_vertices,ds.odf_faces,FA.squeeze(),size=1.5,scale=1.)
    show_blobs(ei.ODF,ei.odf_vertices,ei.odf_faces,FA.squeeze(),size=1.5,scale=1.)

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

    

