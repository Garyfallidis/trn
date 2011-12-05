import tempfile
from time import time
import numpy as np
from visualize_dsi import show_blobs
from dipy.reconst.dni import DiffusionNabla
from dipy.reconst.dsi import DiffusionSpectrum
from dipy.data import get_data
from dipy.sims.phantom import orbital_phantom, add_rician_noise
from dipy.viz import fvtk


def f(t):
        x=.2*np.sin(t)
        y=.2*np.cos(t)
        #z=np.zeros(t.shape)
        z=np.linspace(-.1,.1,len(x))
        return x,y,z

def f1(t):
    #x=np.linspace(-1,1,len(t))
    x=np.zeros(t.shape)
    y=np.linspace(-1,1,len(t))        
    z=np.zeros(x.shape)
    return x,y,z

def f2(t):
    x=np.linspace(-1,1,len(t))
    #y=-np.linspace(-1,1,len(t))
    y=np.zeros(t.shape)    
    z=np.zeros(x.shape)
    return x,y,z


if __name__ == '__main__':
        
    btable=np.loadtxt(get_data('dsi515btable'))
    bvals=btable[:,0]
    bvecs=btable[:,1:]
    
    """
    #first direction
    vol1=orbital_phantom(bvals=bvals,
                         bvecs=bvecs,
                         datashape=(64,64,64,len(bvals)),
                         func=f1,
                         radii=np.linspace(0.2,2,6))
    
    f1=tempfile.NamedTemporaryFile()
    fvol=np.memmap(f1,
                   dtype=vol1.dtype,
                   mode='w+',
                   shape=vol1.shape)
    
    fvol[:]=vol1[:]    
    del vol1
    
    #second direction
    vol2=orbital_phantom(bvals=bvals,
                         bvecs=bvecs,
                         datashape=(64,64,64,len(bvals)),                         
                         func=f2,
                         radii=np.linspace(0.2,2,6))        
    
    #double crossing
    #vol12=vol1+vol2
    #add some noise    
    #vol=add_rician_noise(vol12)
    fvol[:]+=vol2[:]    
    del vol2
    
    #data=fvol
    np.save('/tmp/test.npy',fvol)
    
    #stop
    
    """
    fvol=np.load('/tmp/test.npy')
    
    #data=np.load('/tmp/test.npy')    
    dat=fvol[22:42,22:42,32:34]
    
    del fvol
    
    t1=time()
    
    #"""
    dn=DiffusionNabla(dat,bvals,bvecs,odf_sphere='symmetric362',
                      auto=True,save_odfs=True,fast=True)
    """
    dn.radius=np.arange(0,6,.2)
    dn.radiusn=len(dn.radius)
    dn.create_qspace(bvals,bvecs,16,8)    
    dn.peak_thr=.4
    dn.iso_thr=.7
    dn.radon_params(64)
    dn.precompute_interp_coords()
    dn.fit()
    """
    
    #from dipy.reconst.dsi import DiffusionSpectrum
    #mask=dat[...,0]>0
    #dn=DiffusionSpectrum(dat,bvals,bvecs,save_odfs=True)
        
    t2=time()
    print t2-t1,'.sec'
    show_blobs(dn.odfs(),dn.odf_vertices,dn.odf_faces)
        
    #r=fvtk.ren()
    #fvtk.add(r,fvtk.volume(dat[...,0]))
    #fvtk.show(r)

    
