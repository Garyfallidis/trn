import os
import numpy as np
import nibabel as nib
from dipy.reconst.dti import Tensor
from dipy.reconst.gqi import GeneralizedQSampling
from dipy.reconst.dsi import DiffusionSpectrum
from dipy.tracking.propagation import EuDX
from dipy.io.dpy import Dpy
from dipy.external.fsl import create_displacements, warp_displacements, warp_displacements_tracks
from dipy.viz import fvtk
from dipy.external.fsl import flirt2aff
from dipy.sims.phantom import orbital_phantom
from visualize_dsi import show_blobs
from dipy.segment.quickbundles import QuickBundles
from dipy.reconst.recspeed import peak_finding

def test():
    pass

    


def transform_tracks(tracks,affine):
        return [(np.dot(affine[:3,:3],t.T).T + affine[:3,3]) for t in tracks]

def humans():   

    no_seeds=10**6
    visualize = False
    dirname = "data/"
    
    for root, dirs, files in os.walk(dirname):
        if root.endswith('101_32'):
            
            base_dir = root+'/'
            filename = 'raw'
            base_filename = base_dir + filename
            nii_filename = base_filename + 'bet.nii.gz'
            bvec_filename = base_filename + '.bvec'
            bval_filename = base_filename + '.bval'
            flirt_mat = base_dir + 'DTI/flirt.mat'    
            fa_filename = base_dir + 'DTI/fa.nii.gz'
            fsl_ref = '/usr/share/fsl/data/standard/FMRIB58_FA_1mm.nii.gz'
    
            print bvec_filename
            
            img = nib.load(nii_filename)
            data = img.get_data()
            affine = img.get_affine()
            bvals = np.loadtxt(bval_filename)
            gradients = np.loadtxt(bvec_filename).T # this is the unitary direction of the gradient
            
            tensors = Tensor(data, bvals, gradients, thresh=50)
            FA = tensors.fa()
            euler = EuDX(a=FA, ind=tensors.ind(), seeds=no_seeds, a_low=.2)
            tensor_tracks = [track for track in euler]
    
            dpy_filename = base_dir + 'DTI/res_tracks_dti.dpy'
            dpw = Dpy(dpy_filename, 'w')
            dpw.write_tracks(tensor_tracks)
            dpw.close()
            
            #gqs=GeneralizedQSampling(data,bvals,gradients)
            #euler=EuDX(a=gqs.qa(),ind=gqs.ind(),seeds=no_seeds,a_low=.0239)
            #gqs_tracks = [track for track in euler]
            
            #dpy_filename = base_dir + 'DTI/res_tracks_gqi.dpy'
            #dpw = Dpy(dpy_filename, 'w')
            #dpw.write_tracks(gqs_tracks)
            #dpw.close()
            
            img_fa =nib.load(fa_filename)
            img_ref =nib.load(fsl_ref)
            mat=flirt2aff(np.loadtxt(flirt_mat),img_fa,img_ref)
            del img_fa
            del img_ref       
            
            tensor_linear = transform_tracks(tensor_tracks,mat)
            #gqi_linear = transform_tracks(gqs_tracks,mat)        
        
            if visualize:
                renderer = fvtk.ren()
                fvtk.add(renderer, fvtk.line(tensor_tracks, fvtk.red, opacity=1.0))
                fvtk.show(renderer)
            
            #save tracks_warped_linear
            dpy_filename = base_dir + 'DTI/tensor_linear.dpy'
            print dpy_filename
            dpr_linear = Dpy(dpy_filename, 'w')
            dpr_linear.write_tracks(tensor_linear)
            dpr_linear.close()        
            
            #save tracks_warped_linear
            #dpy_filename = base_dir + 'gqi_linear.dpy'
            #print dpy_filename
            #dpr_linear = Dpy(dpy_filename, 'w')
            #dpr_linear.write_tracks(gqi_linear)
            #dpr_linear.close()
            
            break
       
#def create_phantom():

if __name__ == '__main__':
        
    SNR=100.
    no_seeds=10**3
    final_name='/home/eg309/Data/orbital_phantoms/'+str(SNR)+'_beauty'
    
    print 'Loading data'        
    #btable=np.loadtxt(get_data('dsi515btable'))
    #bvals=btable[:,0]
    #bvecs=btable[:,1:]
    
    bvals=np.loadtxt('data/subj_01/101_32/raw.bval')
    bvecs=np.loadtxt('data/subj_01/101_32/raw.bvec').T
       
    def f2(t):
        x=np.linspace(-1,1,len(t))
        y=np.linspace(-1,1,len(t))
        z=np.zeros(x.shape)
        return x,y,z
    
    vol2=orbital_phantom(bvals=bvals,bvecs=bvecs,evals=np.array([0.0017,0.0003,0.0003]),func=f2,datashape=(64,64,64,len(bvals)))    
    fvol2 = np.memmap('/tmp/t2', dtype='f8', mode='w+', shape=(64,64,64,len(bvals)))
    fvol2[:]=vol2[:]
    del vol2
    
    print 'Created first direction'
    norm=fvol2[...,0]/100.
    norm[norm==0]=1
    fvol2[:]=fvol2[:]/norm[...,None]  
    
    print 'Removed partial volume effects'
    def f3(t):
        x=np.linspace(-1,1,len(t))
        y=-np.linspace(-1,1,len(t))    
        z=np.zeros(x.shape)
        return x,y,z
    
    #second direction
    vol3=orbital_phantom(bvals=bvals,bvecs=bvecs,evals=np.array([0.0017,0.0003,0.0003]),func=f3,datashape=(64,64,64,len(bvals)))
    fvol3 = np.memmap('/tmp/t3', dtype='f8', mode='w+', shape=(64,64,64,len(bvals)))
    fvol3[:]=vol3[:]
    del vol3
    
    print 'Created second direction'
    norm=fvol3[...,0]/100.
    norm[norm==0]=1
    fvol3[:]=fvol3[:]/norm[...,None]
    
    print 'Removed partial volume effects'        
    fvolfinal = np.memmap(final_name, dtype='f8', mode='w+', shape=(64,64,64,len(bvals)))
    fvolfinal[:]=fvol2[:]+fvol3[:] #+fvol4
    
    print 'Adding two directions together'
    norm=fvolfinal[...,0]/100.
    norm[norm==0]=1
    fvolfinal[:]=fvolfinal[:]/norm[...,None]
    
    print 'Removed partial volume effects'
    print 'Adding noise'
    
    def gaussian_noise(vol,snr):
        voln=np.random.randn(*vol.shape[:3])
        pvol=np.sum(vol[:,:,:,0]**2) #power of initial volume
        pnoise=np.sum(np.random.randn(*voln.shape[:3])**2) #power of noise volume    
        K=pvol/pnoise
        #print pvol,pnoise,K
        return np.sqrt(K/np.float(snr))*np.random.randn(*vol.shape)
    
    print 'Noise 1'
    voln=np.random.randn(*fvolfinal[:].shape[:3])
    pvol=np.sum(fvolfinal[:,:,:,0]**2) #power of initial volume
    pnoise=np.sum(np.random.randn(*voln.shape)**2) #power of noise volume
    K=pvol/pnoise
    noise1 = np.memmap('/tmp/n1', dtype='f8', mode='w+', shape=(64,64,64,len(bvals)))
    noise1[:] = np.random.randn(*fvolfinal[:].shape)[:]
    noise1[:] = np.sqrt(K/np.float(SNR))*noise1[:]#*np.random.randn(*fvolfinal[:].shape)
    
    print 'Noise 2'
    voln=np.random.randn(*fvolfinal[:].shape[:3])
    pvol=np.sum(fvolfinal[:,:,:,0]**2) #power of initial volume
    pnoise=np.sum(np.random.randn(*voln.shape)**2) #power of noise volume
    K=pvol/pnoise
    noise2 = np.memmap('/tmp/n2', dtype='f8', mode='w+', shape=(64,64,64,len(bvals)))
    noise2[:] = np.random.randn(*fvolfinal[:].shape)[:]
    noise2[:] = np.sqrt(K/np.float(SNR))*noise2[:]
    
    print 'Adding both noise components'    
    fvolfinal[:]=np.sqrt((fvolfinal[:]+noise1[:])**2+noise2[:]**2)
    
    #"""
    print 'Noise added'
    print 'Obtaining only a part from the data'
    
    data=fvolfinal[32-10:32+10,32-10:32+10,31:34,:]
    
    #stop
    
    ds=DiffusionSpectrum(data,bvals,bvecs,odf_sphere='symmetric362',half_sphere_grads=True,auto=True,save_odfs=True)
    #gq=GeneralizedQSampling(data,bvals,bvecs,1.2,odf_sphere='symmetric642',squared=False,save_odfs=False)
    """
    ei=EquatorialInversion(data,bvals,bvecs,odf_sphere='symmetric642',
            half_sphere_grads=True,auto=False,save_odfs=True,fast=True)
    ei.radius=np.arange(0,5,0.1)
    ei.gaussian_weight=0.05
    ei.set_operator('laplacian')
    ei.update()
    ei.fit()
    """
    
    #print 'Showing data'
    #show_blobs(ds.ODF[:,:,0,:][:,:,None,:],ds.odf_vertices,ds.odf_faces,size=1.5,scale=1.)    
        
    def simple_peaks(ODF,faces,thr):
        x,y,z,g=ODF.shape
        S=ODF.reshape(x*y*z,g)
        f,g=S.shape
        PK=np.zeros((f,5))
        IN=np.zeros((f,5))
        for (i,odf) in enumerate(S):
            peaks,inds=peak_finding(odf,faces)
            ibigp=np.where(peaks>thr*peaks[0])[0]
            l=len(ibigp)
            if l>3:
                l=3
            PK[i,:l]=peaks[:l]/np.float(peaks[0])
            IN[i,:l]=inds[:l]            
        PK=PK.reshape(x,y,z,5)
        IN=IN.reshape(x,y,z,5)
        return PK,IN
    
    PK,IN=simple_peaks(ds.ODF,ds.odf_faces,0.7)    
    #stop
    
    """
    #tensors = Tensor(fvolfinal[:], bvals, bvecs, thresh=50)
    tensors = Tensor(data, bvals, bvecs, thresh=50)
    FA = tensors.fa()
    euler = EuDX(a=FA, ind=tensors.ind(), seeds=no_seeds, a_low=.2)    
    """            
    #euler = EuDX(a=ds.GFA, ind=ds.ind()[:,:,:,0], seeds=no_seeds, a_low=.2)
    #euler = EuDX(a=gq.QA, ind=gq.ind(), seeds=no_seeds, a_low=.0239)
    #tracks = [track for track in euler]
    euler = EuDX(a=PK, ind=IN, seeds=no_seeds, a_low=.2)
    tracks = [track for track in euler]
        
    qb=QuickBundles(tracks,4,12)
    virtuals=qb.virtuals()
    
    r=fvtk.ren()
    fvtk.add(r,fvtk.line(tracks,fvtk.red))
    fvtk.show(r)
    fvtk.clear(r)
    fvtk.add(r,fvtk.line(virtuals,fvtk.red))
    fvtk.show(r)

