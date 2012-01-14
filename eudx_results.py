import os
import numpy as np
import nibabel as nib
from dipy.reconst.dti import Tensor
from dipy.reconst.gqi import GeneralizedQSampling
from dipy.reconst.dsi import DiffusionSpectrum
from dipy.reconst.dni import EquatorialInversion
from dipy.tracking.eudx import EuDX
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
def f2(t):
    x=np.linspace(-1,1,len(t))
    y=np.linspace(-1,1,len(t))
    z=np.zeros(x.shape)
    return x,y,z
    
def f2_spiral(t,b=0.2):        
    r=10*t+b*t   
    x=r*np.cos(t)
    y=r*np.sin(t)
    z=np.zeros(x.shape)        
    X=np.array([x,y,z]).T        
    R=X[:,0]**2+X[:,1]**2+X[:,2]**2
    r_max=R.max()
    return x/r_max,y/r_max,z/r_max
    
def f2_ellipse(t,r1=0.5,r2=0.3):
    x=r1*np.cos(t)
    y=r2*np.sin(t)
    z=np.zeros(x.shape)        
    return x,y,z

def f3(t):
    x=np.linspace(-0.5,0,len(t))
    y=-np.linspace(-0.5,0,len(t))    
    z=np.zeros(x.shape)
    return x,y,z

def gaussian_noise(vol,snr):
    voln=np.random.randn(*vol.shape[:3])
    pvol=np.sum(vol[:,:,:,0]**2) #power of initial volume
    pnoise=np.sum(np.random.randn(*voln.shape[:3])**2) #power of noise volume
    K=pvol/pnoise
    #print pvol,pnoise,K
    return np.sqrt(K/np.float(snr))*np.random.randn(*vol.shape)

def simple_peaks(ODF,faces,thr,low):
    x,y,z,g=ODF.shape
    S=ODF.reshape(x*y*z,g)
    f,g=S.shape
    PK=np.zeros((f,5))
    IN=np.zeros((f,5))
    for (i,odf) in enumerate(S):
        if odf.max()>low:
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


def create_phantom():        
    SNR=100.
    no_seeds=20**3
    final_name='/home/eg309/Data/orbital_phantoms/'+str(SNR)+'_beauty'    
    print 'Loading data'
    #btable=np.loadtxt(get_data('dsi515btable'))
    #bvals=btable[:,0]
    #bvecs=btable[:,1:]
    bvals=np.loadtxt('data/subj_01/101_32/raw.bval')
    bvecs=np.loadtxt('data/subj_01/101_32/raw.bvec').T        
    print 'generate first simulation'
    f2=f2_ellipse
    vol2=orbital_phantom(bvals=bvals,bvecs=bvecs,evals=np.array([0.0017,0.0001,0.0001]),func=f2,
                         t=np.linspace(np.pi/2.,3*np.pi/2.,1000),datashape=(64,64,64,len(bvals)))    
    fvol2 = np.memmap('/tmp/t2', dtype='f8', mode='w+', shape=(64,64,64,len(bvals)))
    fvol2[:]=vol2[:]
    del vol2    
    print 'Created first component'
    norm=fvol2[...,0]/100.
    norm[norm==0]=1
    print 'Removing partial volume effects'
    fvol2[:]=fvol2[:]/norm[...,None]    
    print 'generate second simulation'
    vol3=orbital_phantom(bvals=bvals,bvecs=bvecs,evals=np.array([0.0017,0.0001,0.0001]),func=f3,datashape=(64,64,64,len(bvals)))
    fvol3 = np.memmap('/tmp/t3', dtype='f8', mode='w+', shape=(64,64,64,len(bvals)))
    fvol3[:]=vol3[:]
    del vol3    
    print 'Created second direction'
    norm=fvol3[...,0]/100.
    norm[norm==0]=1
    print 'Removing partial volume effects'
    fvol3[:]=fvol3[:]/norm[...,None]        
    print 'Creating final volume'        
    fvolfinal = np.memmap(final_name, dtype='f8', mode='w+', shape=(64,64,64,len(bvals)))
    fvolfinal[:]=fvol2[:]+fvol3[:]    
    print 'Adding two directions together'
    norm=fvolfinal[...,0]/100.
    norm[norm==0]=1
    print 'Removing partial volume effects'
    fvolfinal[:]=fvolfinal[:]/norm[...,None]    
    print 'Adding noise'    
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
    print 'Noise added'
    print 'Obtaining only a part from the data'
    
if __name__ == '__main__':
    visual=False
    no_seeds=20**3
    final_name='/home/eg309/Data/orbital_phantoms/100.0_beauty'
    bvals=np.loadtxt('data/subj_01/101_32/raw.bval')
    bvecs=np.loadtxt('data/subj_01/101_32/raw.bvec').T   
    fvolfinal = np.memmap(final_name, dtype='f8', mode='r', shape=(64,64,64,len(bvals)))
    
    sz=20
    
    data=fvolfinal[32-sz:32+sz,32-sz:32+sz,31-6:34+6,:]
    ds=DiffusionSpectrum(data,bvals,bvecs,odf_sphere='symmetric642',half_sphere_grads=True,auto=True,save_odfs=True)
    gq=GeneralizedQSampling(data,bvals,bvecs,1.2,odf_sphere='symmetric642',squared=False,save_odfs=True)    
    ei=EquatorialInversion(data,bvals,bvecs,odf_sphere='symmetric642',half_sphere_grads=True,auto=False,save_odfs=True,fast=True)
    ei.radius=np.arange(0,5,0.4)
    ei.gaussian_weight=0.05
    ei.set_operator('laplacian')
    ei.update()
    ei.fit()
    if visual:
        #print 'Showing data'
        #show_blobs(ds.ODF[:,:,0,:][:,:,None,:],ds.odf_vertices,ds.odf_faces,size=1.5,scale=1.)    
        show_blobs(ds.ODF[:20,20:40,6,:][:,:,None,:],ds.odf_vertices,ds.odf_faces,size=1.5,scale=1.,norm=True)
        show_blobs(ei.ODF[:20,20:40,6,:][:,:,None,:],ds.odf_vertices,ds.odf_faces,size=1.5,scale=1.,norm=True)
        show_blobs(gq.ODF[:20,20:40,6,:][:,:,None,:],ds.odf_vertices,ds.odf_faces,size=1.5,scale=1.,norm=True)
    #
    #PK,IN=simple_peaks(ds.ODF,ds.odf_faces,0.7,0.1*ds.ODF.max())    
    #PK2,IN2=simple_peaks(ei.ODF,ei.odf_faces,0.7,0)
    #PK2=ei.PK
    #IN2=ei.IN
    #
    tensors = Tensor(data, bvals, bvecs, thresh=50)
    FA = tensors.fa()    
    #MASK=np.zeros(FA.shape)
    #MASK=(FA>.5)&(FA<.8)    
    #cleanup background     
    ds.PK[FA<.2]=np.zeros(5) 
    ei.PK[FA<.2]=np.zeros(5)
    gq.PK[FA<.2]=np.zeros(5) 
            
    #euler = EuDX(a=FA, ind=tensors.ind(), seeds=no_seeds, a_low=.2)
    #euler = EuDX(a=ds.GFA, ind=ds.ind()[:,:,:,0], seeds=no_seeds, a_low=.2)
    #euler = EuDX(a=gq.QA, ind=gq.ind(), seeds=no_seeds, a_low=.0239)
    #tracks = [track for track in euler]
    
    x,y,z,g=ei.PK.shape
    seeds=np.zeros((no_seeds,3))
    for i in range(no_seeds):        
        rx=(x-1)*np.random.rand()
        ry=(y-1)*np.random.rand()
        rz=(z-1)*np.random.rand()            
        seed=np.ascontiguousarray(np.array([rx,ry,rz]),dtype=np.float64)
        seeds[i]=seed
    
    euler = EuDX(a=ds.PK, ind=ds.IN, seeds=seeds, odf_vertices=ds.odf_vertices, a_low=.2)
    tracks = [track for track in euler]    
    euler2 = EuDX(a=ei.PK, ind=ei.IN, seeds=seeds, odf_vertices=ei.odf_vertices, a_low=.2)
    tracks2 = [track for track in euler2]
    euler3 = EuDX(a=gq.PK, ind=gq.IN, seeds=seeds, odf_vertices=ei.odf_vertices, a_low=.2)
    tracks3 = [track for track in euler3]
        
    #simplify
    #qb=QuickBundles(tracks,4,12)
    #virtuals=qb.virtuals()
    #show tracks
    
    if visual:
        
        r=fvtk.ren()
        r.SetBackground(1.,1.,1.)
        fvtk.add(r,fvtk.line(tracks,fvtk.red))
        fvtk.show(r)
        fvtk.clear(r)    
        fvtk.add(r,fvtk.line(tracks2,fvtk.red,linewidth=1.))    
        fvtk.show(r)
        fvtk.clear(r)
        fvtk.add(r,fvtk.line(tracks3,fvtk.red,linewidth=1.))    
        fvtk.show(r)

