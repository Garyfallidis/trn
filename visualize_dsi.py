import numpy as np

import nibabel as nib
import matplotlib.pyplot as plt
from dipy.viz import fvtk
from dipy.data import get_data, get_sphere
from dipy.reconst.recspeed import peak_finding
from dipy.reconst.dti import Tensor
from dipy.reconst.gqi import GeneralizedQSampling
from dipy.reconst.dsi import DiffusionSpectrum
from dipy.reconst.dni import DiffusionNabla
from dipy.sims.voxel import SticksAndBall, SingleTensor
from scipy.fftpack import fftn, fftshift, ifftn,ifftshift
from scipy.ndimage import map_coordinates
from dipy.utils.spheremakers import sphere_vf_from
from scipy.ndimage.filters import laplace
from dipy.core.geometry import sphere2cart,cart2sphere,vec2vec_rotmat
from enthought.mayavi import mlab
from dipy.core.sphere_stats import compare_orientation_sets, angular_similarity
from dipy.core.geometry import sphere2cart
from dipy.reconst.dni import EquatorialInversion
from dipy.core.geometry import rodriguez_axis_rotation
from dipy.core.sphere_stats import random_uniform_on_sphere
from dipy.sims.phantom import orbital_phantom, add_rician_noise

import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def test():

    #img=nib.load('/home/eg309/Data/project01_dsi/connectome_0001/tp1/RAWDATA/OUT/mr000001.nii.gz')    
    btable=np.loadtxt(get_data('dsi515btable'))
    #volume size
    sz=16
    #shifting
    origin=8
    #hanning width
    filter_width=32.
    #number of signal sampling points
    n=515
    #odf radius
    radius=np.arange(2.1,6,.2)
    #create q-table
    bv=btable[:,0]
    bmin=np.sort(bv)[1]
    bv=np.sqrt(bv/bmin)
    qtable=np.vstack((bv,bv,bv)).T*btable[:,1:]
    qtable=np.floor(qtable+.5)
    #copy bvals and bvecs
    bvals=btable[:,0]
    bvecs=btable[:,1:]
    #S=img.get_data()[38,50,20]#[96/2,96/2,20]
    S,stics=SticksAndBall(bvals, bvecs, d=0.0015, S0=100, angles=[(0, 0),(60,0),(90,90)], fractions=[0,0,0], snr=None)
    
    '''
    print S[1:].max(), S[1:].min()
    plt.plot(bvals, np.log(S))
    plt.show()    
    '''
    
    bv=bvals
    bmin=np.sort(bv)[1]
    bv=np.sqrt(bv/bmin)        
    qtable=np.vstack((bv,bv,bv)).T*bvecs
    qtable=np.floor(qtable+.5)

    ql = [tuple(q) for q in qtable]
    print set(ql)
    print len(ql), len(set(ql))
    return ql    
    
    S2=S.copy()
    S2=S2.reshape(1,len(S))        
    dn=DiffusionNabla(S2,bvals,bvecs,auto=False)    
    pR=dn.equators    
    odf=dn.odf(S)    
    #Xs=dn.precompute_interp_coords()    
    '''
    peaks,inds=peak_finding(odf.astype('f8'),dn.odf_faces.astype('uint16'))    
    print peaks
    print peaks/peaks.min()    
    #print dn.PK
    dn.fit()
    print dn.PK
    '''
    '''
    ren=fvtk.ren()
    colors=fvtk.colors(odf,'jet')
    fvtk.add(ren,fvtk.point(dn.odf_vertices,colors,point_radius=.05,theta=8,phi=8))
    fvtk.show(ren)
    '''
    
    #ds=DiffusionSpectrum(S2,bvals,bvecs)
    #tpr=ds.pdf(S)
    #todf=ds.odf(tpr)
    
    """
    #show projected signal
    Bvecs=np.concatenate([bvecs[1:],-bvecs[1:]])
    X0=np.dot(np.diag(np.concatenate([S[1:],S[1:]])),Bvecs)    
    ren=fvtk.ren()
    fvtk.add(ren,fvtk.point(X0,fvtk.yellow,1,2,16,16))    
    fvtk.show(ren)
    """
    #qtable=5*matrix[:,1:]
    
    #calculate radius for the hanning filter
    r = np.sqrt(qtable[:,0]**2+qtable[:,1]**2+qtable[:,2]**2)
        
    #setting hanning filter width and hanning
    hanning=.5*np.cos(2*np.pi*r/filter_width)
    
    #center and index in q space volume
    q=qtable+origin
    q=q.astype('i8')
    
    #apply the hanning filter
    values=S*hanning
    
    """
    #plot q-table
    ren=fvtk.ren()
    colors=fvtk.colors(values,'jet')
    fvtk.add(ren,fvtk.point(q,colors,1,0.1,6,6))
    fvtk.show(ren)
    """    
    
    #create the signal volume    
    Sq=np.zeros((sz,sz,sz))
    for i in range(n):        
        Sq[q[i][0],q[i][1],q[i][2]]+=values[i]
    
    #apply fourier transform
    Pr=fftshift(np.abs(np.real(fftn(fftshift(Sq),(sz,sz,sz)))))
    
    #"""
    ren=fvtk.ren()
    vol=fvtk.volume(Pr)
    fvtk.add(ren,vol)
    fvtk.show(ren)
    #"""
    
    """
    from enthought.mayavi import mlab
    mlab.pipeline.volume(mlab.pipeline.scalar_field(Sq))
    mlab.show()
    """
    
    #vertices, edges, faces  = create_unit_sphere(5)    
    vertices, faces = sphere_vf_from('symmetric362')           
    odf = np.zeros(len(vertices))
        
    for m in range(len(vertices)):
        
        xi=origin+radius*vertices[m,0]
        yi=origin+radius*vertices[m,1]
        zi=origin+radius*vertices[m,2]        
        PrI=map_coordinates(Pr,np.vstack((xi,yi,zi)),order=1)
        for i in range(len(radius)):
            odf[m]=odf[m]+PrI[i]*radius[i]**2
    
    """
    ren=fvtk.ren()
    colors=fvtk.colors(odf,'jet')
    fvtk.add(ren,fvtk.point(vertices,colors,point_radius=.05,theta=8,phi=8))
    fvtk.show(ren)
    """
    
    """
    #Pr[Pr<500]=0    
    ren=fvtk.ren()
    #ren.SetBackground(1,1,1)
    fvtk.add(ren,fvtk.volume(Pr))
    fvtk.show(ren)
    """
    
    peaks,inds=peak_finding(odf.astype('f8'),faces.astype('uint16'))
        
    Eq=np.zeros((sz,sz,sz))
    for i in range(n):        
        Eq[q[i][0],q[i][1],q[i][2]]+=S[i]/S[0]
    
    LEq=laplace(Eq)
    
    #Pr[Pr<500]=0    
    ren=fvtk.ren()
    #ren.SetBackground(1,1,1)
    fvtk.add(ren,fvtk.volume(Eq))
    fvtk.show(ren)    
        
    phis=np.linspace(0,2*np.pi,100)
           
    planars=[]    
    for phi in phis:
        planars.append(sphere2cart(1,np.pi/2,phi))
    planars=np.array(planars)
    
    planarsR=[]
    for v in vertices:
        R=vec2vec_rotmat(np.array([0,0,1]),v)       
        planarsR.append(np.dot(R,planars.T).T)
    
    """
    ren=fvtk.ren()
    fvtk.add(ren,fvtk.point(planarsR[0],fvtk.green,1,0.1,8,8))
    fvtk.add(ren,fvtk.point(2*planarsR[1],fvtk.red,1,0.1,8,8))
    fvtk.show(ren)
    """
    
    azimsums=[]
    for disk in planarsR:
        diskshift=4*disk+origin
        #Sq0=map_coordinates(Sq,diskshift.T,order=1)
        #azimsums.append(np.sum(Sq0))
        #Eq0=map_coordinates(Eq,diskshift.T,order=1)
        #azimsums.append(np.sum(Eq0))
        LEq0=map_coordinates(LEq,diskshift.T,order=1)
        azimsums.append(np.sum(LEq0))
    
    azimsums=np.array(azimsums)
   
    #"""
    ren=fvtk.ren()
    colors=fvtk.colors(azimsums,'jet')
    fvtk.add(ren,fvtk.point(vertices,colors,point_radius=.05,theta=8,phi=8))
    fvtk.show(ren)
    #"""
    
    
    #for p in planarsR[0]:            
    """
    for m in range(len(vertices)):
        for ri in radius:
            xi=origin+ri*vertices[m,0]
            yi=origin+ri*vertices[m,1]
            zi=origin+ri*vertices[m,2]
        
        #ri,thetai,phii=cart2sphere(xi,yi,zi)
        
        sphere2cart(ri,pi/2.,phi)
        LEq[ri,thetai,phii]
    """
    #from volume_slicer import VolumeSlicer
    #vs=VolumeSlicer(data=Pr)
    #vs.configure_traits()
    
def show_blobs(blobs, v, faces,fa_slice=None,colormap='jet',size=1.5,scale=2.2):
    """Mayavi gets really slow when triangular_mesh is called too many times
    so this function stacks blobs and calls triangular_mesh once
    """
 
    print blobs.shape
    xcen = blobs.shape[0]/2.
    ycen = blobs.shape[1]/2.
    zcen = blobs.shape[2]/2.
    faces = np.asarray(faces, 'int')
    xx = []
    yy = []
    zz = []
    count = 0
    ff = []
    mm = []
    for ii in xrange(blobs.shape[0]):
        for jj in xrange(blobs.shape[1]):
            for kk in xrange(blobs.shape[2]):
                m = blobs[ii,jj,kk]
                m /= (size*abs(m).max())
                x, y, z = v.T*m/size                
                x += scale*(ii - xcen)
                y += scale*(jj - ycen)
                z += scale*(kk - zcen)                
                ff.append(count+faces)
                count += len(x)
                xx.append(x)
                yy.append(y)
                zz.append(z)
                mm.append(m)
    ff = np.concatenate(ff)
    xx = np.concatenate(xx)
    yy = np.concatenate(yy)
    zz = np.concatenate(zz)
    mm = np.concatenate(mm)
    mlab.triangular_mesh(xx, yy, zz, ff, scalars=mm, colormap=colormap)
    if fa_slice!=None:        
        mlab.imshow(fa_slice, colormap='gray', interpolate=False)
    mlab.colorbar()
    mlab.show()
    
def create_data(bvals,bvecs,d=0.0015,S0=100,snr=None):
    
    #data=np.zeros((19,len(bvals)))
    data=np.zeros((2,len(bvals)))
    #0 isotropic
    S,stics=SticksAndBall(bvals, bvecs, d, S0, 
                          angles=[(0, 0),(90,0),(90,90)], 
                          fractions=[0,0,0], snr=snr)
    data[0]=S.copy()
    '''
    #1 one fiber    
    S,stics=SticksAndBall(bvals, bvecs, d, S0, 
                          angles=[(30, 0),(90,0),(90,90)], 
                          fractions=[100,0,0], snr=snr)
    data[1]=S.copy()
    #2 two fibers
    S,stics=SticksAndBall(bvals, bvecs, d, S0, 
                          angles=[(0, 0),(90,0),(90,90)], 
                          fractions=[50,50,0], snr=snr)
    data[2]=S.copy()
    #3 three fibers
    S,stics=SticksAndBall(bvals, bvecs, d, S0, 
                          angles=[(0, 0),(90,0),(90,90)], 
                          fractions=[33,33,33], snr=snr)
    data[3]=S.copy()
    #4 three fibers iso
    S,stics=SticksAndBall(bvals, bvecs, d, S0, 
                          angles=[(0, 0),(90,0),(90,90)], 
                          fractions=[23,23,23], snr=snr)
    data[4]=S.copy()
    #5 three fibers more iso
    S,stics=SticksAndBall(bvals, bvecs, d, S0, 
                          angles=[(0, 0),(90,0),(90,90)], 
                          fractions=[13,13,13], snr=snr)
    data[5]=S.copy()
    #6 three fibers one at 60
    S,stics=SticksAndBall(bvals, bvecs, d, S0, 
                          angles=[(0, 0),(60,0),(90,90)], 
                          fractions=[33,33,33], snr=snr)
    data[6]=S.copy()    
    
    #7 three fibers one at 90,90 one smaller
    S,stics=SticksAndBall(bvals, bvecs, d, S0, 
                          angles=[(0, 0),(60,0),(90,90)], 
                          fractions=[33,33,23], snr=snr)
    data[7]=S.copy()
    
    #8 three fibers one at 90,90 one even smaller
    S,stics=SticksAndBall(bvals, bvecs, d, S0, 
                          angles=[(0, 0),(60,0),(90,90)], 
                          fractions=[33,33,13], snr=snr)
    data[8]=S.copy()
    
    #9 two fibers one at 0, second 30 
    S,stics=SticksAndBall(bvals, bvecs, d, S0, 
                          angles=[(0, 0),(45,0),(90,90)],
                          fractions=[50,50,0], snr=snr)
    data[9]=S.copy()
    
    #10 two fibers one at 0, second 30 
    S,stics=SticksAndBall(bvals, bvecs, d, S0, 
                          angles=[(0, 0),(45,0),(90,90)], 
                          fractions=[50,30,0], snr=snr)
    data[10]=S.copy()
    
        
    #11 one fiber with 80% isotropic
    S,stics=SticksAndBall(bvals, bvecs, d, S0, 
                          angles=[(0, 0),(30,0),(90,90)], 
                          fractions=[20,0,0], snr=snr)    
    data[11]=S.copy()
    
    #12 one fiber with tensor
    evals=np.array([1.4,.35,.35])*10**(-3)
    #evals=np.array([1.4,.2,.2])*10**(-3)
    S=SingleTensor(bvals,bvecs,S0,evals=evals,evecs=np.eye(3),snr=snr)
    data[12]=S.copy()
    
    #13 three fibers with two tensors
    evals=np.array([1.4,.35,.35])*10**(-3)
    S=SingleTensor(bvals,bvecs,S0,evals=evals,evecs=np.eye(3),snr=None)
    evals=np.array([.35,1.4,.35])*10**(-3)
    S=SingleTensor(bvals,bvecs,S0,evals=evals,evecs=np.eye(3),snr=None)+S    
    #std=2*S0/snr
    #S=S+np.random.randn(len(S))*std    
    data[13]=S.copy()
    
    #14 three fibers with three tensors
    evals=np.array([1.4,.35,.35])*10**(-3)
    S=SingleTensor(bvals,bvecs,S0,evals=evals,evecs=np.eye(3),snr=None)
    evals=np.array([.35,1.4,.35])*10**(-3)
    S=SingleTensor(bvals,bvecs,S0,evals=evals,evecs=np.eye(3),snr=None)+S
    evals=np.array([.35,.35,1.4])*10**(-3)
    S=SingleTensor(bvals,bvecs,S0,evals=evals,evecs=np.eye(3),snr=None)+S
        
    #std=2*S0/snr
    #S=S+np.random.randn(len(S))*std    
    data[14]=S.copy()
    '''
    
    #15 isotropic with spherical tensor
    #evals=np.array([.15,.15,.15])*10**(-2)
    evals=np.array([d,d,d])
    #evals=np.array([.35,.35,.35])*10**(-3)
    #evals=np.array([.15,.15,.15])*10**(-2)
    S=SingleTensor(bvals,bvecs,S0,evals=evals,evecs=np.eye(3),snr=snr)
    #data[15]=S.copy()  
    data[1]=S.copy()  
    
    '''
    #16 angles
    
    angles=[(0,0),(60,0),(90,90)]       
    R=[sphere2cart(1,np.deg2rad(pair[0]),np.deg2rad(pair[1])) for pair in angles]
    R=np.array(R)
    R=R.T
    #print R
    
    evals=np.array([1.4,.35,.35])*10**(-3)
    S=SingleTensor(bvals,bvecs,S0,evals=evals,evecs=R,snr=None)
    evals=np.array([.35,1.4,.35])*10**(-3)
    S=SingleTensor(bvals,bvecs,S0,evals=evals,evecs=R,snr=None)+S
    evals=np.array([.35,.35,1.4])*10**(-3)
    S=SingleTensor(bvals,bvecs,S0,evals=evals,evecs=R,snr=None)+S
    
    data[16]=S.copy()
    
    #17 angles
    
    angles=[(0,0),(30,0),(90,90)]       
    R=[sphere2cart(1,np.deg2rad(pair[0]),np.deg2rad(pair[1])) for pair in angles]
    R=np.array(R)
    R=R.T
    #print R
    
    evals=np.array([1.4,.05,.05])*10**(-3)
    S=SingleTensor(bvals,bvecs,S0,evals=evals,evecs=R,snr=None)
    evals=np.array([.05,1.4,.05])*10**(-3)
    S=SingleTensor(bvals,bvecs,S0,evals=evals,evecs=R,snr=None)+S
    evals=np.array([.05,.05,1.4])*10**(-3)
    S=SingleTensor(bvals,bvecs,S0,evals=evals,evecs=R,snr=None)+S
    
    data[17]=S.copy()
        
    #18 angles
    
    angles=[(0,0),(20,0),(90,90)]       
    R=[sphere2cart(1,np.deg2rad(pair[0]),np.deg2rad(pair[1])) for pair in angles]
    R=np.array(R)
    R=R.T
    #print R
    
    evals=np.array([1.4,.05,.05])*10**(-3)
    S=SingleTensor(bvals,bvecs,S0,evals=evals,evecs=R,snr=None)
    evals=np.array([.05,1.4,.05])*10**(-3)
    S=SingleTensor(bvals,bvecs,S0,evals=evals,evecs=R,snr=None)+S
    evals=np.array([.05,.05,1.4])*10**(-3)
    S=SingleTensor(bvals,bvecs,S0,evals=evals,evecs=R,snr=None)+S
    
    data[18]=S.copy()
    '''
    
    return data

def create_data_2fibers(bvals,bvecs,d=0.0015,S0=100,angles=np.arange(0,92,5),snr=None):
    
    data=np.zeros((len(angles),len(bvals)))
        
    for (i,a) in enumerate(angles):
        #2 two fibers
        S,stics=SticksAndBall(bvals, bvecs, d, S0, 
                          angles=[(0, 0),(a,0),(0,0)], 
                          fractions=[50,50,0], snr=snr)
        data[i]=S.copy()
        
    return data

def radial_points_on_sphere(index,odf_vertices,angle):
        
        close_points=[]
        v0=odf_vertices[index]
        for (i,v) in enumerate(odf_vertices):
            if np.abs(np.rad2deg(np.arccos(np.dot(v0,v)))) < angle:
                if i!=index:
                    close_points.append(i)
        return close_points
    
def super_reduced_peaks(odfs,odf_vertices,odf_faces,angle):
        
        final=[]
        for (i,odf) in enumerate(odfs):
            pks,ins=peak_finding(odf,odf_faces)
            peaks=pks[:3]
            inds=ins[:3]
            print '#', peaks
            del_peaks=[]
            for (j,ind) in enumerate(inds):
                pts=radial_points_on_sphere(ind,odf_vertices,angle)
                for p in pts:
                    if peaks[j]<odf[p]:
                        del_peaks.append(j)
            
            peaks=np.delete(peaks,del_peaks)
            print '@',peaks
            print ' ', len(peaks)
            final.append(len(peaks))
        print(final)    

def botox_weighting(odfs,smooth,level,odf_vertices):        
    W=np.dot(odf_vertices,odf_vertices.T)
    W=W.astype('f8')**2        
    nodfs=np.zeros(odfs.shape)        
    for (i,odf) in enumerate(odfs):
        #odf=odf.ravel()
        #print odf.shape            
        W2=np.dot(1./np.abs(odf[:,None]),np.abs(odf[None,:]))
        Z=np.zeros(W.shape)
        Z[W2>1]=1.
        E=np.exp(Z*W/smooth)
        E=E/np.sum(E,axis=1)[:,None]        
        low_odf=odf.copy()
        low_odf[odf<level*odf.max()]=0        
        nodfs[i]=np.dot(low_odf[None,:],E)
        #nodfs[i]=np.dot(odf[None,:],E)
    return nodfs
           
def uplifting_weighting(odfs,smooth,odf_vertices):        
    W=np.dot(odf_vertices,odf_vertices.T)
    W=W.astype('f8')**2        
    nodfs=np.zeros(odfs.shape)        
    for (i,odf) in enumerate(odfs):
        W2=np.dot(1./np.abs(odf[:,None]),np.abs(odf[None,:]))
        Z=np.zeros(W.shape)
        Z[W2>1]=1.
        E=np.exp(Z*W/smooth)
        E=E/np.sum(E,axis=1)[:,None]        
        nodfs[i]=np.dot(odf[None,:],E)
    return nodfs


def angular_weighting(odfs,smooth,odf_vertices):
    A=np.dot(odf_vertices,odf_vertices.T)
    A=A.astype('f8')**2
    E=np.exp(A/smooth)        
    E=E/np.sum(E,axis=1)[:,None]
    #print E.min(), E.max()        
    nodfs=np.zeros(odfs.shape)
    for (i,odf) in enumerate(odfs):            
        #print odfs.min(), E.min(),E.max(), W.min(),W.max()            
        nodfs[i]=np.dot(odf[None,:],E)
    return nodfs
    
def low_weighting(odfs,smooth,level=0.5):
    A=np.dot(dn.odf_vertices,dn.odf_vertices.T)
    A=A.astype('f8')**2
    E=np.exp(A/smooth)
    E=E/np.sum(E,axis=1)[:,None]
    #print E.min(), E.max()
    nodfs=np.zeros(odfs.shape)
    for (i,odf) in enumerate(odfs):
        low_odf=odf.copy()
        low_odf[odf<level*odf.max()]=0   
        nodfs[i]=np.dot(low_odf[None,:],E)
    return nodfs

def extended_peak_filtering(odfs,odf_faces,thr=0.3):
        new_peaks=[]        
        for (i,odf) in enumerate(odfs):
            peaks,inds=peak_finding(odf,odf_faces)
            
            ismallp=np.where(peaks/peaks[0]<thr)
            if len(ismallp[0])>0:
                l=ismallp[0][0]
            else:
                l=len(peaks)
       
            print '#',i,peaknos[i]
            if l==0:
                print peaks[0]/peaks[0]
            else:
                print peaks[:l]/peaks[0]

def unique_qs(qtable):
    
    Q=qtable*np.array([50,500,5000.])
    sq=np.sum(Q,axis=1)
    u=[]
    for (i,q) in enumerate(sq):
        u.append(np.sum(sq==q))
    return np.array(u)



    

def show_simulated_multiple_crossings():
    
    '''
    S=np.array([[-1,0,0],[0,1,0],[0,0,1]])
    #T=np.array([[1,0,0],[0,0,]])
    T=np.array([[1,0,0],[-0.76672114,  0.26105963, -0.58650369]])
    print 'yo ',compare_orientation_sets(S,T)
    '''
    
    btable=np.loadtxt(get_data('dsi515btable'))
    bvals=btable[:,0]
    bvecs=btable[:,1:]
    
    """
    path='/home/eg309/Data/PROC_MR10032/subj_03/101_32/1312211075232351192010121313490254679236085ep2dadvdiffDSI10125x25x25STs004a001'    
    #path='/home/eg309/Data/101_32/1312211075232351192010121313490254679236085ep2dadvdiffDSI10125x25x25STs004a001'
    bvals=np.loadtxt(path+'.bval')
    bvecs=np.loadtxt(path+'.bvec').T   
    bvals=np.append(bvals.copy(),bvals[1:].copy())
    bvecs=np.append(bvecs.copy(),-bvecs[1:].copy(),axis=0)        
    """
    #bvals=np.load('/tmp/bvals.npy')
    #bvecs=np.load('/tmp/bvecs.npy')    
    #data=create_data(bvals,bvecs,d=0.00075,S0=100,snr=None)    
    #dvals=[0.0003, 0.0006, 0.0012, 0.0024, 0.0048]
    dvals=[0.0003, 0.0005, 0.0010, 0.0015, 0.0020, 0.0025, 0.0030]
    data = multi_d_isotropic_data(bvals,bvecs,dvals=dvals,S0=100,snr=None)
    dn=DiffusionNabla(data,bvals,bvecs,odf_sphere='symmetric362',
                      auto=False,save_odfs=True,fast=False,rotation=None)
    #rotation=np.array([[0,0,1],[np.sqrt(0.5),np.sqrt(0.5),0.],[np.sqrt(0.5),-np.sqrt(0.5),0.]])
    dn.peak_thr=.4
    dn.iso_thr=.05
    dn.radius=np.arange(0,6,0.2)
    dn.radiusn=len(dn.radius)
    
    dn.create_qspace(bvals,bvecs,16,8)
    dn.radon_params(64)
    dn.precompute_interp_coords()
    dn.precompute_fast_coords()
    dn.precompute_equator_indices(5.)
    dn.precompute_angular(0.01)
    dn.fit()
    
    '''

    ds=DiffusionSpectrum(data,bvals,bvecs,odf_sphere='symmetric642',auto=True,save_odfs=True)
    
    gq=GeneralizedQSampling(data,bvals,bvecs,1.2,odf_sphere='symmetric642',squared=False,auto=False,save_odfs=True)
    gq.peak_thr=.6
    gq.iso_thr=.7
    gq.fit()
    
    gq2=GeneralizedQSampling(data,bvals,bvecs,2.,odf_sphere='symmetric642',squared=True,auto=False,save_odfs=True)
    gq2.peak_thr=.6
    gq2.iso_thr=.7
    gq2.fit()
    '''
        
    dn_odfs=dn.odfs()
   
    #blobs=np.zeros((1,19,4,dn.odfn))
    blobs=np.zeros((1,len(dvals),1,dn.odfn))
    #blobs=np.zeros((1,19,1,dn.odfn))
    #blobs[0,:,0]=dn.odfs()
    for i,d in enumerate(dvals):
        blobs[0,i,0]=dn.ODF[i]/dn.ODF[i].mean()
        
    '''        
    blobs[0,:,1]=ds.odfs()
    blobs[0,:,2]=gq.odfs()    
    gq2odfs=gq2.odfs()
    gq2odfs[gq2odfs<0]=0
    
    print 'gq2',gq2odfs.min(),gq2odfs.max()
    blobs[0,:,3]=gq2odfs
    '''
    '''
    peaknos=np.array([0,1,2,3,3,3,3,3,3,2,2,1,1,2,3,0,3,3,3])    
    print peaknos
    print np.sum(dn.pk()>0,axis=1),np.sum(np.sum(dn.pk()>0,axis=1)-peaknos),len(peaknos)
    '''
    '''
    print np.sum(ds.qa()>0,axis=1),np.sum(np.sum(ds.qa()>0,axis=1)-peaknos),len(peaknos)
    print np.sum(gq.qa()>0,axis=1),np.sum(np.sum(gq.qa()>0,axis=1)-peaknos),len(peaknos)
    print np.sum(gq2.qa()>0,axis=1),np.sum(np.sum(gq2.qa()>0,axis=1)-peaknos),len(peaknos)
    '''
    show_blobs(blobs,dn.odf_vertices,dn.odf_faces,colormap='jet')

    for i, d in enumerate(dvals):
        m = dn.ODF[i].min()
        M = dn.ODF[i].max()
        vM = np.argmax(dn.ODF[i])
        print i, d, 50*(M-m)/(M+m), dn.odf_vertices[vM]
        
    #odf=dn.fast_odf(data[0])
    #r=fvtk.ren()
    #fvtk.add(r,fvtk.volume(dn.Eq))
    #fvtk.show(r)
    
    """
    blobs2=np.zeros((1,1,3,dn.odfn))    
    blobs2[0,:,0]=dn.log_fast_odf(data[15])
    blobs2[0,:,1]=dn.fast_odf(data[15])
    blobs2[0,:,2]=dn.log_slow_odf(data[15])
    show_blobs(blobs2,dn.odf_vertices,dn.odf_faces)
    
    e=data[15]/data[15,0]    
    ne=np.sqrt(-np.log(e))
    norm=np.sum(dn.qtable**2,axis=1)
    """
        
    """
    r=fvtk.ren()    
    fvtk.add(r,fvtk.point(dn.odf_vertices[eqinds[0]],fvtk.red))
    fvtk.add(r,fvtk.point(dn.odf_vertices[eqinds[30]],fvtk.green))
    fvtk.show(r)
    """
    
def show_simulated_2fiber_crossings():

    btable=np.loadtxt(get_data('dsi515btable'))
    bvals=btable[:,0]
    bvecs=btable[:,1:]
    
    data=create_data_2fibers(bvals,bvecs,d=0.0015,S0=100,angles=np.arange(0,47,5),snr=None)
    #data=create_data_2fibers(bvals,bvecs,d=0.0015,S0=100,angles=np.arange(0,92,5),snr=None)
    
    print data.shape
    #stop
    
    
    #"""
    dn=DiffusionNabla(data,bvals,bvecs,odf_sphere='symmetric642',
                      auto=False,save_odfs=True,fast=True)    
    dn.peak_thr=.4
    dn.iso_thr=.05
    dn.radius=np.arange(0,5,0.1)
    dn.radiusn=len(dn.radius)    
    dn.create_qspace(bvals,bvecs,16,8)
    dn.radon_params(64)
    dn.precompute_interp_coords()
    dn.precompute_fast_coords()
    dn.precompute_equator_indices(5.)
    dn.precompute_angular(None)#0.01)
    dn.fit()
    
    dn2=DiffusionNabla(data,bvals,bvecs,odf_sphere='symmetric642',
                      auto=False,save_odfs=True,fast=False)    
    dn2.peak_thr=.4
    dn2.iso_thr=.05
    dn2.radius=np.arange(0,5,0.1)
    dn2.radiusn=len(dn.radius)    
    dn2.create_qspace(bvals,bvecs,16,8)
    dn2.radon_params(64)
    dn2.precompute_interp_coords()
    dn2.precompute_fast_coords()
    dn2.precompute_equator_indices(5.)
    dn2.precompute_angular(None)#0.01)
    dn2.fit()
    #"""
    
        
    eis=EquatorialInversion(data,bvals,bvecs,odf_sphere='symmetric642',
                      auto=False,save_odfs=True,fast=True)    
    #eis.radius=np.arange(0,6,0.2)
    eis.radius=np.arange(0,5,0.1)    
    eis.gaussian_weight=None
    eis.set_operator('signal')#'laplap')
    eis.update()
    eis.fit()
    
    eil=EquatorialInversion(data,bvals,bvecs,odf_sphere='symmetric642',
                      auto=False,save_odfs=True,fast=True)
    #eil.radius=np.arange(0,6,0.2)
    eil.radius=np.arange(0,5,0.1)    
    eil.gaussian_weight=None
    eil.set_operator('laplacian')#'laplap')
    eil.update()
    eil.fit()
    
    eil2=EquatorialInversion(data,bvals,bvecs,odf_sphere='symmetric642',
                      auto=False,save_odfs=True,fast=True)    
    #eil2.radius=np.arange(0,6,0.2)
    eil2.radius=np.arange(0,5,0.1)
    eil2.gaussian_weight=None
    eil2.set_operator('laplap')
    eil2.update()
    eil2.fit()
        
    ds=DiffusionSpectrum(data,bvals,bvecs,odf_sphere='symmetric642',auto=True,save_odfs=True)

    gq=GeneralizedQSampling(data,bvals,bvecs,1.2,odf_sphere='symmetric642',squared=False,auto=False,save_odfs=True)
    gq.fit()
    
    gq2=GeneralizedQSampling(data,bvals,bvecs,3.,odf_sphere='symmetric642',squared=True,auto=False,save_odfs=True)
    gq2.fit()    

    #blobs=np.zeros((19,1,8,dn.odfn))
    blobs=np.zeros((10,1,6,dn.odfn))
    """
    blobs[:,0,0]=dn.odfs()
    blobs[:,0,1]=dn2.odfs()
    blobs[:,0,2]=eis.odfs()
    blobs[:,0,3]=eil.odfs()    
    blobs[:,0,4]=eil2.odfs()         
    blobs[:,0,5]=ds.odfs()
    blobs[:,0,6]=gq.odfs()
    blobs[:,0,7]=gq2.odfs()
    """
    blobs[:,0,0]=dn2.odfs()
    blobs[:,0,1]=eil.odfs()
    eo=eil2.odfs()
    #eo[eo<0.05*eo.max()]=0
    blobs[:,0,2]=eo         
    blobs[:,0,3]=ds.odfs()
    blobs[:,0,4]=gq.odfs()
    blobs[:,0,5]=gq2.odfs()
        
    show_blobs(blobs,dn.odf_vertices,dn.odf_faces,scale=0.5)

    
    
def ang_vec(u,v):    
    return np.abs(np.rad2deg(np.arccos(np.dot(u,v))))

def generate_gold_data(bvals,bvecs,fibno=1,axesn=10,SNR=None):
        
    #1 fiber case
    if fibno==1:
        data=np.zeros((axesn,len(bvals)))
        gold=np.zeros((axesn,3))    
        X=random_uniform_on_sphere(axesn,'xyz')        
        for (j,x) in enumerate(X):
            S,sticks=SticksAndBall(bvals,bvecs,d=0.0015,S0=100,angles=np.array([x]),fractions=[100],snr=SNR)
            data[j,:]=S[:]
            gold[j,:3]=x
        return data,gold,None,None,axesn,None

    #2 fibers case
    if fibno==2:        
        angles=np.arange(0,92,2.5)#5
        u=np.array([0,0,1])    
        anglesn=len(angles)        
        data=np.zeros((anglesn*axesn,len(bvals)))
        gold=np.zeros((anglesn*axesn,6))
        angs=[]
        for (i,a) in enumerate(angles):        
            v=sphere2cart(1,np.deg2rad(a),0)
            X=random_uniform_on_sphere(axesn,'xyz')        
            for (j,x) in enumerate(X):
                R=rodriguez_axis_rotation(x,360*np.random.rand())                
                ur=np.dot(R,u)
                vr=np.dot(R,v)                        
                S,sticks=SticksAndBall(bvals,bvecs,d=0.0015,S0=100,angles=np.array([ur,vr]),fractions=[50,50],snr=SNR)
                data[i*axesn+j,:]=S[:]
                gold[i*axesn+j,:3]=ur
                gold[i*axesn+j,3:]=vr
                angs.append(a)
        return data,gold,angs,anglesn,axesn,angles
               
    #3 fibers case
    if fibno==3:    
        #phis=np.array([0.,120.,240.])
        #angles=np.arange(0,92,5)
        angles=np.linspace(0,54.73,40)#20           
        anglesn=len(angles)        
        data=np.zeros((anglesn*axesn,len(bvals)))
        gold=np.zeros((anglesn*axesn,9))
        angs=[]
        for (i,a) in enumerate(angles): 
            u=sphere2cart(1,np.deg2rad(a),np.deg2rad(0))
            v=sphere2cart(1,np.deg2rad(a),np.deg2rad(120))
            w=sphere2cart(1,np.deg2rad(a),np.deg2rad(240))        
            X=random_uniform_on_sphere(axesn,'xyz')        
            for (j,x) in enumerate(X):
                R=rodriguez_axis_rotation(x,360*np.random.rand())
                ur=np.dot(R,u)
                vr=np.dot(R,v)
                wr=np.dot(R,w)         
                S,sticks=SticksAndBall(bvals,bvecs,d=0.0015,S0=100,angles=np.array([ur,vr,wr]),fractions=[33,33,33],snr=SNR)
                print i,j
                data[i*axesn+j,:]=S[:]
                gold[i*axesn+j,:3]=ur
                gold[i*axesn+j,3:6]=vr
                gold[i*axesn+j,6:]=wr                                        
                angs.append(a)
                
        return data,gold,angs,anglesn,axesn,angles

def do_compare(gold,vts,A,IN,thr,anglesn,axesn,type=2):
    if type==1:
        res=[]
        for (k,a) in enumerate(A):
            peaks=np.where(A[k]>thr)[0]
            print len(peaks),peaks
            if len(peaks)>0:
                inds=IN[k][peaks].astype('i8')
                ur=gold[k,:3]                             
                if len(peaks)>=1:
                    un=vts[inds[0]]
                    #cp=compare_orientation_sets(np.array([ur]),np.array([un]))*1.
                    cp=angular_similarity(np.array([ur]),np.array([un]))                    
                    res.append(cp)
            else:
                res.append(0)            
        res=np.array(res)
        return res,None,None
    
    if type==2:
        return do_compare_2(gold,vts,A,IN,thr,anglesn,axesn)        
    if type==3:
        return do_compare_3(gold,vts,A,IN,thr,anglesn,axesn)

def do_compare_2(gold,vts,A,IN,thr,anglesn,axesn):     
    res=[]
    for (k,a) in enumerate(A):
        peaks=np.where(A[k]>thr)[0]
        print len(peaks),peaks
        if len(peaks)>0:
            inds=IN[k][peaks].astype('i8')
            ur=gold[k,:3]
            vr=gold[k,3:]             
            if len(peaks)==1:
                un=vts[inds[0]]                    
                #cp=compare_orientation_sets(np.array([ur,vr]),np.array([un]))*1
                cp=angular_similarity(np.array([ur,vr]),np.array([un]))
                res.append(cp)
            if len(peaks)>1:
                un=vts[inds[0]]
                vn=vts[inds[1]]
                #cp=compare_orientation_sets(np.array([ur,vr]),np.array([un,vn]))*2
                cp=angular_similarity(np.array([ur,vr]),np.array([un,vn]))
                res.append(cp)
        else:
            res.append(0)            
    res=np.array(res).reshape(anglesn,axesn)
    me=np.mean(res,axis=1)
    st=np.std(res,axis=1)
    return res,me,st
    
                
def do_compare_3(gold,vts,A,IN,thr,anglesn,axesn):     
    res=[]
    for (k,a) in enumerate(A):
        peaks=np.where(A[k]>thr)[0]
        print len(peaks),peaks
        if len(peaks)>0:
            inds=IN[k][peaks].astype('i8')
            ur=gold[k,:3]
            vr=gold[k,3:6]
            wr=gold[k,6:]             
            if len(peaks)==1:
                un=vts[inds[0]]                    
                #cp=compare_orientation_sets(np.array([ur,vr,wr]),np.array([un]))*1.
                cp=angular_similarity(np.array([ur,vr,wr]),np.array([un]))
                res.append(cp)
            if len(peaks)==2:
                un=vts[inds[0]]
                vn=vts[inds[1]]
                #cp=compare_orientation_sets(np.array([ur,vr,wr]),np.array([un,vn]))*2.
                cp=angular_similarity(np.array([ur,vr,wr]),np.array([un,vn]))
                res.append(cp)                    
            if len(peaks)>=3:
                un=vts[inds[0]]
                vn=vts[inds[1]]
                wn=vts[inds[2]]
                #cp=compare_orientation_sets(np.array([ur,vr,wr]),np.array([un,vn,wn]))*3.
                cp=angular_similarity(np.array([ur,vr,wr]),np.array([un,vn,wn]))
                res.append(cp)                                    
        else:
            res.append(0)
    res=np.array(res).reshape(anglesn,axesn)
    me=np.mean(res,axis=1)
    st=np.std(res,axis=1)
    return res,me,st

def viz_signal_simulations_spherical_grid():
    
    #fimg,fbvals,fbvecs=get_data('small_101D')
    
    #bvals=np.loadtxt(fbvals)
    #bvecs=np.loadtxt(fbvecs).T
    v,f=get_sphere('symmetric642')
    
    bvals=8000*np.ones(len(v))
    bvecs=v
    bvals=np.insert(bvals,0,0)
    bvecs=np.insert(bvecs,0,np.array([0,0,0]),axis=0)
       
    #img=nib.load(fimg)    
    #S,sticks=SticksAndBall(bvals,bvecs,d=0.0015,S0=100,angles=[(0,0)],fractions=[100],snr=None)
    S,sticks=SticksAndBall(bvals,bvecs,d=0.0015,S0=100,angles=[(0,0),(45,0)],fractions=[50,50],snr=None)

    show_blobs(S[None,None,None,1:,],v,f)


def compare_dni_dsi_gqi_gqi2_eit():

    #stop
    
    btable=np.loadtxt(get_data('dsi515btable'))
    bvals=btable[:,0]
    bvecs=btable[:,1:]
    
    type=3
    axesn=200
    SNR=20
    
    data,gold,angs,anglesn,axesn,angles=generate_gold_data(bvals,bvecs,fibno=type,axesn=axesn,SNR=SNR)
    
    gq=GeneralizedQSampling(data,bvals,bvecs,1.2,odf_sphere='symmetric642',squared=False,save_odfs=True)    
    gq2=GeneralizedQSampling(data,bvals,bvecs,3.,odf_sphere='symmetric642',squared=True,save_odfs=True)
    
    ds=DiffusionSpectrum(data,bvals,bvecs,odf_sphere='symmetric642',auto=False,save_odfs=True)
    ds.filter_width=32.    
    ds.update()
    ds.fit()
    
    ei=EquatorialInversion(data,bvals,bvecs,odf_sphere='symmetric642',auto=False,save_odfs=True,fast=True)    
    ei.radius=np.arange(0,5,0.1)
    ei.gaussian_weight=None#0.01
    ei.set_operator('laplacian')
    ei.update()
    ei.fit()
    
    ei2=EquatorialInversion(data,bvals,bvecs,odf_sphere='symmetric642',auto=False,save_odfs=True,fast=True)    
    ei2.radius=np.arange(0,5,0.1)
    ei2.gaussian_weight=None
    ei2.set_operator('laplap')
    ei2.update()
    ei2.fit()
    
    ei3=EquatorialInversion(data,bvals,bvecs,odf_sphere='symmetric642',auto=False,save_odfs=True,fast=True)    
    ei3.radius=np.arange(0,5,0.1)
    ei3.gaussian_weight=None
    ei3.set_operator('signal')
    ei3.update()
    ei3.fit()    
    
    """
    blobs=np.zeros((2,4,642))
    
    no=200
    blobs[0,0,:]=gq.ODF[no]
    blobs[0,1,:]=gq2.ODF[no]
    blobs[0,2,:]=ds.ODF[no]
    blobs[0,3,:]=ei.ODF[no]
    no=399
    blobs[1,0,:]=gq.ODF[no]
    blobs[1,1,:]=gq2.ODF[no]
    blobs[1,2,:]=ds.ODF[no]
    blobs[1,3,:]=ei.ODF[no]       
    show_blobs(blobs[None,:,:,:],ei.odf_vertices,ei.odf_faces,1.2)
    """
    #stop
    
    vts=gq.odf_vertices
    
    def simple_peaks(ODF,faces,thr):
        x,g=ODF.shape
        PK=np.zeros((x,5))
        IN=np.zeros((x,5))
        for (i,odf) in enumerate(ODF):
            peaks,inds=peak_finding(odf,faces)
            ibigp=np.where(peaks>thr*peaks[0])[0]
            l=len(ibigp)
            if l>3:
                l=3
            PK[i,:l]=peaks[:l]
            IN[i,:l]=inds[:l]
        return PK,IN
    
    
    thresh=0.5
    
    PK,IN=simple_peaks(ds.ODF,ds.odf_faces,thresh)
    res,me,st =do_compare(gold,vts,PK,IN,0,anglesn,axesn,type)
    
    PK,IN=simple_peaks(gq.ODF,ds.odf_faces,thresh)
    res2,me2,st2 =do_compare(gold,vts,PK,IN,0,anglesn,axesn,type)
    
    PK,IN=simple_peaks(gq2.ODF,ds.odf_faces,thresh)
    res3,me3,st3 =do_compare(gold,vts,PK,IN,0,anglesn,axesn,type)
    
    PK,IN=simple_peaks(ei.ODF,ds.odf_faces,thresh)
    res4,me4,st4 =do_compare(gold,vts,PK,IN,0,anglesn,axesn,type)
        
    PK,IN=simple_peaks(ei2.ODF,ds.odf_faces,thresh)
    res5,me5,st5 =do_compare(gold,vts,PK,IN,0,anglesn,axesn,type)
      
    PK,IN=simple_peaks(ei3.ODF,ei3.odf_faces,thresh)
    res6,me6,st6 =do_compare(gold,vts,PK,IN,0,anglesn,axesn,type)
    
    #res7,me7,st7 =do_compare(ei2.PK,ei2.IN,0,anglesn,axesn,type)
    
    if type==1:    
        plt.figure(1)       
        plt.plot(res,'r',label='DSI')
        plt.plot(res2,'g',label='GQI')
        plt.plot(res3,'b',label='GQI2')
        plt.plot(res4,'k',label='EITL')
        plt.plot(res5,'k--',label='EITL2')
        plt.plot(res6,'k-.',label='EITS')
        #plt.xlabel('angle')
        #plt.ylabel('resolution')
        #plt.title('Angular accuracy')
        plt.legend()
        plt.show()   
    else:
        
        if type==3:
            x,y,z=sphere2cart(np.ones(len(angles)),np.deg2rad(angles),np.zeros(len(angles)))
            x2,y2,z2=sphere2cart(np.ones(len(angles)),np.deg2rad(angles),np.deg2rad(120*np.ones(len(angles))))
            angles2=[]        
            for i in range(len(x)):
                angles2.append(np.rad2deg(np.arccos(np.dot([x[i],y[i],z[i]],[x2[i],y2[i],z2[i]]))))
            angles=angles2    
            
        plt.figure(1)
        plt.plot(angles,me,'r',linewidth=3.,label='DSI')
        plt.plot(angles,me2,'g',linewidth=3.,label='GQI')
        plt.plot(angles,me3,'b',linewidth=3.,label='GQI2')
        plt.plot(angles,me4,'k',linewidth=3.,label='EITL')
        plt.plot(angles,me5,'k--',linewidth=3.,label='EITL2')
        plt.plot(angles,me6,'k-.',linewidth=3.,label='EITS')
        #plt.plot(angles,me7,'r--',linewidth=3.,label='EITL2')
        plt.xlabel('angle')
        plt.ylabel('similarity')
        
        title='Angular similarity of ' + str(type) + '-fibres crossing with SNR ' + str(SNR)
        plt.title(title)
        plt.legend(loc='center right')
        plt.savefig('/tmp/test.png',dpi=300)
        plt.show()
        
def compare_sdni_eitl():
    
    btable=np.loadtxt(get_data('dsi515btable'))
    bvals=btable[:,0]
    bvecs=btable[:,1:]   
    
    type=3
    axesn=200
    SNR=20
    
    data,gold,angs,anglesn,axesn,angles=generate_gold_data(bvals,bvecs,fibno=type,axesn=axesn,SNR=SNR)
    
    ei=EquatorialInversion(data,bvals,bvecs,odf_sphere='symmetric642',
                           auto=False,save_odfs=True,fast=True)    
    ei.radius=np.arange(0,5,0.1)
    ei.gaussian_weight=None
    ei.set_operator('laplacian')
    #{'reflect','constant','nearest','mirror', 'wrap'}
    ei.set_mode(order=1,zoom=1,mode='constant')    
    ei.update()
    ei.fit()

    dn=DiffusionNabla(data,bvals,bvecs,odf_sphere='symmetric642',
                      auto=False,save_odfs=True,fast=False)    
    dn.radius=np.arange(0,5,0.1)
    dn.gaussian_weight=None
    dn.update()
    dn.fit()
    
    vts=ei.odf_vertices
    
    def simple_peaks(ODF,faces,thr):
        x,g=ODF.shape
        PK=np.zeros((x,5))
        IN=np.zeros((x,5))
        for (i,odf) in enumerate(ODF):
            peaks,inds=peak_finding(odf,faces)
            ibigp=np.where(peaks>thr*peaks[0])[0]
            l=len(ibigp)
            if l>3:
                l=3
            PK[i,:l]=peaks[:l]
            IN[i,:l]=inds[:l]
        return PK,IN
    
    print "test0"
    thresh=0.5
    PK,IN=simple_peaks(ei.ODF,ei.odf_faces,thresh)
    res,me,st =do_compare(gold,vts,PK,IN,0,anglesn,axesn,type)
    #PKG,ING=simple_peaks(eig.ODF,ei.odf_faces,thresh)
    #resg,meg,stg =do_compare(gold,vts,PKG,ING,0,anglesn,axesn,type)
    PK,IN=simple_peaks(dn.ODF,dn.odf_faces,thresh)
    res2,me2,st2 =do_compare(gold,vts,PK,IN,0,anglesn,axesn,type)
    if type==3:
            x,y,z=sphere2cart(np.ones(len(angles)),np.deg2rad(angles),np.zeros(len(angles)))
            x2,y2,z2=sphere2cart(np.ones(len(angles)),np.deg2rad(angles),np.deg2rad(120*np.ones(len(angles))))
            angles2=[]
            for i in range(len(x)):
                angles2.append(np.rad2deg(np.arccos(np.dot([x[i],y[i],z[i]],[x2[i],y2[i],z2[i]]))))
            angles=angles2
    print "test1"
    plt.figure(1)
    plt.plot(angles,me,'k:',linewidth=3.,label='EITL')
    plt.plot(angles,me2,'k--',linewidth=3.,label='sDNI')
    #plt.plot(angles,me7,'r--',linewidth=3.,label='EITL2')
    plt.xlabel('angle')
    plt.ylabel('similarity')
    title='Angular similarity of ' + str(type) + '-fibres crossing with SNR ' + str(SNR)
    plt.title(title)
    plt.legend(loc='center right')
    plt.savefig('/tmp/test.png',dpi=300)
    plt.show()


def show_phantom_crossing(fname,type='dsi',weight=None):

    btable=np.loadtxt(get_data('dsi515btable'))
    bvals=btable[:,0]
    bvecs=btable[:,1:]

    #fname='/home/eg309/Desktop/t5'
    #fname='/home/eg309/Data/orbital_phantoms/20.0'
    #fname='/tmp/t5'
    fvol = np.memmap(fname, dtype='f8', mode='r', shape=(64,64,64,len(bvals)))
    data=fvol[32-10:32+10,32-10:32+10,31:34,:]
    #data=fvol[32,32,32,:][None,None,None,:]
    ten=Tensor(data,bvals,bvecs,thresh=50)
    FA=ten.fa()
       
    def H(x):
        res=(2*x*np.cos(x) + (x**2-2)*np.sin(x))/x**3
        res[np.isnan(res)]=1/3.
        return res
    
    if type=='dsi':
        mod=DiffusionSpectrum(data,bvals,bvecs,odf_sphere='symmetric642',auto=True,save_odfs=True)        
    if type=='gqi':
        mod=GeneralizedQSampling(data,bvals,bvecs,1.2,odf_sphere='symmetric642',squared=False,save_odfs=True)
    if type=='gqi2':
        mod=GeneralizedQSampling(data,bvals,bvecs,3.,odf_sphere='symmetric642',squared=True,save_odfs=True)
    if type=='eitl':
        ei=EquatorialInversion(data,bvals,bvecs,odf_sphere='symmetric642',auto=False,save_odfs=True,fast=True)
        ei.radius=np.arange(0,5,0.1)
        ei.gaussian_weight=weight
        ei.set_operator('laplacian')
        ei.update()
        ei.fit()
        mod=ei
    if type=='eitl2':
        ei=EquatorialInversion(data,bvals,bvecs,odf_sphere='symmetric642',auto=False,save_odfs=True,fast=True)
        ei.radius=np.arange(0,5,0.1)
        ei.gaussian_weight=weight
        ei.set_operator('laplap')
        ei.update()
        ei.fit()
        mod=ei
    if type=='eits':
        ei=EquatorialInversion(data,bvals,bvecs,odf_sphere='symmetric642',auto=False,save_odfs=True,fast=True)
        ei.radius=np.arange(0,5,0.1)
        ei.gaussian_weight=weight
        ei.set_operator('signal')
        ei.update()
        ei.fit()
        mod=ei   
        
    show_blobs(mod.ODF[:,:,0,:][:,:,None,:],mod.odf_vertices,mod.odf_faces,fa_slice=FA[:,:,0],size=1.5,scale=1.)    
    #gq=GeneralizedQSampling(data,bvals,bvecs,1.2,odf_sphere='symmetric642',squared=True,save_odfs=True)
    #gq2=GeneralizedQSampling(data,bvals,bvecs,3*1.2,odf_sphere='symmetric642',squared=True,save_odfs=True)
    #gq3=GeneralizedQSampling(data,bvals,bvecs,6*1.2,odf_sphere='symmetric642',squared=True,save_odfs=True)
    #gq4=GeneralizedQSampling(data,bvals,bvecs,9.4*1.2,odf_sphere='symmetric642',squared=True,save_odfs=True)
    #x=np.linspace(0,8,200)
    #ei=EquatorialInversion(data,bvals,bvecs,odf_sphere='symmetric642',auto=False,save_odfs=True,fast=True)
    #ei.radius=np.arange(0,5,0.1)
    #ei.gaussian_weight=None
    #ei.set_operator('laplacian')
    #ei.update()
    #ei.fit()
    #show_blobs(ei.ODF[:,:,0,:][:,:,None,:],ei.odf_vertices,ei.odf_faces,fa_slice=FA[:,:,0],scale=1.)

def create_software_phantom_no_pv_effects():
#if __name__=='__main__':
 
    SNR=5.
    final_name='/home/eg309/Data/orbital_phantoms/'+str(SNR)+'_fat'
   
    print 'Loading data'        
    btable=np.loadtxt(get_data('dsi515btable'))
    bvals=btable[:,0]
    bvecs=btable[:,1:]
   
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
    fvolfinal[:]=fvol2[:]+fvol3[:]#+fvol4
    
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
    
    print 'Noise added'
    print 'Obtaining only a part from the data'
    data=fvolfinal[32-10:32+10,32-10:32+10,31:34,:]
    ds=DiffusionSpectrum(data,bvals,bvecs,odf_sphere='symmetric642',auto=True,save_odfs=True)
    print 'Showing data'
    show_blobs(ds.ODF[:,:,0,:][:,:,None,:],ds.odf_vertices,ds.odf_faces,scale=5.)
    
    #voln=add_rician_noise(vol234)
    #gq=GeneralizedQSampling(data,bvals,bvecs,1.2,odf_sphere='symmetric642',squared=False,save_odfs=True)


def show_gaussian_smoothing_effects():

    SNR=20
    type='laplacian'
    
    print 'Loading data'        
    btable=np.loadtxt(get_data('dsi515btable'))
    bvals=btable[:,0]
    bvecs=btable[:,1:]
    S,stics=SticksAndBall(bvals, bvecs, d=0.0015, S0=100, 
                          angles=[(0, 0),(90.,0),(90.,90.)], 
                          fractions=[33,33,33], snr=SNR)
    data=S[None,:]
        
    ei=EquatorialInversion(data,bvals,bvecs,odf_sphere='symmetric642',auto=False,save_odfs=True,fast=True)
    ei.radius=np.arange(0,5,0.1)
    ei.gaussian_weight=None
    ei.set_operator(type)
    ei.update()
    ei.fit()    
    
    ei2=EquatorialInversion(data,bvals,bvecs,odf_sphere='symmetric642',auto=False,save_odfs=True,fast=True)
    ei2.radius=np.arange(0,5,0.1)
    ei2.gaussian_weight=0.01
    ei2.set_operator(type)
    ei2.update()
    ei2.fit()

    ei3=EquatorialInversion(data,bvals,bvecs,odf_sphere='symmetric642',auto=False,save_odfs=True,fast=True)
    ei3.radius=np.arange(0,5,0.1)
    ei3.gaussian_weight=0.05
    ei3.set_operator(type)
    ei3.update()
    ei3.fit()
    
    ei4=EquatorialInversion(data,bvals,bvecs,odf_sphere='symmetric642',auto=False,save_odfs=True,fast=True)
    ei4.radius=np.arange(0,5,0.1)
    ei4.gaussian_weight=0.1
    ei4.set_operator(type)
    ei4.update()
    ei4.fit()
        
    blobs=np.zeros((4,len(ei.odf_vertices)))
    blobs[0]=ei.ODF.ravel()
    blobs[1]=ei2.ODF.ravel()
    blobs[2]=ei3.ODF.ravel()
    blobs[3]=ei4.ODF.ravel()
    
    show_blobs(blobs[None,None,:,:],ei.odf_vertices,ei.odf_faces,size=1.5,scale=1.)
    #show_blobs(ei.ODF[None,None,:,:],ei.odf_vertices,ei.odf_faces,size=1.5,scale=1.)


