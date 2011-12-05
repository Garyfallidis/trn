import numpy as np

import nibabel as nib
from dipy.viz import fvtk
from dipy.data import get_data, get_sphere
from dipy.reconst.recspeed import peak_finding
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
from dipy.core.sphere_stats import compare_orientation_sets
from dipy.core.geometry import sphere2cart

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
    
def show_blobs(blobs, v, faces,fa_slice=None,colormap='jet'):
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
                #m /= (2.2*m.max())
                #m /= (2.2*abs(m).max())
                #m /= (2.2*1.)
                #m[m<.4*abs(m).max()]=0
                
                x, y, z = v.T*m/2.2
                x += ii - xcen
                y += jj - ycen
                z += kk - zcen
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
    
def multi_d_isotropic_data(bvals,bvecs,dvals=[0.0015],S0=100,snr=None):
    print dvals, len(dvals)
    #data=np.zeros((19,len(bvals)))
    data=np.zeros((len(dvals),len(bvals)))
    
    for i, d in enumerate(dvals):
        #0 isotropic
        S,stics=SticksAndBall(bvals, bvecs, d, S0, 
                          angles=[(0, 0),(90,0),(90,90)], 
                          fractions=[0,0,0], snr=snr)
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
    
if __name__ == '__main__':
    
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
    
    gq2=GeneralizedQSampling(data,bvals,bvecs,20,odf_sphere='symmetric642',squared=True,auto=False,save_odfs=True)
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
        
    """
    r=fvtk.ren()    
    fvtk.add(r,fvtk.point(dn.odf_vertices[eqinds[0]],fvtk.red))
    fvtk.add(r,fvtk.point(dn.odf_vertices[eqinds[30]],fvtk.green))
    fvtk.show(r)
    """

     
