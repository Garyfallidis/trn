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
        
    S2=S.copy()
    S2=S2.reshape(1,len(S))        
    dn=DiffusionNabla(S2,bvals,bvecs,auto=False)    
    pR=dn.equators    
    odf=dn.odf(S)    
    #Xs=dn.precompute_interp_coords()    
    peaks,inds=peak_finding(odf.astype('f8'),dn.odf_faces.astype('uint16'))    
    print peaks
    print peaks/peaks.min()    
    #print dn.PK
    dn.fit()
    print dn.PK
    
    #"""
    ren=fvtk.ren()
    colors=fvtk.colors(odf,'jet')
    fvtk.add(ren,fvtk.point(dn.odf_vertices,colors,point_radius=.05,theta=8,phi=8))
    fvtk.show(ren)
    #"""
    
    stop
    
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
    
def color_patches(vertices, faces, heights, colors):
    """Mayavi gets really slow when triangular_mesh is called too many times
    so this function stacks blobs and calls triangular_mesh once
    """
 
    print colors.shape
    xcen = colors.shape[0]/2.
    ycen = colors.shape[1]/2.
    zcen = colors.shape[2]/2.
    faces = np.asarray(faces, 'int')
    xx = []
    yy = []
    zz = []
    count = 0
    ff = []
    cc = []
    for ii in xrange(colors.shape[0]):
        for jj in xrange(colors.shape[1]):
            for kk in xrange(colors.shape[2]):
                c = colors[ii,jj,kk]
                #m[m<.4*abs(m).max()]=0
                m = heights[ii,jj,kk]
                
                print 'vertices', vertices.shape
                print 'm', m.shape
                
                x, y, z = vertices.T*m
                x += ii - xcen
                y += jj - ycen
                z += kk - zcen
                ff.append(count+faces)
                count += len(x)
                xx.append(x)
                yy.append(y)
                zz.append(z)
                cc.append(c)
    ff = np.concatenate(ff)
    xx = np.concatenate(xx)
    yy = np.concatenate(yy)
    zz = np.concatenate(zz)
    cc = np.concatenate(cc)
    mlab.triangular_mesh(xx, yy, zz, ff, scalars=cc)
    mlab.show()
    
def create_data(d=0.0015,S0=100,snr=None):
    
    data=np.zeros((16,len(bvals)))
    #isotropic
    S,stics=SticksAndBall(bvals, bvecs, d, S0, 
                          angles=[(0, 0),(90,0),(90,90)], 
                          fractions=[0,0,0], snr=snr)
    data[0]=S.copy()
    #one fiber    
    S,stics=SticksAndBall(bvals, bvecs, d, S0, 
                          angles=[(30, 0),(90,0),(90,90)], 
                          fractions=[100,0,0], snr=snr)
    data[1]=S.copy()
    #two fibers
    S,stics=SticksAndBall(bvals, bvecs, d, S0, 
                          angles=[(0, 0),(90,0),(90,90)], 
                          fractions=[50,50,0], snr=snr)
    data[2]=S.copy()
    #three fibers
    S,stics=SticksAndBall(bvals, bvecs, d, S0, 
                          angles=[(0, 0),(90,0),(90,90)], 
                          fractions=[33,33,33], snr=snr)
    data[3]=S.copy()
    #three fibers iso
    S,stics=SticksAndBall(bvals, bvecs, d, S0, 
                          angles=[(0, 0),(90,0),(90,90)], 
                          fractions=[23,23,23], snr=snr)
    data[4]=S.copy()
    #three fibers more iso
    S,stics=SticksAndBall(bvals, bvecs, d, S0, 
                          angles=[(0, 0),(90,0),(90,90)], 
                          fractions=[13,13,13], snr=snr)
    data[5]=S.copy()
    #three fibers one at 60
    S,stics=SticksAndBall(bvals, bvecs, d, S0, 
                          angles=[(0, 0),(60,0),(90,90)], 
                          fractions=[33,33,33], snr=snr)
    data[6]=S.copy()    
    
    #three fibers one at 90,90 one smaller
    S,stics=SticksAndBall(bvals, bvecs, d, S0, 
                          angles=[(0, 0),(60,0),(90,90)], 
                          fractions=[33,33,23], snr=snr)
    data[7]=S.copy()
    
    #three fibers one at 90,90 one even smaller
    S,stics=SticksAndBall(bvals, bvecs, d, S0, 
                          angles=[(0, 0),(60,0),(90,90)], 
                          fractions=[33,33,13], snr=snr)
    data[8]=S.copy()
    
    #two fibers one at 0, second 30 
    S,stics=SticksAndBall(bvals, bvecs, d, S0, 
                          angles=[(0, 0),(30,0),(90,90)],
                          fractions=[50,50,0], snr=snr)
    data[9]=S.copy()
    
    #two fibers one at 0, second 30 
    S,stics=SticksAndBall(bvals, bvecs, d, S0, 
                          angles=[(0, 0),(30,0),(90,90)], 
                          fractions=[50,30,0], snr=snr)
    data[10]=S.copy()
    
        
    #1 fiber with 80% isotropic
    S,stics=SticksAndBall(bvals, bvecs, d, S0, 
                          angles=[(0, 0),(30,0),(90,90)], 
                          fractions=[20,0,0], snr=snr)    
    data[11]=S.copy()
    
    #1 fiber with tensor
    evals=np.array([1.4,.35,.35])*10**(-3)
    #evals=np.array([1.4,.2,.2])*10**(-3)
    S=SingleTensor(bvals,bvecs,S0,evals=evals,evecs=np.eye(3),snr=snr)
    data[12]=S.copy()
    
    #2 fibers with two tensors
    evals=np.array([1.4,.35,.35])*10**(-3)
    S=SingleTensor(bvals,bvecs,S0,evals=evals,evecs=np.eye(3),snr=None)
    evals=np.array([.35,1.4,.35])*10**(-3)
    S=SingleTensor(bvals,bvecs,S0,evals=evals,evecs=np.eye(3),snr=None)+S    
    #std=2*S0/snr
    #S=S+np.random.randn(len(S))*std    
    data[13]=S.copy()
    
    #3 fibers with two tensors
    evals=np.array([1.4,.35,.35])*10**(-3)
    S=SingleTensor(bvals,bvecs,S0,evals=evals,evecs=np.eye(3),snr=None)
    evals=np.array([.35,1.4,.35])*10**(-3)
    S=SingleTensor(bvals,bvecs,S0,evals=evals,evecs=np.eye(3),snr=None)+S
    evals=np.array([.35,.35,1.4])*10**(-3)
    S=SingleTensor(bvals,bvecs,S0,evals=evals,evecs=np.eye(3),snr=None)+S
        
    #std=2*S0/snr
    #S=S+np.random.randn(len(S))*std    
    data[14]=S.copy()
    
    evals=np.array([.35,.35,.35])*10**(-3)
    S=SingleTensor(bvals,bvecs,S0,evals=evals,evecs=np.eye(3),snr=snr)
    data[15]=S.copy()  
    
    
    
    return data
    
    
if __name__ == '__main__':
    
    '''
    S=np.array([[-1,0,0],[0,1,0],[0,0,1]])
    #T=np.array([[1,0,0],[0,0,]])
    T=np.array([[1,0,0],[-0.76672114,  0.26105963, -0.58650369]])
    print 'yo ',compare_orientation_sets(S,T)
    
    
    btable=np.loadtxt(get_data('dsi515btable'))
    bvals=btable[:,0]
    bvecs=btable[:,1:]
        
    data=create_data(d=0.0015,S0=100,snr=None)
    
    """
    dn=DiffusionNabla(data,bvals,bvecs,odf_sphere='symmetric642',auto=False,save_odfs=True)
    dn.radius=np.arange(0,6,.2)
    dn.radiusn=len(dn.radius)
    dn.create_qspace(bvals,bvecs,16,8)
    dn.peak_thr=.4
    dn.iso_thr=.7
    dn.radon_params(64)
    dn.precompute_interp_coords()
    dn.precompute_fast_coords()
    dn.precompute_equator_indices(5)    
    dn.fit()
    """
    
    dn=DiffusionNabla(data,bvals,bvecs,odf_sphere='symmetric642',
                      auto=False,save_odfs=True,fast=True)
    
    dn.peak_thr=.6
    dn.iso_thr=.9
    dn.precompute_fast_coords()
    dn.precompute_equator_indices(5.)
    dn.fit()
    
    dn2=DiffusionNabla(data,bvals,bvecs,odf_sphere='symmetric642',
                      auto=True,save_odfs=True,fast=False)
       
    
    ds=DiffusionSpectrum(data,bvals,bvecs,odf_sphere='symmetric642',auto=True,save_odfs=True)
    gq=GeneralizedQSampling(data,bvals,bvecs,1.2,odf_sphere='symmetric642',squared=False,auto=False,save_odfs=True)
    gq.peak_thr=.6
    gq.iso_thr=.7
    gq.fit()
               
    blobs=np.zeros((1,4,16,dn.odfn))
    dn_odfs=dn.odfs()
    blobs[0,0]=dn_odfs-np.abs(dn_odfs.min(axis=1)[:,None])
    blobs[0,1]=dn2.odfs()
    blobs[0,2]=ds.odfs()
    gq_odfs=gq.odfs()
    blobs[0,3]=gq_odfs#-gq_odfs.min(axis=1)[:,None]
    #blobs[0,4]=gq_odfs-gq_odfs.min(axis=1)[:,None]    
    #"""
    
    
    peaknos=np.array([0,1,2,3,3,3,3,3,3,2,2,1,1,2,3,0])
    print peaknos
    print np.sum(dn.pk()>0,axis=1),np.sum(np.sum(dn.pk()>0,axis=1)-peaknos),len(peaknos)
    print np.sum(dn2.pk()>0,axis=1),np.sum(np.sum(dn2.pk()>0,axis=1)-peaknos),len(peaknos)        
    print np.sum(ds.qa()>0,axis=1),np.sum(np.sum(ds.qa()>0,axis=1)-peaknos),len(peaknos)
    print np.sum(gq.qa()>0,axis=1),np.sum(np.sum(gq.qa()>0,axis=1)-peaknos),len(peaknos)
    
    np.set_printoptions(3,suppress=True)
    '''
    
    def extended_peak_filtering(odfs,odf_faces):
        new_peaks=[]        
        for (i,odf) in enumerate(odfs):
            peaks,inds=peak_finding(odf,odf_faces)
            dpeaks=np.abs(np.diff(peaks[:3]))
            print '#',i,peaknos[i]
            print peaks[:3]
            print dpeaks
            print odf.min()
            print peaks[:3]/peaks[0]
            print peaks[1:3]/peaks[1]
            print 
            
            """
            ismallp=np.where(dpeaks<2)
            if len(ismallp[0])>0:
                l=ismallp[0][0]
            else:
                l=len(peaks)
            """
            """
            for i in range(len(dpeaks)-1):
                if dpeaks[i]>2 and dpeaks[i+1]<=2:
                    break
            l=i+1
            print peaks[:l]
            """        
    
    #extended_peak_filtering(dn_odfs,dn.odf_faces)        
    #show_blobs(blobs,dn.odf_vertices,dn.odf_faces)
        
    #dn.precompute_fast_coords()
    #dn.precompute_equator_indices(10)    
    #odf=dn.fast_odf(data[1])
    #show equators
    """
    r=fvtk.ren()    
    fvtk.add(r,fvtk.point(dn.odf_vertices[eqinds[0]],fvtk.red))
    fvtk.add(r,fvtk.point(dn.odf_vertices[eqinds[30]],fvtk.green))
    fvtk.show(r)
    """
    
    #odfs=dn.odfs()
    #for odf in odfs:        
    #    peaks,inds=peak_finding(odf,dn.odf_faces)
    #    print 1.,peaks[0]/peaks[1], peaks[0]/peaks[2], np.var(peaks)
    
    '''
    faces = dn.odf_faces
    vertices = dn.odf_vertices
    odf=dn_odfs[13]    

    import pickle
    f = open('dn.dump','w')
    pickle.dump([faces,vertices,odf],f)
    '''

    from dipy.core import meshes

    def new_peaks(odf,faces):
        # DO NOT USE THIS: IT IS VERY SLOW
        # use dominant instead
        np = []
        for point, height in enumerate(odf):
            neighbours = meshes.vertinds_to_neighbors([point], faces)
            #print point, neighbours
            if max(odf[neighbours]) <= height:
                #print point, height, max(odf[neighbours]) 
                np += [point]
            #adjacent_faces = meshes.vertinds_faceinds([point], faces)
            '''
            if max(odf[neighbours]) <= height:
                np += [point]
            '''
        return np
    
    def overlaps_counts(arrays):
        n = len(arrays)
        m = np.zeros((n,n),'int')
        for i in range(n):
            for j in range(n):
                m[i,j] = len(set.intersection(set(arrays[i]),set(arrays[j])))
        return m
    
    def dominant(faces, odf):
        #to find peaks of odf by identiying vertices which are not dominated
        n = len(odf)
        d = np.ones(n, 'int')
        for f in faces:
            s = np.argsort(odf[f])
            d[f[s[:2]]] = 0
        p = np.where(d==1)[0]
        pi = np.argsort(odf[p])[::-1]
        return p[pi]
    
    def overlaps_faces(arrays):
        n = len(arrays)
        m = set([])
        for i in range(n-1):
            for j in range(i+1,n):
                m = set.union(m,set.intersection(set(faces[arrays[i]].ravel()),set(faces[arrays[j]].ravel())))
        return m
    
    import pickle
    f = open('dn.dump','r')
    print 'picked dn reloading ...\n'
    [faces,vertices,odf] = pickle.load(f)
    print 'picked dn reloaded ...\n'
    print 'old peak hunting ...\n'
    peaks,peakinds=peak_finding(odf,faces)
    print 'new peak hunting ...\n'
    newpeaks = dominant(faces, odf)
    #oldpeaks = [305, 317, 171, 170, 172, 169,  40,  45,   2]
    newpeaks = [2, 40, 45, 169, 170, 171, 172, 305, 317, 323, 361, 366, 490, 491, 492, 493, 626, 638]
    #newpeaks = [2, 40, 45, 169, 170, 171, 172, 305][::-1]

    print 'starting green paint job ...\n'

    #print 'old peaks', peakinds
    #print 'half the (old) peaks:', oldpeaks, '\n'
    #print 'half the peaks:', newpeaks, '\n'
    print 'all the new peaks:', newpeaks, '\n'
        
    white = 0.
    red = 1.
    black = 2.
    green = 3.

    vertex_floods = []
    face_floods = []
    
    vertex_colour = np.zeros(vertices.shape[0])
    face_colour = np.zeros(faces.shape[0])

    for i, peak in enumerate(newpeaks):
    #for i, peak in enumerate(newpeaks):
    #for i, peak in enumerate([newpeaks[0]]):
                
        vertex_colour[peak]=green

        uav, uaf = meshes.adjacent_uncoloured([peak], vertex_colour, face_colour, faces)
        for uf in uaf:
            face_colour[uf] = green
            vertex_colour[faces[uf]] = green
                
        green_vertices = np.where(vertex_colour == green)[0]
        green_faces = np.where(face_colour == green)[0]

        orig = len(green_vertices)
        
        while True:
#        for j in range(2):
               
            #print 'green_vertices', len(green_vertices), green_vertices, odf[green_vertices]
            #print 'green_faces', len(green_faces), green_faces, [list(faces[g]) for g in green_faces]

            uav, uaf = meshes.adjacent_uncoloured(green_vertices, vertex_colour, face_colour, faces)
                    
            candidate_green_faces=[]
            candidate_green_vertices=[]
            
            for uf in uaf:
                if np.sum(vertex_colour[faces[uf]] == green) == 2:
                    #print uf, vertex_colour[faces[uf]], list(faces[uf]), list(odf[faces[uf]])
                    new_vertex = list(faces[uf][list(np.where(vertex_colour[faces[uf]]!=green)[0])])[0]
                    if odf[new_vertex] <= np.min(odf[faces[uf]]) and vertex_colour[new_vertex] != red:
                        #print uf, odf[faces[uf]], new_vertex, odf[new_vertex]
                        candidate_green_faces += [uf]
                        candidate_green_vertices += [new_vertex]
                elif np.sum(vertex_colour[faces[uf]] == green) == 3:  
                        candidate_green_faces += [uf]
                    
            #print 'candidate vertices', len(candidate_green_vertices), candidate_green_vertices
            #print 'candidate faces', len(candidate_green_faces), candidate_green_faces
             
            #print '-----------------\n'
            
            face_colour[candidate_green_faces] = green
            vertex_colour[candidate_green_vertices] = green
                
            green_vertices = np.where(vertex_colour == green)[0]
            green_faces = np.where(face_colour == green)[0]
            
            if len(green_vertices) == orig:
                break
            else:
                orig = len(green_vertices)

        #print 'green_vertices', len(green_vertices), green_vertices, odf[green_vertices], '\n'
        #print 'green_faces', len(green_faces), green_faces, [list(faces[g]) for g in green_faces], '\n'
        print 'peak', i+1, 'at', peak, ' at height', odf[peak], 'covers', len(green_faces), 'faces'     
        vertex_floods += [green_vertices]
        vertex_colour[green_vertices] = red
        face_floods += [green_faces]
        face_colour[green_faces] = red
    '''
    vertex_lengths = map(len,vertex_floods)
    print vertex_lengths, sum(vertex_lengths)
    print overlaps_counts(vertex_floods)    
    '''

    face_lengths = map(len,face_floods)
    print face_lengths, sum(face_lengths)
    '''
    print overlaps_counts(face_floods)    
    '''
    
    overlaps = list(overlaps_faces(face_floods))
    
    colors = np.zeros((1,1,1,len(vertices)))
    for jj, ff in enumerate(face_floods):
        for v3 in faces[ff]:
            for v in v3:
                colors[0,0,0,v] = jj + 1

    colors[0,0,0,overlaps] = len(face_floods)
                
    heights = np.zeros((1,1,1,len(vertices)))
    heights[0,0,0,:] = odf

    #print dominant(faces, odf)
    
    color_patches(vertices, faces, heights, colors)
