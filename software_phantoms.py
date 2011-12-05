import numpy as np
import nibabel as nib
from dipy.reconst.dti import Tensor
from dipy.reconst.gqi import GeneralizedQSampling
from dipy.reconst.dsi import DiffusionSpectrum
from dipy.tracking.propagation import EuDX
from dipy.tracking.distances import local_skeleton_clustering
from dipy.tracking.metrics import downsample, length
from dipy.viz import fvtk
from dipy.viz import colormap as cm
from dipy.data import get_sphere
from dipy.reconst.recspeed import peak_finding 

phantoms='mc'

if phantoms=='mc':
    dname='/home/eg309/Data/Software_Phantoms/'
    #img=nib.load(dname+'bezier_phantom_2curves_SNR30_1.4_0.35_0.35.hdr')
    img=nib.load(dname+'bezier_phantom_2lines_SNR30_1.4_0.35_0.35.hdr')      
    data=img.get_data()
    print data.shape
    #imshow(data[:,:,5,0])
    aff=img.get_affine()
    #?Tensor
    dirs=np.loadtxt(dname+'DSI_101dir')
    bvecs=dirs[:,2:]
    bvals=dirs[:,1]
    mask=data[:,:,:,0] > 2
    length_thr=0
    virtual_thr=0
    fa_thr=0.2
    qa_thr=0.0239
    
    
if phantoms=='fc':
    dname='/home/eg309/Data/Fibre_Cup/3x3x3/'
    img=nib.load(dname+'dwi-b1500.nii.gz')
    data=img.get_data()
    print data.shape
    aff=img.get_affine()
    bvecs=np.loadtxt(dname+'diffusion_directions_corrected.txt')
    bvals=1500*np.ones(len(bvecs))
    mask=data[:,:,:,0] > 50
    length_thr=0
    virtual_thr=0
    fa_thr=0.05
    qa_thr=0.0239
    


def show(T,A,IND,VERTS,scale):
    
    r=fvtk.ren()
    fvtk.clear(r)
    fvtk.add(r,fvtk.line(T,fvtk.red))
    fvtk.show(r)
    
    Td=[downsample(t,20) for t in T]
    C=local_skeleton_clustering(Td,3)
    fvtk.clear(r)
    lent=float(len(T))
    
    for c in C:
        color=np.random.rand(3)
        virtual=C[c]['hidden']/float(C[c]['N'])
        if length(virtual)> virtual_thr: 
            linewidth=100*len(C[c]['indices'])/lent
            if linewidth<1.:
                linewidth=1
            #fvtk.add(r,fvtk.line(virtual,color,linewidth=linewidth))
            #fvtk.add(r,fvtk.label(r,str(len(C[c]['indices'])),pos=virtual[0],scale=3,color=color ))
        #print C[c]['hidden'].shape
    
    print A.shape
    print IND.shape
    print VERTS.shape
    
    all,allo=fvtk.crossing(A,IND,VERTS,scale,True)
    colors=np.zeros((len(all),3))
    for (i,a) in enumerate(all):
        if allo[i][0]==0 and allo[i][1]==0 and allo[i][2]==1:
            pass
        else:            
            colors[i]=cm.boys2rgb(allo[i])
    
    fvtk.add(r,fvtk.line(all,colors))    
    fvtk.show(r)


ten=Tensor(data,bvals,bvecs,mask)
FA=ten.fa()
FA[np.isnan(FA)]=0
eu=EuDX(FA,ten.ind(),seeds=10**4,a_low=fa_thr,length_thr=length_thr)
T =[e for e in eu]
#show(T,FA,ten.ind(),eu.odf_vertices,scale=1)

#r=fvtk.ren()
#fvtk.add(r,fvtk.point(eu.odf_vertices,cm.orient2rgb(eu.odf_vertices),point_radius=.5,theta=30,phi=30))
#fvtk.show(r)                     

gqs=GeneralizedQSampling(data,bvals,bvecs,Lambda=1.,mask=mask,squared=False)
eu2=EuDX(gqs.qa(),gqs.ind(),seeds=10**4,a_low=0,length_thr=length_thr)
T2=[e for e in eu2]
show(T2,gqs.qa(),gqs.ind(),eu2.odf_vertices,scale=1)

ds=DiffusionSpectrum(data,bvals,bvecs,mask=mask)
eu3=EuDX(ds.gfa(),ds.ind()[...,0],seeds=10**4,a_low=0,length_thr=length_thr)
T3=[e for e in eu3]
#show(T3,ds.gfa(),ds.ind()[...,0],eu3.odf_vertices,scale=1)

eu4=EuDX(ds.nfa(),ds.ind(),seeds=10**4,a_low=0,length_thr=length_thr)
T4=[e for e in eu4]
#show(T4,ds.nfa(),ds.ind(),eu4.odf_vertices,scale=1)

eu5=EuDX(ds.qa(),ds.ind(),seeds=10**4,a_low=0,length_thr=length_thr)
T5=[e for e in eu5]
#show(T5,ds.qa(),ds.ind(),eu5.odf_vertices,scale=1)


"""
#data[mask == False]=0
#ind=np.zeros(data.shape)
#ind[:,:,:]=np.arange(0,data.shape[-1])

#ind=ind.astype(np.uint8)

fname=get_sphere('symmetric362')
sph=np.load(fname)
verts=sph['vertices']
faces=sph['faces']

#s=data[24,34,5,1:]
#s=data[47,46,5,1:]

#curves
#point=(24,34,5)
#point=(47,46,5)
#point=(11,8,5)
#lines
point=(30,36,5)
#point=(42,23,5)
#point=(46,52,5)

i,j,k=point

s=data[i,j,k,1:]
s0=data[i,j,k,0]

ns=s/float(s0)
ns=100*np.ones(s.shape)

ls=np.log(s)-np.log(s0)
ind=np.arange(1,data.shape[-1])

ob=-1/bvals[1:].astype(np.float)
d=ob*ls
dd=np.abs(ls-ls.mean())

print len(ind)
print len(ls)
print 's',s.min(), s.max(), s.mean() 
print 's0',s0
print 'ns',ns.min(),ns.max(),ns.mean(),ns.std()
print 'd',d.min(),d.max(),d.mean(),d.std()
print 'dd',dd.min(),dd.max(),dd.mean(),dd.std()

r=fvtk.ren()

all=fvtk.crossing(s,ind,bvecs,scale=1)
fvtk.add(r,fvtk.line(all,fvtk.coral))

all2=fvtk.crossing(d,ind,bvecs,scale=10000)
fvtk.add(r,fvtk.line(all2,fvtk.green))

weighting=np.abs(np.dot(bvecs,verts.T))
dw=np.dot(d,weighting[1:,:])
peaks,inds=peak_finding(dw,faces)
colors=fvtk.colors(dw,'jet')

all3=fvtk.crossing(dw,inds,verts,scale=1000)
fvtk.add(r,fvtk.line(all3,fvtk.aquamarine,linewidth=3.))

print '============'
print peaks
print inds
print verts.shape
print colors.shape
print dw.shape
fvtk.add(r,fvtk.point(5*verts,colors,point_radius=.6,theta=10,phi=10))

#all3=fvtk.crossing(dd,ind,bvecs,scale=10000)
#fvtk.add(r,fvtk.line(all3,fvtk.azure))

all4=[]
v=verts[ten.ind()[i,j,k]]
all4.append(10*np.vstack((v,-v)))
fvtk.add(r,fvtk.line(all4,fvtk.red,linewidth=2.))

all5=fvtk.crossing(gqs.qa()[i,j,k],gqs.ind()[i,j,k],verts,50)
print all5
fvtk.add(r,fvtk.line(all5,fvtk.blue,linewidth=10.))
print '=================='
print gqs.qa()[i,j,k]
print gqs.ind()[i,j,k]

fvtk.show(r)
"""






 
