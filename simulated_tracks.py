import numpy as np
from dipy.viz import fvtk
from dipy.tracking.distances import local_skeleton_clustering, bundles_distances_mdf
from dipy.tracking.metrics import downsample,length
from hcluster import pdist, linkage, dendrogram, squareform
from scikits.learn import cluster
import matplotlib.pyplot as plt

t=np.linspace(-10,10,200)

r=fvtk.ren()

fibno=150

bundle=[]
for i in np.linspace(3,5,fibno):
    pts=5*np.vstack((np.cos(t),np.sin(t),t/i)).T #helix diverging
    bundle.append(pts)
    #fvtk.add(r,fvtk.line(bundle,fvtk.red))

bundle2=[]
for i in np.linspace(-5,5,fibno):
    pts=np.vstack((np.cos(t),t,i*np.ones(t.shape))).T # parallel waves
    bundle2.append(pts)
    #fvtk.add(r,fvtk.line(bundle2,fvtk.green))

#print(bundles_distances_mdf(bundle2,bundle2)[0])

bundle2r=[]
index=np.random.permutation(range(len(bundle2)))
for i in index:
    bundle2r.append(bundle2[i])

bundle3=[]
for i in np.linspace(-1,1,fibno):
    pts=np.vstack((i**3*t/2.,t,np.cos(t))).T # spider - diverging
    bundle3.append(pts)
    #fvtk.add(r,fvtk.line(bundle3,fvtk.azure))
    
bundle4=[]
for i in np.linspace(-1,1,fibno):    
    #pts4=np.vstack((0*t+i*np.cos(.2*t),0*t,t)).T
    pts=2*np.vstack((0*t+2*i*np.cos(.2*t),np.cos(.4*t),t)).T # thick neck   
    bundle4.append(pts)
    #fvtk.add(r,fvtk.line(bundle4,fvtk.gray))
    
def show_zero_level(r,bundle,dist):

    T=[downsample(b,12) for b in bundle]
    C=local_skeleton_clustering(T,dist)
    vs=[]
    colors=np.zeros((len(T),3))
    for c in C:
        vs.append(C[c]['hidden']/C[c]['N'])
        color=np.random.rand(3,)
        #fvtk.add(r,fvtk.line(vs,color,linewidth=4.5))
        for i in C[c]['indices']:
            colors[i]=color
            fvtk.label(r,text=str(i),pos=(bundle[i][-1]),scale=(.5,.5,.5),color=(color[0],color[1],color[2]))    
    fvtk.add(r,fvtk.line(T,colors,linewidth=2.))    

def taleton_old(T,dist=[8.]):
    
    Cs=[]
    id=0
    C=local_skeleton_clustering(T,dist[id])
    Cs.append(C)    
    vs=[C[c]['hidden']/float(C[c]['N']) for c in C]    
    #ls=[len(C[c]['indices']) for c in C]    
    not_converged=1
    id+=1
    while id<len(dist):                         
        C=local_skeleton_clustering(vs,dist[id])
        vs=[C[c]['hidden']/float(C[c]['N']) for c in C]
        #ls=[len(C[c]['indices']) for c in C]
        Cs.append(C)
        id+=1            
    return Cs

def line(vs,color,centered=False):
    try:
        if color.ndim==2:
            cols=np.random.rand(len(vs),4)
            cols[:]=np.asarray(color).astype('f4')
    except:    
        cols=np.random.rand(len(vs),4)
        cols[:]=np.asarray(color).astype('f4')
          
    c=InteractiveCurves(vs,colors=cols,centered=centered,line_width=4.)
    return c


def shift(vs,s):    
   return [v+s for v in vs]

#from LSC_limits import taleton,bring_virtuals

#show_zero_level(r,bundle2r,1.2)
#Cs=taleton_old(bundle2r)

bun=bundle2+bundle3 # merging

bun2=bundle+bundle2+bundle3+bundle4

bun3=bundle+bundle3+bundle4

from fos import World, Window, DefaultCamera
from fos.actor.curve import InteractiveCurves

"""
w=World()
wi=Window(bgcolor=(1.,1.,1.,1.),width=1000,height=1000)
wi.attach(w)    
w.add(line(bundle,[1,0,0,1]))
w.add(line(bundle3,[0,1,0,1]))
w.add(line(bundle4,[0,0,1,1]))
print len(bun3)
"""

r=fvtk.ren()

r.SetBackground(1,1,1.)

fvtk.add(r,fvtk.line(bundle,fvtk.red,linewidth=3))
fvtk.add(r,fvtk.line(bundle3,fvtk.green,linewidth=3))
fvtk.add(r,fvtk.line(bundle4,fvtk.blue,linewidth=3))
fvtk.show(r,size=(800,800))


from LSC_limits import bring_virtuals

Td=[downsample(t,80) for t in bun3]

C8=local_skeleton_clustering(Td,8)
vs,ls,tot=bring_virtuals(C8)
vs2=shift(vs,np.array([0,0,0],'f4'))

"""
wi2=Window(bgcolor=(1.,1.,1.,1.),width=1000,height=1000)
wi3=Window(bgcolor=(1.,1.,1.,1.),width=1000,height=1000)
w2=World()
w3=World()
wi2.attach(w2)
wi3.attach(w3)
w2.add(line(vs2,np.array([[1,0,1,1],[0,1,0,1]],'f4')))
"""

fvtk.clear(r)
fvtk.add(r,fvtk.line(vs2,np.array([[1,0,1],[0,1,0]],'f4'),linewidth=3))
fvtk.show(r,size=(800,800))


C2=local_skeleton_clustering(Td,1)
vs,ls,tot=bring_virtuals(C2)
vs2=shift(vs,np.array([0,0,0],'f4'))
"""
w3.add(line(vs2,np.array([[1,0,0,1],[1,0,0,1],[0,1,0,1],[0,1,0,1],[0,1,0,1],[0,0,1,1],[0,0,1,1],[0,0,1,1]],'f4') ))
"""
fvtk.clear(r)
fvtk.add(r,fvtk.line(vs2,np.array([[1,0,0],[1,0,0],[0,1,0],[0,1,0],[0,1,0],[0,0,1],[0,0,1],[0,0,1]],'f4'),linewidth=3))
fvtk.show(r,size=(800,800))


#fvtk.add(r,fvtk.line(bundle2,fvtk.blue))
#fvtk.add(r,fvtk.line(bundle3,fvtk.red))
#fvtk.show(r)

#bun=[downsample(b,12) for b in bun]

#Cs=taleton(bundle+bundle2+bundle3+bundle4,10*[1.])
#Cs=taleton(bun,10*[1.4])

#print len(Cs)

#C0=Cs[0]
#vs0=[C0[c]['hidden']/float(C0[c]['N']) for c in C0]
#print len(vs0)
#vs0_,ls0_,tot0_=bring_virtuals(C0)

#C1=Cs[1]
#vs1=[C1[c]['hidden']/float(C1[c]['N']) for c in C1]
#print len(vs1)
#vs1_,ls1_,tot1_=bring_virtuals(C1)

"""
r=fvtk.ren()
#fvtk.add(r,fvtk.line(vs0,fvtk.green,linewidth=2.))
#for i,v in enumerate(vs0):
#    fvtk.add(r,fvtk.label(r,str(i),pos=v[0],scale=.3,color=fvtk.green ))

fvtk.add(r,fvtk.line(vs1,fvtk.blue,linewidth=3.))
for i,v in enumerate(vs1):
    fvtk.add(r,fvtk.label(r,str(i),pos=v[-1],scale=.3,color=fvtk.blue ))

#print vs1.shape

#fvtk.add(r,fvtk.line(bundle+bundle2+bundle3+bundle4,fvtk.red))
#fvtk.add(r,fvtk.line(bundle2+bundle3,fvtk.red))

#fvtk.add(r,fvtk.line(bundle2,fvtk.green))
#fvtk.add(r,fvtk.line(bundle3,fvtk.blue))
#fvtk.add(r,fvtk.line(bundle4,fvtk.azure))

fvtk.show(r)
"""


"""

D=bundles_distances_mdf(bundle2,bundle2)
#Y=squareform(D)
#Z = linkage(Y,'average')
#dendrogram(Z)

x=np.vstack((t,0*t,0*t)).T
y=np.vstack((0*t,t,0*t)).T
z=np.vstack((0*t,0*t,t)).T
#fvtk.add(r,fvtk.line(x,fvtk.red))
#fvtk.add(r,fvtk.line(y,fvtk.green))
#fvtk.add(r,fvtk.line(z,fvtk.blue))
r.SetBackground(1,1,1)

fvtk.show(r)

n_samples = 300

# generate random sample, two components
np.random.seed(0)
C = np.array([[0., -0.7], [3.5, .7]])
X = np.r_[np.dot(np.random.randn(n_samples, 2), C),
          np.random.randn(n_samples, 2) + np.array([20, 20])]

#plt.plot()
centroids,labels,intertia=cluster.k_means(X,2)
centers, labels = cluster.mean_shift(X)
#labels, centers = cluster.spectral_clustering()
centeris,labels=cluster.affinity_propagation(-D)

"""



