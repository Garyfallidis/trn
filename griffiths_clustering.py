import numpy as np
from dipy.data import get_data
from dipy.viz import fvtk
import nibabel.trackvis as tv
from dipy.tracking import metrics as tm
from dipy.tracking import distances as td

fnx=get_data('fornix')
print(fnx)
streams,hdr=tv.read(fnx)

#list comprehension
#T=[s[0] for s in streams]
#same as
T=[]
for s in streams:
    T.append(s[0])


r=fvtk.ren()
linea=fvtk.line(T,fvtk.red)
fvtk.add(r,linea)
fvtk.show(r)


#for more complicated visualizations use mayavi
#or the new fos when released

dT=[tm.downsample(t,10) for t in T]
C=td.local_skeleton_clustering(dT,d_thr=5)

ldT=[tm.length(t) for t in dT]
#average length
avg_ldT=sum(ldT)/len(dT)
print(avg_ldT)

"""
r=fvtk.ren()
#fvtk.clear(r)
colors=np.zeros((len(T),3))
for c in C:
    color=np.random.rand(1,3)
    for i in C[c]['indices']:
        colors[i]=color
fvtk.add(r,fvtk.line(T,colors,opacity=1))
fvtk.show(r)
"""

r=fvtk.ren()



bundle=[]
for i in C[2]['indices']:
    bundle.append(T[i])
         
si,s=td.most_similar_track_mam(bundle,'avg')
fvtk.add(r,fvtk.line(bundle,fvtk.red))
fvtk.add(r,fvtk.line(bundle[si],fvtk.cyan))
fvtk.show(r)





