path = '/home/ian/Data/PBC/pbc2009icdm'
#path =  '/home/eg01/Data/PBC/pbc2009icdm'
import pbc
from dipy.viz import fos
tract = 2
G,hdr = pbc.load_training_set(path)
b1=G[1]['tracks']
from dipy.core import track_metrics as tm
si,s = tm.most_similar_track_mam([b1[t] for t in range(0,2800,10)])
print si
reference = b1[range(0,2800,10)[si]]
index = len(reference)/2
p = reference[index]
along = reference[index+1]-reference[index]
import numpy as np
normal=along/np.sqrt(np.inner(along,along))
crossings = list([])
hit_div = list([])
for k in range(len(b1)):
    t = b1[k]
    cross= -1
    for i in range(len(t))[:-1]:
        q = t[i]
        r = t[i+1]
        if np.inner(normal,q-p)*np.inner(normal,r-p) <= 0:
#            print "Segment %d of track %d crosses the normal plane" % (i,k)
            cross = i
            crossings.append([k,cross])
            if np.inner((r-q),normal) != 0:
                alpha = np.inner((p-q),normal)/np.inner((r-q),normal)
                hit = q+alpha*(r-q)
                divergence = (r-q)-np.inner(r-q,normal)*normal
                hit_div.append([hit,divergence])
            else:
                hit_div.append([hit,0])
            break
#    if cross<0:
#        print "No crossing segment"
#    if cross >= 0:
print "%d tracks cross the plane" % (len(crossings))
r = fos.ren()
fos.add(r,fos.points(np.array([h[0] for h in hit_div])))
fos.show()
