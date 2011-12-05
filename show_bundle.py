path =  '/home/eg01/Data/PBC/pbc2009icdm'
import pbc
from dipy.viz import fos
tract = 2
G,hdr = pbc.load_training_set(path)
r = fos.ren()
fos.add(r,fos.line(G[tract]['tracks'],fos.blue,opacity=0.1))
fos.show(r)