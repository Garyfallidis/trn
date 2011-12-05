import pbc
from dipy.viz import fos

#path='/home/eg01/Data/PBC/pbc2009icdm'
path='/home/eg309/Data/PBC/pbc2009icdm'

skeletal=pbc.load_pickle(path+'/skeletal_5000_11.pkl') 
tracks=pbc.load_approximate_tracks(path,1,1)

r=fos.ren()

tracksS=[tracks[i] for i in skeletal]

SL=fos.line(tracksS,fos.red,opacity=0.4)

fos.add(r,SL)
fos.show(r)
