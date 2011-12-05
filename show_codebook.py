import numpy as np
from dipy.viz import fvtk
from prepare_phantoms import get_data,codebook,simulations_dipy
    
    
data,bvals,bvecs=get_data(name='118_32',par=0)
CBK,STK,FRA,REG=codebook(bvals,bvecs,fractions=True)
Bvecs=np.concatenate([bvecs[1:],-bvecs[1:]])
#S0,sticks0=simulations_dipy(bvals,bvecs,d=0.0015,S0=100,angles=[(0,0),(45,0),(90,90)],fractions=[100,0,0],snr=None)

SIN=CBK[:33]

ST0=SIN[0]
ST1=SIN[2]
ST2=SIN[4]

XST0=np.dot(np.diag(np.concatenate([ST0[1:],ST0[1:]])),Bvecs)
XST1=np.dot(np.diag(np.concatenate([ST1[1:],ST1[1:]])),Bvecs)
XST2=np.dot(np.diag(np.concatenate([ST2[1:],ST2[1:]])),Bvecs)

r=fvtk.ren()
#fvtk.add(r,fvtk.point(.5*XST0-1.*XST1,fvtk.green,1,2,8,8))
the_vals = XST0*0.5+XST1*0.5
fvtk.add(r,fvtk.point(XST0,fvtk.green,1,2,8,8))
fvtk.add(r,fvtk.point(XST1,fvtk.red,1,2,8,8))
fvtk.add(r,fvtk.point(the_vals,fvtk.yellow,1,2,8,8))
fvtk.add(r,fvtk.point(XST0-XST1,fvtk.cyan,1,2,8,8))
#fvtk.add(r,fvtk.point(-XST0,fvtk.green,1,2,8,8))

fvtk.show(r)

