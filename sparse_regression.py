import numpy as np
from dipy.viz import fvtk
from prepare_phantoms import get_data,codebook,simulations_dipy,draw_needles

data,bvals,bvecs=get_data(name='118_32',par=0)
CBK,STK,FRA,REG=codebook(bvals,bvecs,fractions=True)
Bvecs=np.concatenate([bvecs[1:],-bvecs[1:]])

S0,sticks0=simulations_dipy(bvals,bvecs,d=0.0015,S0=100,angles=[(0,0),(90,0),(90,90)],fractions=[100,0,0],snr=None)
S1,sticks1=simulations_dipy(bvals,bvecs,d=0.0015,S0=100,angles=[(0,0),(90,0),(90,90)],fractions=[50,50,0],snr=None)
S2,sticks2=simulations_dipy(bvals,bvecs,d=0.0015,S0=100,angles=[(0,0),(45,0),(90,90)],fractions=[50,50,0],snr=None)

SIN=CBK[:33]

r=fvtk.ren()

X0=np.dot(np.diag(np.concatenate([S0[1:],S0[1:]])),Bvecs)
fvtk.add(r,fvtk.point(X0,fvtk.red,1,2,8,8))
X1=np.dot(np.diag(np.concatenate([S1[1:],S1[1:]])),Bvecs)
fvtk.add(r,fvtk.point(X1+np.array([200,0,0]),fvtk.green,1,2,8,8))
X2=np.dot(np.diag(np.concatenate([S2[1:],S2[1:]])),Bvecs)
fvtk.add(r,fvtk.point(X2+np.array([400,0,0]),fvtk.blue,1,2,8,8))


for i in range(0,10):#len(SIN)):
        S=SIN[i]        
        X=np.dot(np.diag(np.concatenate([S[1:],S[1:]])),Bvecs)
        fvtk.add(r,fvtk.point(X+np.array([i*200,-200,0]),fvtk.green,1,2,8,8))
        #draw_needles(r,sticks0,100,2,off=(i+1)*np.array([200,0,0]))
fvtk.show(r)