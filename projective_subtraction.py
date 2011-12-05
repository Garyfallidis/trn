import numpy as np
from dipy.viz import fvtk
from prepare_phantoms import get_data,codebook,simulations_dipy,draw_needles
from dipy.core.triangle_subdivide import create_unit_sphere,create_half_unit_sphere
from sphere_tools import random_uniform_on_sphere

CNT=0

def show_different_ds(bvals,bvecs):
    global CNT
    r=fvtk.ren()
    r.SetBackground(1.,1.,1.)
    Signals=[]
    for d in [0.0005,0.0010,0.0015,0.0020,0.0025,0.0030,0.0035,0.0040]:
        S1,sticks1=simulations_dipy(bvals,bvecs,d=d,S0=100,angles=[(0,0),(45,0),(90,90)],fractions=[100,0,0],snr=None)
        Signals.append(S1[1:]/S1[0])
    show_signals(r,Signals,bvecs)
    CNT=0
    draw_needles(r,sticks1,100,2,off=np.array([0,0,0]))

    Signals=[]
    for d in [0.0005,0.0010,0.0015,0.0020,0.0025,0.0030,0.0035,0.0040]:
        S1,sticks1=simulations_dipy(bvals,bvecs,d=d,S0=100,angles=[(0,0),(45,0),(90,90)],fractions=[50,50,0],snr=None)
        Signals.append(S1[1:]/S1[0])
    show_signals(r,Signals,bvecs,-200)
    CNT=0
    
    Signals=[]
    for d in [0.0005,0.0010,0.0015,0.0020,0.0025,0.0030,0.0035,0.0040]:
        S1,sticks1=simulations_dipy(bvals,bvecs,d=d,S0=100,angles=[(0,0),(45,0),(90,90)],fractions=[33,33,33],snr=None)
        Signals.append(S1[1:]/S1[0])
    show_signals(r,Signals,bvecs,-400)
    """
    Signals=[]
    for d in [0.0005,0.0010,0.0015,0.0025,0.0030,0.0035,0.0040]:
        S1,sticks1=simulations_dipy(bvals,bvecs,d=d,S0=100,angles=[(0,0),(90,0),(90,90)],fractions=[20,50,0],snr=None)
        Signals.append(S1[1:]/S1[0])
    show_signals(r,Signals,bvecs,-600)
    
    Signals=[]
    for d in [0.0005,0.0010,0.0015,0.0025,0.0030,0.0035,0.0040]:
        S1,sticks1=simulations_dipy(bvals,bvecs,d=d,S0=100,angles=[(0,0),(90,0),(90,90)],fractions=[10,50,0],snr=None)
        Signals.append(S1[1:]/S1[0])
    show_signals(r,Signals,bvecs,-800)
        
    Signals=[]
    for d in [0.0005,0.0010,0.0015,0.0025,0.0030,0.0035,0.0040]:
        S1,sticks1=simulations_dipy(bvals,bvecs,d=d,S0=100,angles=[(0,0),(90,0),(90,90)],fractions=[0,50,0],snr=None)
        Signals.append(S1[1:]/S1[0])
    show_signals(r,Signals,bvecs,-1000)
    
    """
            
    fvtk.show(r)



def needlebook(bvals,bvecs,d=0.0015,steps=21,subdiv=3,fractions=[1.]):
    """ Single needles book    
    """            
    sticks,e,t=create_half_unit_sphere(subdiv)    
    CBK=[]; STK=[]; FRA=[]; REG=[]    
    def single_needles(CBK,STK,FRA,bvals,bvecs,sticks,fraction):
        for s in sticks:    
            S=np.zeros(len(bvecs)-1)
            for (i,g) in enumerate(bvecs[1:]):
                S[i]=(1-fraction)*np.exp(-bvals[i+1]*d)+fraction*np.exp(-bvals[i+1]*d*np.dot(s,g)**2)
            CBK.append(S)
            STK.append(s)
            FRA.append(fraction)         
    #fractions=np.linspace(0,1,steps)    
    for f in fractions:    
        single_needles(CBK,STK,FRA,bvals,bvecs,sticks,fraction=f)
    CBK=np.array(CBK)
    STK=np.array(STK)    
    FRA=np.array(FRA)    
    return CBK,STK,FRA   

def psi(S,CBK,STK,FRA):
    """ Projective Subtraction Diffusion MRI
    
    Parameters
    -----------
    S: normalized signal
    CBK: signal codebook
    STK: codebook needles
    
    Returns
    --------
    A: approximated orientation
    
    """
    #N=S/np.float(S[0])    
    values=[]
    for i in range(len(CBK)):
        D=S-CBK[i]        
        X=np.dot(np.diag(np.concatenate([D,D])),Bvecs)
        mX=np.mean(np.sqrt(X[:,0]**2+X[:,1]**2+X[:,2]**2))        
        #values.append(np.var(D))
        values.append(mX)
    values=np.array(values)
    return values

def less90(angle):
    if angle > 90:
        return 180-angle
    else:
        return angle    

def angular_distances(STK,sticks):
    A=np.zeros((len(STK),6))    
    for i in range(len(STK)):        
        A[i,:3]=STK[i,:]
        A[i,3]=less90(np.rad2deg(np.arccos(np.dot(STK[i],sticks[0]))))
        A[i,4]=less90(np.rad2deg(np.arccos(np.dot(STK[i],sticks[1]))))
        A[i,5]=less90(np.rad2deg(np.arccos(np.dot(STK[i],sticks[2]))))        
    return A
    

def random_simulations(no,bvals,gradients,d=0.0015,fractions=[100,0,0],snr=None):    
    sticks=random_uniform_on_sphere(n=no,coords='xyz')      
    fractions=[f/100. for f in fractions]
    f0=1-np.sum(fractions)
    S=np.zeros(len(gradients)-1)        
    for (i,g) in enumerate(gradients[1:]):
        S[i]=f0*np.exp(-bvals[i+1]*d)+ np.sum([fractions[j]*np.exp(-bvals[i+1]*d*np.dot(s,g)**2) for (j,s) in enumerate(sticks)])
    if snr!=None:
        std=S/snr
        S=S+np.random.randn(len(S))*std
    return S,sticks


def single_stick_simulations(no,bvals,bvecs,d,fractions,snr):    
    Bvecs=np.concatenate([bvecs[1:],-bvecs[1:]])
    res=[]
    for i in range(no):    
        S0,sticks0=random_simulations(3,bvals,bvecs,d,fractions=fractions,snr=snr)
        X0=np.dot(np.diag(np.concatenate([S0,S0])),Bvecs)        
        values=psi(S0,CBK,STK,FRA)
        A=angular_distances(STK[values.argsort()],sticks0)
        res.append(A[0,3])    
    return res

def show_signals(r,Signals,bvecs,offset=0):
    #r=fvtk.ren()
    global CNT
    Bvecs=np.concatenate([bvecs[1:],-bvecs[1:]])
    for (i,S) in enumerate(Signals):
        X=np.dot(np.diag(np.concatenate([50*S,50*S])),Bvecs)
        mX=np.mean(np.sqrt(X[:,0]**2+X[:,1]**2+X[:,2]**2))
        print CNT,mX,np.var(S)
        CNT=CNT+1
        fvtk.add(r,fvtk.point(X+np.array([(i+1)*200,offset,0]),fvtk.green,1,2,6,6))
    #fvtk.show(r)

if __name__=='__main__':
    
    np.set_printoptions(4,suppress=True)
    data,bvals,bvecs=get_data(name='118_32',par=0)    
    Bvecs=np.concatenate([bvecs[1:],-bvecs[1:]])    
    
    S0,sticks0=simulations_dipy(bvals,bvecs,d=0.0015,S0=100,angles=[(0,0),(90,0),(90,90)],fractions=[100,0,0],snr=None)
    X0=np.dot(np.diag(np.concatenate([S0[1:],S0[1:]])),Bvecs)    
    
    #CBK,STK,FRA=needlebook(bvals,bvecs,d=0.0015,subdiv=4,fractions=[.8])
    #res0=single_stick_simulations(200,bvals,bvecs,d=0.0015,fractions=[80,0,0],snr=None)
    
    #CBK,STK,FRA=needlebook(bvals,bvecs,d=0.0015,subdiv=3,fractions=[.2])
    #res1=single_stick_simulations(200,bvals,bvecs,d=0.0015,fractions=[80,0,0],snr=None)

    #res0=single_stick_simulations(1000,bvals,bvecs,d=0.0015,fractions=[100,0,0],snr=None)
    #res1=single_stick_simulations(1000,bvals,bvecs,d=0.0015,fractions=[80,0,0],snr=None)
    #res2=single_stick_simulations(1000,bvals,bvecs,d=0.0015,fractions=[60,0,0],snr=None)
    #res3=single_stick_simulations(1000,bvals,bvecs,d=0.0015,fractions=[40,0,0],snr=None)
    #res4=single_stick_simulations(1000,bvals,bvecs,d=0.0015,fractions=[20,0,0],snr=None)
    
#    values=psi(S0[1:]/S0[0],CBK,STK,FRA)
#    r=fvtk.ren()
#    fvtk.add(r,fvtk.point(X0,fvtk.yellow,1,2,8,8))
#    draw_needles(r,sticks0,100,2)
#    for (i,S) in enumerate((CBK[values.argsort()])):
#        NS=S0[1:]-S0[0]*S
#        X=np.dot(np.diag(np.concatenate([NS,NS])),Bvecs)
#        fvtk.add(r,fvtk.point(X+np.array([(i+1)*200,0,0]),fvtk.green,1,2,8,8))                
#        #draw_needles(r,sticks0,100,2,off=np.array([(i+1-14)*200,0,0]))    
#    fvtk.show(r)



