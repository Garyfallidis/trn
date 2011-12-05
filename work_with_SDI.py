import numpy as np
from dipy.data import get_data,get_sphere
from dipy.reconst.dandelion import SphericalDandelion
from dipy.reconst.recspeed import peak_finding
from dipy.reconst.gqi import GeneralizedQSampling
from dipy.data import get_sim_voxels


eds=np.load(get_sphere('symmetric362'))
vertices=eds['vertices']
faces=eds['faces']

fibs=['fib0','fib1','fib2']

for fib in fibs:

    dix=get_sim_voxels(fib)
    
    data=dix['data']
    bvals=dix['bvals']
    gradients=dix['gradients']
    
    no=10
    
    print(bvals.shape, gradients.shape, data.shape)    
    print(dix['fibres'])
    
    np.set_printoptions(2)
    for no in range(len(data)):
    
        sd=SphericalDandelion(data,bvals,gradients)    
        sdf=sd.spherical_diffusivity(data[no])    
    
        gq=GeneralizedQSampling(data,bvals,gradients)
        sodf=gq.odf(data[no])
                
        #print(faces.shape)    
        peaks,inds=peak_finding(np.squeeze(sdf),faces)
        #print(peaks, inds)    
        peaks2,inds2=peak_finding(np.squeeze(sodf),faces)
        #print(peaks2, inds2)    
        print 'sdi',inds,'sodf',inds2, vertices[inds[0]]-vertices[inds2[0]]     
        #print data[no]

