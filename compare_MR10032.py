from time import time

import numpy as np

from nibabel.dicom import dicomreaders as dcm

import dipy.core.generalized_q_sampling as gq
import dipy.core.dti as dt
import dipy.core.track_propagation as tp
from dipy.viz import fos

'''

dname_101='/home/eg01/Data_Backup/Data/MR10032/CBU101204_MR10032/20100914_184223/Series_003_ep2d_advdiff_DSI_25x25x25_b4000'

dname_118='/home/eg01/Data_Backup/Data/MR10032/CBU101204_MR10032/20100914_184223/Series_004_ep2d_advdiff_DTI_25x25x25_STEAM_118dir_b1000'

dname_64='/home/eg01/Data_Backup/Data/MR10032/CBU101204_MR10032/20100914_184223/Series_004_ep2d_advdiff_DTI_25x25x25_STEAM_118dir_b1000'

'''

dname_101='/home/eg01/Data/dipy_data/MR10032/CBU101223_MR10032/20100917_080125/Series_002_ep2d_advdiff_DSI_101_25x25x25_STEAM_b4000'

dname_118='/home/eg01/Data/dipy_data/MR10032/CBU101205_MR10032/20100914_194720/Series_004_ep2d_advdiff_DTI_25x25x25_STEAM_118dir_b1000'

dname_64='/home/eg01/Data/dipy_data/MR10032/CBU101205_MR10032/20100914_194720/Series_003_CBU_DTI_64InLea_2x2x2'


series=[dname_101,dname_118,dname_64]

ALL_T=[]

for dname in series:
    
    
    t1=time()

    data,affine,bvals,gradients=dcm.read_mosaic_dir(dname)

    t2=time()
    print ('load data in %d secs' %(t2-t1))

    x,y,z,g = data.shape

    print('data shape is ',data.shape)

    #calculate QA
    gqs=gq.GeneralizedQSampling(data,bvals,gradients)
    print('gqs.QA.shape ',gqs.QA.shape)


    t3=time()
    print ('Generate QA in %d secs' %(t3-t2))

    T=tp.FACT_Delta(gqs.QA,gqs.IN,seeds_no=10000).tracks
    t4=time()
    print ('Create %d QA tracks in %d secs' %(len(T),t4-t3))

    #calculate single tensor
    ten=dt.Tensor(data,bvals,gradients,thresh=50)
    t5=time()
    print('Create FA in %d secs' %(t5-t4))

    T2=tp.FACT_Delta(ten.FA,ten.IN,seeds_no=10000,qa_thr=0.2).tracks

    t6=time()
    print ('Create %d FA tracks in %d secs' %(len(T2),t6-t5))

    T2=[t+np.array([100,0,0]) for t in T2]

    print dname
    print('Red tracks propagated based on QA')
    print('Green tracks  propagated based on FA')

    r=fos.ren()
    fos.add(r,fos.line(T,fos.red))
    fos.add(r,fos.line(T2,fos.green))
    fos.show(r)

    ALL_T.append((T,T2))

    
