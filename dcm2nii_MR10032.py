import os
import numpy as np
import dipy as dp
import nibabel as ni
import resources
import time
from subprocess import Popen,PIPE

def pipe(cmd):    
    p = Popen(cmd, shell=True,stdout=PIPE,stderr=PIPE)
    sto=p.stdout.readlines()
    ste=p.stderr.readlines()
    print(sto)
    print(ste)

def dcm2nii(dname,outdir,filt='*.dcm',options='-d n -g y -i n -o'):
    cmd='dcm2nii '+options +' ' + outdir +' ' + dname + '/' + filt
    print(cmd)
    pipe(cmd)

def dcm2nii_same(dname,filt='*.dcm',options='-d n -g y -i n '):
    cmd='dcm2nii '+options +' ' + dname + '/' + filt
    print(cmd)
    pipe(cmd)


dirs=[['/tmp/MR10032/CBU101204_MR10032/20100914_184223/Series_002_CBU_MPRAGE',
      '/tmp/MR10032/CBU101204_MR10032/20100914_184223/Series_003_ep2d_advdiff_DSI_25x25x25_b4000',
      '/tmp/MR10032/CBU101204_MR10032/20100914_184223/Series_004_ep2d_advdiff_DTI_25x25x25_STEAM_118dir_b1000',
      '/tmp/MR10032/CBU101204_MR10032/20100914_184223/Series_006_CBU_DTI_64InLea_2x2x2'],
      
      ['/tmp/MR10032/CBU101205_MR10032/20100914_194720/Series_002_CBU_MPRAGE',
       '/tmp/MR10032/CBU101223_MR10032/20100917_080125/Series_002_ep2d_advdiff_DSI_101_25x25x25_STEAM_b4000',
       '/tmp/MR10032/CBU101205_MR10032/20100914_194720/Series_004_ep2d_advdiff_DTI_25x25x25_STEAM_118dir_b1000',
       '/tmp/MR10032/CBU101205_MR10032/20100914_194720/Series_003_CBU_DTI_64InLea_2x2x2'],
      
      ['/tmp/MR10032/CBU101252_MR10032/20100922_165954/Series_002_CBU_MPRAGE',
       '/tmp/MR10032/CBU101252_MR10032/20100922_165954/Series_004_ep2d_advdiff_DSI_101_25x25x25_STEAM_b4000',
       '/tmp/MR10032/CBU101252_MR10032/20100922_165954/Series_003_ep2d_advdiff_DTI_25x25x25_STEAM_118dir_b1000',
       '/tmp/MR10032/CBU101252_MR10032/20100922_165954/Series_005_CBU_DTI_64InLea_2x2x2'],
      
      ['/tmp/MR10032/CBU101253_MR10032/20100922_175011/Series_002_CBU_MPRAGE',
       '/tmp/MR10032/CBU101253_MR10032/20100922_175011/Series_003_ep2d_advdiff_DSI_101_25x25x25_STEAM_b4000',
       '/tmp/MR10032/CBU101253_MR10032/20100922_175011/Series_004_ep2d_advdiff_DTI_25x25x25_STEAM_118dir_b1000',
       '/tmp/MR10032/CBU101253_MR10032/20100922_175011/Series_005_CBU_DTI_64InLea_2x2x2'],
      
      ['/tmp/MR10032/CBU101254_MR10032/20100922_183836/Series_002_CBU_MPRAGE',
       '/tmp/MR10032/CBU101254_MR10032/20100922_183836/Series_003_ep2d_advdiff_DSI_101_25x25x25_STEAM_b4000',
       '/tmp/MR10032/CBU101254_MR10032/20100922_183836/Series_004_ep2d_advdiff_DTI_25x25x25_STEAM_118dir_b1000',
       '/tmp/MR10032/CBU101254_MR10032/20100922_183836/Series_005_CBU_DTI_64InLea_2x2x2'],
      
      ['/tmp/MR10032/CBU101284_MR10032/20100929_083205/Series_002_CBU_MPRAGE',
       '/tmp/MR10032/CBU101284_MR10032/20100929_083205/Series_004_ep2d_advdiff_DSI_101_25x25x25_STEAM_b4000',
       '/tmp/MR10032/CBU101284_MR10032/20100929_083205/Series_006_ep2d_advdiff_DTI_25x25x25_STEAM_118dir_b1000',
       '/tmp/MR10032/CBU101284_MR10032/20100929_083205/Series_005_CBU_DTI_64InLea_2x2x2'],
      
      ['/tmp/MR10032/CBU101285_MR10032/20100929_092032/Series_002_CBU_MPRAGE',
       '/tmp/MR10032/CBU101285_MR10032/20100929_092032/Series_005_ep2d_advdiff_DSI_101_25x25x25_STEAM_b4000',
       '/tmp/MR10032/CBU101285_MR10032/20100929_092032/Series_004_ep2d_advdiff_DTI_25x25x25_STEAM_118dir_b1000',
       '/tmp/MR10032/CBU101285_MR10032/20100929_092032/Series_006_CBU_DTI_64InLea_2x2x2'],

      ['/tmp/MR10032/CBU101286_MR10032/20100929_101136/Series_002_CBU_MPRAGE',
       '/tmp/MR10032/CBU101286_MR10032/20100929_101136/Series_006_ep2d_advdiff_DSI_101_25x25x25_STEAM_b4000',
       '/tmp/MR10032/CBU101286_MR10032/20100929_101136/Series_005_ep2d_advdiff_DTI_25x25x25_STEAM_118dir_b1000',
       '/tmp/MR10032/CBU101286_MR10032/20100929_101136/Series_004_CBU_DTI_64InLea_2x2x2'],

      ['/tmp/MR10032/CBU101287_MR10032/20100929_110224/Series_002_CBU_MPRAGE',
       '/tmp/MR10032/CBU101287_MR10032/20100929_110224/Series_003_ep2d_advdiff_DSI_101_25x25x25_STEAM_b4000',
       '/tmp/MR10032/CBU101287_MR10032/20100929_110224/Series_005_ep2d_advdiff_DTI_25x25x25_STEAM_118dir_b1000',
       '/tmp/MR10032/CBU101287_MR10032/20100929_110224/Series_004_CBU_DTI_64InLea_2x2x2'],

      ['/tmp/MR10032/CBU101288_MR10032/20100929_114835/Series_002_CBU_MPRAGE',
       '/tmp/MR10032/CBU101288_MR10032/20100929_114835/Series_005_ep2d_advdiff_DSI_101_25x25x25_STEAM_b4000',
       '/tmp/MR10032/CBU101288_MR10032/20100929_114835/Series_006_ep2d_advdiff_DTI_25x25x25_STEAM_118dir_b1000',
       '/tmp/MR10032/CBU101288_MR10032/20100929_114835/Series_004_CBU_DTI_64InLea_2x2x2'],

      ['/tmp/MR10032/CBU101289_MR10032/20100929_124035/Series_002_CBU_MPRAGE',
       '/tmp/MR10032/CBU101289_MR10032/20100929_124035/Series_004_ep2d_advdiff_DSI_101_25x25x25_STEAM_b4000',
       '/tmp/MR10032/CBU101289_MR10032/20100929_124035/Series_005_ep2d_advdiff_DTI_25x25x25_STEAM_118dir_b1000',
       '/tmp/MR10032/CBU101289_MR10032/20100929_124035/Series_006_CBU_DTI_64InLea_2x2x2'],

      ['/tmp/MR10032/CBU101291_MR10032/20100929_151504/Series_002_CBU_MPRAGE',
       '/tmp/MR10032/CBU101291_MR10032/20100929_151504/Series_004_ep2d_advdiff_DSI_101_25x25x25_STEAM_b4000',
       '/tmp/MR10032/CBU101291_MR10032/20100929_151504/Series_005_ep2d_advdiff_DTI_25x25x25_STEAM_118dir_b1000',
       '/tmp/MR10032/CBU101291_MR10032/20100929_151504/Series_006_CBU_DTI_64InLea_2x2x2']]
      

def transform_2nii():

    for subj in dirs:
        for dname in subj:
                dcm2nii(dname,' ')

    

      

      

      

      

      


      

      
       
       
       
       
