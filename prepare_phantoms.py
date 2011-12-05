import numpy as np
from dipy.reconst.gqi import GeneralizedQSampling
from dipy.reconst.dti import Tensor
from dipy.io import pickles as pkl
from dipy.reconst.pdi import ProjectiveDiffusivity
from dipy.viz import fvtk
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from dipy.utils.spheremakers import sphere_vf_from
from dipy.data import get_sphere
from dipy.core.geometry import sphere2cart, cart2sphere
from scipy.optimize import fmin as fmin_simplex, fmin_powell, fmin_cg, leastsq
import nibabel as nib
from LSC_limits import vec2vecrotmat,rotation_vec2mat
from scikits.learn.decomposition import PCA,KernelPCA,fastica
from dipy.data import get_data as get_dataX
from dipy.core.triangle_subdivide import create_unit_sphere,create_half_unit_sphere
from itertools import combinations
from mlp import mlp

''' file  has one row for every voxel, every voxel is repeating 1000
times with the same noise level , then we have 100 different
directions. 1000 * 100 is the number of all rows.

The 100 conditions are given by 10 polar angles (in degrees) 0, 20, 40, 60, 80,
80, 60, 40, 20 and 0, and each of these with longitude angle 0, 40, 80,
120, 160, 200, 240, 280, 320, 360. 
'''

#new complete SimVoxels files
simdata = ['fibres_2_SNR_80_angle_90_l1_1.4_l2_0.35_l3_0.35_iso_0_diso_00',
 'fibres_2_SNR_60_angle_60_l1_1.4_l2_0.35_l3_0.35_iso_0_diso_00',
 'fibres_2_SNR_40_angle_30_l1_1.4_l2_0.35_l3_0.35_iso_0_diso_00',
 'fibres_2_SNR_40_angle_60_l1_1.4_l2_0.35_l3_0.35_iso_0_diso_00',
 'fibres_2_SNR_20_angle_15_l1_1.4_l2_0.35_l3_0.35_iso_1_diso_0.7',
 'fibres_2_SNR_100_angle_90_l1_1.4_l2_0.35_l3_0.35_iso_0_diso_00',
 'fibres_2_SNR_20_angle_30_l1_1.4_l2_0.35_l3_0.35_iso_1_diso_0.7',
 'fibres_2_SNR_40_angle_15_l1_1.4_l2_0.35_l3_0.35_iso_1_diso_0.7',
 'fibres_2_SNR_60_angle_15_l1_1.4_l2_0.35_l3_0.35_iso_1_diso_0.7',
 'fibres_2_SNR_100_angle_90_l1_1.4_l2_0.35_l3_0.35_iso_1_diso_0.7',
 'fibres_1_SNR_60_angle_00_l1_1.4_l2_0.35_l3_0.35_iso_1_diso_0.7',
 'fibres_2_SNR_80_angle_30_l1_1.4_l2_0.35_l3_0.35_iso_0_diso_00',
 'fibres_2_SNR_100_angle_15_l1_1.4_l2_0.35_l3_0.35_iso_0_diso_00',
 'fibres_2_SNR_100_angle_60_l1_1.4_l2_0.35_l3_0.35_iso_1_diso_0.7',
 'fibres_2_SNR_80_angle_60_l1_1.4_l2_0.35_l3_0.35_iso_0_diso_00',
 'fibres_2_SNR_60_angle_30_l1_1.4_l2_0.35_l3_0.35_iso_1_diso_0.7',
 'fibres_2_SNR_40_angle_60_l1_1.4_l2_0.35_l3_0.35_iso_1_diso_0.7',
 'fibres_2_SNR_80_angle_30_l1_1.4_l2_0.35_l3_0.35_iso_1_diso_0.7',
 'fibres_2_SNR_20_angle_30_l1_1.4_l2_0.35_l3_0.35_iso_0_diso_00',
 'fibres_2_SNR_60_angle_60_l1_1.4_l2_0.35_l3_0.35_iso_1_diso_0.7',
 'fibres_1_SNR_100_angle_00_l1_1.4_l2_0.35_l3_0.35_iso_1_diso_0.7',
 'fibres_1_SNR_100_angle_00_l1_1.4_l2_0.35_l3_0.35_iso_0_diso_00',
 'fibres_2_SNR_20_angle_15_l1_1.4_l2_0.35_l3_0.35_iso_0_diso_00',
 'fibres_1_SNR_20_angle_00_l1_1.4_l2_0.35_l3_0.35_iso_1_diso_0.7',
 'fibres_2_SNR_40_angle_15_l1_1.4_l2_0.35_l3_0.35_iso_0_diso_00',
 'fibres_2_SNR_20_angle_60_l1_1.4_l2_0.35_l3_0.35_iso_0_diso_00',
 'fibres_2_SNR_80_angle_15_l1_1.4_l2_0.35_l3_0.35_iso_1_diso_0.7',
 'fibres_1_SNR_80_angle_00_l1_1.4_l2_0.35_l3_0.35_iso_1_diso_0.7',
 'fibres_2_SNR_20_angle_90_l1_1.4_l2_0.35_l3_0.35_iso_1_diso_0.7',
 'fibres_2_SNR_60_angle_90_l1_1.4_l2_0.35_l3_0.35_iso_0_diso_00',
 'fibres_2_SNR_100_angle_30_l1_1.4_l2_0.35_l3_0.35_iso_0_diso_00',
 'fibres_2_SNR_80_angle_90_l1_1.4_l2_0.35_l3_0.35_iso_1_diso_0.7',
 'fibres_2_SNR_60_angle_15_l1_1.4_l2_0.35_l3_0.35_iso_0_diso_00',
 'fibres_2_SNR_20_angle_60_l1_1.4_l2_0.35_l3_0.35_iso_1_diso_0.7',
 'fibres_2_SNR_100_angle_15_l1_1.4_l2_0.35_l3_0.35_iso_1_diso_0.7',
 'fibres_1_SNR_20_angle_00_l1_1.4_l2_0.35_l3_0.35_iso_0_diso_00',
 'fibres_2_SNR_80_angle_60_l1_1.4_l2_0.35_l3_0.35_iso_1_diso_0.7',
 'fibres_1_SNR_80_angle_00_l1_1.4_l2_0.35_l3_0.35_iso_0_diso_00',
 'fibres_2_SNR_100_angle_30_l1_1.4_l2_0.35_l3_0.35_iso_1_diso_0.7',
 'fibres_1_SNR_40_angle_00_l1_1.4_l2_0.35_l3_0.35_iso_1_diso_0.7',
 'fibres_1_SNR_60_angle_00_l1_1.4_l2_0.35_l3_0.35_iso_0_diso_00',
 'fibres_2_SNR_40_angle_30_l1_1.4_l2_0.35_l3_0.35_iso_1_diso_0.7',
 'fibres_2_SNR_60_angle_30_l1_1.4_l2_0.35_l3_0.35_iso_0_diso_00',
 'fibres_2_SNR_40_angle_90_l1_1.4_l2_0.35_l3_0.35_iso_0_diso_00',
 'fibres_2_SNR_60_angle_90_l1_1.4_l2_0.35_l3_0.35_iso_1_diso_0.7',
 'fibres_2_SNR_80_angle_15_l1_1.4_l2_0.35_l3_0.35_iso_0_diso_00',
 'fibres_1_SNR_40_angle_00_l1_1.4_l2_0.35_l3_0.35_iso_0_diso_00',
 'fibres_2_SNR_100_angle_60_l1_1.4_l2_0.35_l3_0.35_iso_0_diso_00',
 'fibres_2_SNR_40_angle_90_l1_1.4_l2_0.35_l3_0.35_iso_1_diso_0.7',
 'fibres_2_SNR_20_angle_90_l1_1.4_l2_0.35_l3_0.35_iso_0_diso_00']

simdir = '/home/eg309/Data/SimVoxels/'
#simdir = '/home/ian/Data/SimVoxels/'



def a_few_phantoms():
    
#fibres_1_SNR_100_angle_00_l1_1.4_l2_0.35_l3_0.35_iso_0_diso_00

    table={}
    
    #sd=['fibres_1_SNR_60_angle_00_l1_1.4_l2_0.35_l3_0.35_iso_0_diso_00']
    #sd=['fibres_1_SNR_20_angle_00_l1_1.4_l2_0.35_l3_0.35_iso_0_diso_00']
    
    sd=['fibres_2_SNR_100_angle_90_l1_1.4_l2_0.35_l3_0.35_iso_0_diso_00']
    
    #for simfile in simdata:
    for simfile in sd:
        
        data=np.loadtxt(simdir+simfile)
        sf=simfile.split('_')
        print sf        
        print sf[1],sf[3],sf[5],sf[7],sf[9],sf[11],sf[13],sf[15]
        
        b_vals_dirs=np.loadtxt(simdir+'Dir_and_bvals_DSI_marta.txt')
        bvals=b_vals_dirs[:,0]*1000
        gradients=b_vals_dirs[:,1:]
        
        data2=data[::1000,:]                
        
        table={'fibres':sf[1],'snr':sf[3],'angle':sf[5],'l1':sf[7],'l2':sf[9],\
               'l3':sf[11],'iso':sf[13],'diso':sf[15],\
               'data':data2,'bvals':bvals,'gradients':gradients}
        
        #print table
        print table['data'].shape        
        pkl.save_pickle('test0.pkl',table)
        break
    
        


def gq_tn_calc_save():

    for simfile in simdata:    
        dataname = simfile
        print dataname
        sim_data=np.loadtxt(simdir+dataname)
        marta_table_fname=simdir+'Dir_and_bvals_DSI_marta.txt'
        b_vals_dirs=np.loadtxt(marta_table_fname)
        bvals=b_vals_dirs[:,0]*1000
        gradients=b_vals_dirs[:,1:]
        gq = GeneralizedQSampling(sim_data,bvals,gradients)
        gqfile = '/tmp/gq'+dataname+'.pkl'
        pkl.save_pickle(gqfile,gq)
        tn = Tensor(sim_data,bvals,gradients)
        tnfile = '/tmp/tn'+dataname+'.pkl'
        pkl.save_pickle(tnfile,tn)

        ''' file  has one row for every voxel, every voxel is repeating 1000
        times with the same noise level , then we have 100 different
        directions. 100 * 1000 is the number of all rows.

        At the moment this module is hardwired to the use of the EDS362
        spherical mesh. I am assumung (needs testing) that directions 181 to 361
        are the antipodal partners of directions 0 to 180. So when counting the
        number of different vertices that occur as maximal directions we wll map
        the indices modulo 181.
        '''


def simulations_marta(): 

    #gq_tn_calc_save()
    #a_few_phantoms()
    
    #sd=['fibres_2_SNR_100_angle_60_l1_1.4_l2_0.35_l3_0.35_iso_0_diso_00']
    #sd=['fibres_2_SNR_60_angle_60_l1_1.4_l2_0.35_l3_0.35_iso_0_diso_00']
    #sd=['fibres_2_SNR_100_angle_60_l1_1.4_l2_0.35_l3_0.35_iso_1_diso_0.7']
    sd=['fibres_2_SNR_100_angle_90_l1_1.4_l2_0.35_l3_0.35_iso_0_diso_00']
    #sd=['fibres_1_SNR_100_angle_00_l1_1.4_l2_0.35_l3_0.35_iso_0_diso_00']
    #sd=['fibres_2_SNR_20_angle_90_l1_1.4_l2_0.35_l3_0.35_iso_0_diso_00']
    #for simfile in simdata:
    
    np.set_printoptions(2)
    
    dotpow=6
    width=6
    sincpow=2
    sampling_length=1.2
    
    print dotpow,width,sincpow
    print sampling_length
    
    verts,faces=get_sphere('symmetric362')
    
    for simfile in sd:
        
        data=np.loadtxt(simdir+simfile)
        sf=simfile.split('_')        
        b_vals_dirs=np.loadtxt(simdir+'Dir_and_bvals_DSI_marta.txt')
        bvals=b_vals_dirs[:,0]*1000
        gradients=b_vals_dirs[:,1:]
            
        data2=data[::1000,:]   
        
        table={'fibres':sf[1],'snr':sf[3],'angle':sf[5],'l1':sf[7],'l2':sf[9],\
               'l3':sf[11],'iso':sf[13],'diso':sf[15],\
               'data':data2,'bvals':bvals,'gradients':gradients}
        
        print table['data'].shape
        pdi=ProjectiveDiffusivity(table['data'],table['bvals'],table['gradients'],dotpow,width,sincpow)
        gqs=GeneralizedQSampling(table['data'],table['bvals'],table['gradients'],sampling_length)
        ten=Tensor(table['data'],table['bvals'],table['gradients'])
        r=fvtk.ren()
        
        for i in range(10):#range(len(sdi.xa())):
            
            print 'No:',i
            print 'simulation fibres ',table['fibres'], ' snr ',table['snr'],' angle ', table['angle']
            pdiind=pdi.ind()[i]
            gqsind=gqs.ind()[i]
            
            print 'indices',pdiind,gqsind,\
                np.rad2deg(np.arccos(np.dot(verts[pdiind[0]],verts[pdiind[1]]))),\
                np.rad2deg(np.arccos(np.dot(verts[gqsind[0]],verts[gqsind[1]])))
                
            #ten.ind()[i],
            print 'peaks', pdi.xa()[i]*10**3,gqs.qa()[i]
            
            pd=pdi.spherical_diffusivity(table['data'][i])#*10**3
            #print 'pd stat',pd.min(),pd.max(),pd.mean(),pd.std()
            
            #colors=fvtk.colors(sdf,'jet')
            sdfcol=np.interp(pd,[pd.mean()-4*pd.std(),pd.mean()+4*pd.std()],[0,1])
            colors=fvtk.colors(sdfcol,'jet',False)        
            fvtk.add(r,fvtk.point(5*pdi.odf_vertices+np.array([12*i,0,0]),colors,point_radius=.6,theta=10,phi=10))
            
            odf=gqs.odf(table['data'][i])
            colors=fvtk.colors(odf,'jet')
            fvtk.add(r,fvtk.point(5*gqs.odf_vertices+np.array([12*i,-12,0]),colors,point_radius=.6,theta=10,phi=10))
             
        fvtk.show(r)

def show_signals(data,bvals,gradients,sticks=None):
    
    s=data[1:]
    s0=data[0]    
    ls=np.log(s)-np.log(s0)
    ind=np.arange(1,data.shape[-1])
    ob=-1/bvals[1:]
    #lg=np.log(s[1:])-np.log(s0)
    d=ob*(np.log(s)-np.log(s0))    
    r=fvtk.ren()
    all=fvtk.crossing(s,ind,gradients,scale=1)
    #fvtk.add(r,fvtk.line(all,fvtk.coral))    
    #d=d-d.min()        
    #all3=fvtk.crossing(d,ind,gradients,scale=10**4)
    #fvtk.add(r,fvtk.line(all3,fvtk.red))    
    #d=d-d.min()        
    all2=fvtk.crossing(d,ind,gradients,scale=10**4)
    fvtk.add(r,fvtk.line(all2,fvtk.green))    
    #"""
    #d2=d*10**4
    #print d2.min(),d2.mean(),d2.max(),d2.std()    
    for a in all2:    
        fvtk.label(r,str(np.round(np.linalg.norm(a[0]),2)),pos=a[0],scale=(.2,.2,.2),color=(1,0,0))        
    if sticks!=None:
        for s in sticks:
            ln=np.zeros((2,3))
            ln[1]=s
            fvtk.add(r,fvtk.line(d.max()*10**4*ln,fvtk.blue)) 
    #data2=data.reshape(1,len(data))
    #pdi=ProjectiveDiffusivity(data2,bvals,gradients,dotpow=6,width=6,sincpow=2)
    #pd=pdi.spherical_diffusivity(data)
    #print pd
    #"""    
    fvtk.show(r)
    
def online_kurtosis(data):
    n = 0
    mean = 0
    M2 = 0
    M3 = 0
    M4 = 0
 
    for x in data:
        n1 = n
        n = n + 1
        delta = x - mean
        delta_n = delta / n
        delta_n2 = delta_n * delta_n
        term1 = delta * delta_n * n1
        mean = mean + delta_n
        M4 = M4 + term1 * delta_n2 * (n*n - 3*n + 3) + 6 * delta_n2 * M2 - 4 * delta_n * M3
        M3 = M3 + term1 * delta_n * (n - 2) - 3 * delta_n * M2
        M2 = M2 + term1
 
    kurtosis = (n*M4) / (M2*M2) - 3
    return kurtosis
    
def simulations_dipy(bvals,gradients,d=0.0015,S0=100,angles=[(0,0),(90,0)],fractions=[35,35],snr=20):
    
    fractions=[f/100. for f in fractions]    
    f0=1-np.sum(fractions)    
    S=np.zeros(len(gradients))        
    sticks=[ sphere2cart(1,np.deg2rad(pair[0]),np.deg2rad(pair[1]))  for pair in angles]
    sticks=np.array(sticks)    
    for (i,g) in enumerate(gradients[1:]):
        S[i+1]=f0*np.exp(-bvals[i+1]*d)+ np.sum([fractions[j]*np.exp(-bvals[i+1]*d*np.dot(s,g)**2) for (j,s) in enumerate(sticks)])
        S[i+1]=S0*S[i+1]    
    S[0]=S0
    
    if snr!=None:
        std=S0/snr
        S=S+np.random.randn(len(S))*std
    
    return S,sticks

def call(x0,Sreal,S0,bvals,gradients):
    
    d=x0[0]/10.**4
    if len(x0)==1:
        S=np.zeros(len(gradients))
        for (i,g) in enumerate(gradients[1:]):
            S[i+1]=S0*np.exp(-bvals[i+1]*d)
        S[0]=S0
        return np.var(np.log(S/Sreal))            
    if len(x0)==4:
        angles=[(x0[1],x0[2])]
        fractions=[x0[3]/100.]
    if len(x0)==7:
        angles=[(x0[1],x0[2]),(x0[3],x0[4])]
        fractions=[x0[5]/100.,x0[6]/100.]
    if len(x0)==10:
        angles=[(x0[1],x0[2]),(x0[3],x0[4]),(x0[5],x0[6])]
        fractions=[x0[7]/100.,x0[8]/100.,x0[9]/100.]    
    f0=1-np.sum(fractions)    
    S=np.zeros(len(gradients))        
    sticks=[ sphere2cart(1,np.deg2rad(pair[0]),np.deg2rad(pair[1]))  for pair in angles]
    sticks=np.array(sticks)    
    for (i,g) in enumerate(gradients[1:]):
        S[i+1]=f0*np.exp(-bvals[i+1]*d)+ np.sum([fractions[j]*np.exp(-bvals[i+1]*d*np.dot(s,g)**2) for (j,s) in enumerate(sticks)])
        S[i+1]=S0*S[i+1]    
    S[0]=S0
    return np.var(np.log(S/Sreal))

def callnonlin(x0,Sreal,S0,bvals,gradients):
    
    d=x0[0]/10.**4
    if len(x0)==1:
        S=np.zeros(len(gradients))
        for (i,g) in enumerate(gradients[1:]):
            S[i+1]=S0*np.exp(-bvals[i+1]*d)
        S[0]=S0
        return Sreal-S
    if len(x0)==4:
        angles=[(x0[1],x0[2])]
        fractions=[x0[3]/100.]
    if len(x0)==7:
        angles=[(x0[1],x0[2]),(x0[3],x0[4])]
        fractions=[x0[5]/100.,x0[6]/100.]
    if len(x0)==10:
        angles=[(x0[1],x0[2]),(x0[3],x0[4]),(x0[5],x0[6])]
        fractions=[x0[7]/100.,x0[8]/100.,x0[9]/100.]    
    f0=1-np.sum(fractions)    
    S=np.zeros(len(gradients))        
    sticks=[ sphere2cart(1,np.deg2rad(pair[0]),np.deg2rad(pair[1]))  for pair in angles]
    sticks=np.array(sticks)    
    for (i,g) in enumerate(gradients[1:]):
        S[i+1]=f0*np.exp(-bvals[i+1]*d)+ np.sum([fractions[j]*np.exp(-bvals[i+1]*d*np.dot(s,g)**2) for (j,s) in enumerate(sticks)])
        S[i+1]=S0*S[i+1]    
    S[0]=S0
    return Sreal-S





    
def avg_diffusivity(s,bvals):
    ob=-1/bvals[1:]       
    d=ob*(np.log(s[1:])-np.log(s[0]))
    if np.sum(np.isnan(d))>0:
        d=np.zeros(d.shape)
    if np.sum(np.isinf(d))>0:
        d=np.zeros(d.shape)    
    return np.mean(d),np.std(d)
    
    

parameters={'101_32':['/home/eg309/Data/PROC_MR10032/subj_03/101_32',
                    '1312211075232351192010121313490254679236085ep2dadvdiffDSI10125x25x25STs004a001','.bval','.bvec','.nii'],
            '118_32':['/home/eg309/Data/PROC_MR10032/subj_03/118_32',\
                      '131221107523235119201012131413348979887031ep2dadvdiffDTI25x25x25STEAMs011a001','.bval','.bvec','.nii'],
            '64_32':['/home/eg309/Data/PROC_MR10032/subj_03/64_32',\
                     '1312211075232351192010121314035338138564502CBUDTI64InLea2x2x2s005a001','.bval','.bvec','.nii'],
            '515_32':['/home/eg309/Data/tp2/NIFTI',\
                      'dsi_bvals.txt','dsi_bvects.txt','DSI.nii']}
       

def get_data(name='101_32',par=0):    
    
    if name=='515_32':
        bvals=np.loadtxt(parameters[name][0]+'/'+parameters[name][1])[::-1]
        bvecs=np.loadtxt(parameters[name][0]+'/'+parameters[name][2])[::-1]
        img=nib.load(parameters[name][0]+'/'+parameters[name][3])
        return img.get_data()[:,:,:,:],bvals,bvecs
       
    
    bvals=np.loadtxt(parameters[name][0]+'/'+parameters[name][1]+parameters[name][2])
    bvecs=np.loadtxt(parameters[name][0]+'/'+parameters[name][1]+parameters[name][3]).T
    img=nib.load(parameters[name][0]+'/'+parameters[name][1]+parameters[name][4])     
    
    if par>0:
        bvals=par*bvals    
    
    return img.get_data(),bvals,bvecs

#show_signals(table['data'][0],table['bvals'],table['gradients'])

def simulation_experiment():
    data,bvals,bvecs=get_data('101_32')    
    S,sticks=simulations_dipy(bvals,bvecs,angles=[(0,0)],fractions=[0])
    xopt=fmin_powell(call,[10],(S,100,bvals,bvecs))
    print np.round(xopt,2)    
    S,sticks=simulations_dipy(bvals,bvecs,angles=[(90,0)],fractions=[60])
    xopt=fmin_powell(call,[0,90,0,20],(S,100,bvals,bvecs))
    print np.round(xopt,2)    
    S,sticks=simulations_dipy(bvals,bvecs,angles=[(0,0),(90,0)],fractions=[50,50])
    xopt=fmin_powell(call,[20,10,10,90,0,20,30],(S,100,bvals,bvecs))
    print np.round(xopt,2)    
    S,sticks=simulations_dipy(bvals,bvecs,angles=[(0,0),(90,0),(90,90)],fractions=[33,33,33])
    xopt=fmin_powell(call,[20,10,0,90,0,45,50,20,30,20],(S,100,bvals,bvecs))
    print np.round(xopt,2)


def simulation_experiment_nonlin():
    data,bvals,bvecs=get_data('101_32')    
    #S,sticks=simulations_dipy(bvals,bvecs,angles=[(0,0),(90,0),(90,90)],fractions=[33,33,33])
    #S,sticks=simulations_dipy(bvals,bvecs,angles=[(90,0)],fractions=[60])
    S,sticks=simulations_dipy(bvals,bvecs,angles=[(10,10)],fractions=[100])    
    xopt,ier=leastsq(callnonlin,[5,90,0,20],(S,100,bvals,bvecs))
    print np.round(xopt,2)
    S,sticks=simulations_dipy(bvals,bvecs,angles=[(0,0),(90,0),(90,90)],fractions=[33,33,33])
    xopt,ier=leastsq(callnonlin,[5,90,0,20,20,50,70,20,20,20],(S,100,bvals,bvecs))
    print np.round(xopt,2)

def show_sph_grid():
    r=fvtk.ren()
    print verts.shape, faces.shape
    sph=np.abs(np.dot(gradients[1],verts.T)**2)
    print sph.shape
    cols=fvtk.colors(sph,'jet')
    #fvtk.add(r,fvtk.point(verts,cols,point_radius=.1,theta=10,phi=10))
    
    for (i,v) in enumerate(gradients):    
        fvtk.label(r,str(i),pos=v,scale=(.02,.02,.02))
        if i in [62,76,58]:
            fvtk.label(r,str(i),pos=v,scale=(.05,.05,.05),color=(1,0,0))
    fvtk.show(r)


def real_datanonlinsq():

    data,bvals,bvecs=get_data(name='64_32')
    S=data[:,:,42]
    #S=data[40:60,40:60,42]
    #S=data[:,:,27]
    N=np.zeros((data.shape[0],data.shape[1]))
    M=np.zeros((data.shape[0],data.shape[1]))
    X0=np.zeros((data.shape[0],data.shape[1]))
    X1=np.zeros((data.shape[0],data.shape[1]))
    X2=np.zeros((data.shape[0],data.shape[1]))
    X3=np.zeros((data.shape[0],data.shape[1]))
    for (i,j) in np.ndindex(S.shape[0],S.shape[1]):
        s=S[i,j]
        ad,std=avg_diffusivity(s,bvals)
        N[i,j]=ad
        M[i,j]=std
        print i,j
        #xopt0,ier=leastsq(callnonlin,[15],(s,s[0],bvals,bvecs))
        #X0[i,j]=xopt0#[0]
        xopt1,ier=leastsq(callnonlin,[15,90,0,20],(s,s[0],bvals,bvecs))
        X1[i,j]=xopt1[0]
        #xopt2,ier=leastsq(callnonlin,[15,90,0,20,20,50,70],(s,s[0],bvals,bvecs))
        #X2[i+40,j+40]=xopt2[0]
        #xopt3,ier=leastsq(callnonlin,[5,90,0,20,20,50,70,20,20,20],(s,s[0],bvals,bvecs))
        #X3[i+40,j+40]=xopt3[0]
        #xopt=fmin_powell(call,[10],(s,s[0],bvals,bvecs))    
        #print i,j,xopt
        #N[i+40,j+40]=xopt
    

def playing_with_diffusivities_and_simulations():

    data,bvals,bvecs=get_data(name='101_32')
    #S=data[:,:,42]
    #drange=np.linspace(0,40,5)
    
    S1,sticks1=simulations_dipy(bvals,bvecs,d=0.0015,S0=200,angles=[(0,0),(90,0)],fractions=[50,50],snr=None)
    S2,sticks2=simulations_dipy(bvals,bvecs,d=0.0015,S0=200,angles=[(0,0),(90,0),(90,90)],fractions=[33,33,33],snr=None)
    
    scale=10**4
    
    ob=-1/bvals[1:]       
    D1=ob*(np.log(S1[1:])-np.log(S1[0]))*scale
    D2=ob*(np.log(S2[1:])-np.log(S2[0]))*scale
    
    Bvecs=np.concatenate([bvecs[1:],-bvecs[1:]])
    SS1=np.concatenate([S1[1:],S1[1:]])
    SS2=np.concatenate([S2[1:],S2[1:]])
    
    DD1=np.concatenate([D1,D1])
    DD2=np.concatenate([D2,D2])
    
    X1=np.dot(np.diag(DD1),Bvecs)
    X2=np.dot(np.diag(DD2),Bvecs)
    
    U,s,V=np.linalg.svd(np.dot(X1.T,X2),full_matrices=True)
    R=np.dot(U,V.T)
    
    print np.round(R,2)
    print np.sum(np.dot(X1,R)-X2,axis=0)
    
    r=fvtk.ren()
    fvtk.add(r,fvtk.point(X1,fvtk.red,1,.5,16,16))
    fvtk.add(r,fvtk.point(X2,fvtk.green,1,.5,16,16))
    fvtk.add(r,fvtk.axes((10,10,10)))
    fvtk.show(r)
    
    U1,s1,V1=np.linalg.svd(np.dot(X1.T,X1),full_matrices=True)
    U2,s2,V2=np.linalg.svd(np.dot(X2.T,X2),full_matrices=True)
    
    u1=U1[:,2]
    u2=U2[:,2]
    
    R2=vec2vecrotmat(u1,u2)
    
    print np.round(R2,2)
    print np.rad2deg(np.arccos(np.dot(u1,u2)))
    
    uu1=np.zeros((2,3))
    uu1[1]=u1
    uu2=np.zeros((2,3))
    uu2[1]=u2
    
    kX1=online_kurtosis(X1)
    kX2=online_kurtosis(X2)
    
    print 's1',s1
    print 's2',s2
    print 'kX1',kX1
    print 'kX2',kX2
    print 'average diffusivity 1',np.mean(np.linalg.norm(X1))
    print 'average diffusivity 2',np.mean(np.linalg.norm(X2))


def diffusivitize(S,bvals,bvecs,scale=10**4,invert=True):
    ob=-1/bvals[1:]       
    D=ob*(np.log(S[1:])-np.log(S[0]))*scale        
    DD=np.concatenate([D,D])    
    Bvecs=np.concatenate([bvecs[1:],-bvecs[1:]])
    if invert==True:
        X=np.dot(np.diag(DD.max()-DD),Bvecs)
    else:
        X=np.dot(np.diag(DD),Bvecs)
    return X
    
def show_many_fractions(name):

    data,bvals,bvecs=get_data(name)
    
    S0s=[];S1s=[];S2s=[];
    X0s=[];X1s=[];X2s=[];
    U0s=[];U1s=[];U2s=[];
    
    Fractions=[0,10,20,30,40,50,60,70,80,90,100]
    #Fractions=[90,100]
    for f in Fractions:
        S0,sticks=simulations_dipy(bvals,bvecs,d=0.0015,S0=200,angles=[(0,0)],fractions=[f],snr=None)
        X0=diffusivitize(S0,bvals,bvecs);S0s.append(S0);X0s.append(X0)
        
        S1,sticks=simulations_dipy(bvals,bvecs,d=0.0015,S0=200,angles=[(0,0),(90,0)],fractions=[f/2.,f/2.],snr=None)
        X1=diffusivitize(S1,bvals,bvecs);S1s.append(S1);X1s.append(X1)
        
        S2,sticks=simulations_dipy(bvals,bvecs,d=0.0015,S0=200,angles=[(0,0),(90,0),(90,90)],fractions=[f/3.,f/3.,f/3.],snr=None)
        X2=diffusivitize(S2,bvals,bvecs);S2s.append(S2);X2s.append(X2)
        U,s,V=np.linalg.svd(np.dot(X2.T,X2),full_matrices=True)
        U2s.append(U)
        
    r=fvtk.ren()
    for i in range(len(Fractions)):
        fvtk.add(r,fvtk.point(X0s[i]+np.array([30*i,0,0]),fvtk.red,1,.5,8,8))
        fvtk.add(r,fvtk.point(X1s[i]+np.array([30*i,30,0]),fvtk.green,1,.5,8,8))
        fvtk.add(r,fvtk.point(X2s[i]+np.array([30*i,60,0]),fvtk.yellow,1,.5,8,8))
        
    fvtk.show(r)

def axes_from_diffusivities(X,type='ica'):
    if type=='pca':
        U,s,V=np.linalg.svd(np.dot(X.T,X),full_matrices=True)
    if type=='ica':
        K,U,SR=fastica(X,3)
        print U.shape
        s=[100,100,100]
        
    axes0=[np.sqrt(s[0])*np.array([U[:,0].T,-U[:,0]])]
    axes1=[np.sqrt(s[1])*np.array([U[:,1].T,-U[:,1]])]
    axes2=[np.sqrt(s[2])*np.array([U[:,2].T,-U[:,2]])]    
    return axes0,axes1,axes2

def test(x,y,z):
    
    fimg,fbvals,fbvecs=get_dataX('small_101D')
    img=nib.load(fimg)
    data=img.get_data()
    print('data.shape (%d,%d,%d,%d)' % data.shape)
    bvals=np.loadtxt(fbvals)
    bvecs=np.loadtxt(fbvecs).T
    S=data[x,y,z,:]
    print('S.shape (%d)' % S.shape)
    X=diffusivitize(S,bvals,bvecs,scale=10**4)
    r=fvtk.ren()
    fvtk.add(r,fvtk.point(X,fvtk.green,1,.5,16,16))
    fvtk.show(r)

def rotate_needles(needles=[(0,0),(90,0),(90,90)],angles=(90,0,90)):

    theta,phi,psi=angles
    
    rot_theta = np.deg2rad(theta)
    rot_phi = np.deg2rad(phi)
    rot_psi = np.deg2rad(psi)
    #ANGLES FOR ROTATION IN EULER 
    
    v = np.array([np.cos(rot_phi)*np.sin(rot_theta),np.sin(rot_phi)*np.sin(rot_theta),np.cos(rot_theta)])
    #print v
    k_cross = np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]])
    #print k_cross
    #rot = rotation_vec2mat(r)
    rot = np.eye(3)+np.sin(rot_psi)*k_cross+(1-np.cos(rot_psi))*np.dot(k_cross,k_cross)
    #print rot
    
    #needles=[(0,0),(90,0),(90,90)]
    #INITIAL NEEDLES in (theta, phi)
    
    needle_rads = []
    for n in needles:
        needle_rads+=[[1]+list(np.deg2rad(n))]
    #print needle_rads 
    #initial needles in (r, theta, phi)
    
    rad_needles=[]
    for n in needle_rads:
        rad_needles+=[cart2sphere(*tuple(np.dot(rot,sphere2cart(*n))))[1:3]]
    #print rad_needles
    #ROTATED NEEDLES IN SPHERICAL COORDS (RADIANS)
    
    deg_angles = []
    for n in rad_needles:
        a = []
        for c in n:
            a+=[np.rad2deg(c)]
        deg_angles += [a]
    #print deg_angles
    #ROTATED NEEDLES IN SPHERICAL COORDS (DEGREES)
    
    return deg_angles


    
def kernelmatrix(data,kernel,param=np.array([3,2])):    
    if kernel=='linear':
        return np.dot(data,data.T)
    elif kernel=='gaussian':
        K = np.zeros((data.shape[0],data.shape[0]))
        for i in range(data.shape[0]):
            for j in range(i+1,data.shape[0]):
                K[i,j] = np.sum((data[i,:]-data[j,:])**2)
                K[j,i] = K[i,j]
        return np.exp(-K/(2*param[0]**2))#K**2
    elif kernel=='polynomial':
        return (np.dot(data,data.T)+param[0])**param[1]
    elif kernel=='cityblock':
        K = np.zeros((data.shape[0],data.shape[0]))
        for i in range(data.shape[0]):
            for j in range(i+1,data.shape[0]):
                K[i,j] = np.sum(np.abs(data[i,:]-data[j,:]))
                K[j,i] = K[i,j]
        return np.exp(-K/(2*param[0]))
    
def kernelpca(data,kernel,redDim,param):    
    nData = data.shape[0]
    nDim = data.shape[1]    
    K = kernelmatrix(data,kernel,param)    
    # Compute the transformed data
    D = np.sum(K,axis=0)/nData
    E = np.sum(D)/nData
    J = np.ones((nData,1))*D
    K = K - J - J.T + E*np.ones((nData,nData))    
    # Perform the dimensionality reduction
    evals,evecs = np.linalg.eig(K) 
    indices = np.argsort(evals)
    indices = indices[::-1]
    evecs = evecs[:,indices[:redDim]]
    evals = evals[indices[:redDim]]    
    sqrtE = np.zeros((len(evals),len(evals)))
    #for i in range(len(evals)):
    #    sqrtE[i,i] = np.sqrt(evals[i])    
    sqrtE = np.diag(np.sqrt(evals))   
    #print shape(sqrtE), shape(data)
    newData = (np.dot(sqrtE,evecs.T)).T    
    return newData

def investigate_formula_pca():
    Fractions=[[0,0,0],[100,0,0],[50,0,0],[50,50,0],[75,25,0],[33,33,33],[20,20,20],[60,20,20]]    
    Needles=[[(0,0),(90,0),(90,90)]]#,[(0,0),(30,0),(90,90)],[(0,0),(60,0),(90,90)],[(0,0),(60,0),(60,60)]]    
    Angles=[(0,0,0),(20,90,30),(60,20,10)]    
    for needles in Needles:
        for fractions in Fractions:
            for angles in Angles:
            
                deg_angles=rotate_needles(needles=needles,angles=angles)
                S0,sticks0=simulations_dipy(bvals,bvecs,d=0.0015,S0=100,angles=deg_angles,fractions=fractions,snr=None)
                X0=diffusivitize(S0,bvals,bvecs,invert=True)         
                U,s,V=np.linalg.svd(np.dot(X0.T,X0),full_matrices=True)
                print '#',angles,needles,fractions
                print np.abs(np.dot(sticks0,U))
                print s/np.sum(s)
            raw_input('Enter ...')
            #print np.abs(np.dot(axes,sticks0.T))

def investigate_pca():
        
    data,bvals,bvecs=get_data(name='118_32')
    
    deg_angles=rotate_needles(needles=[(0,0),(90,0),(90,90)],angles=(0,0,0))
    S0,sticks0=simulations_dipy(bvals,bvecs,d=0.0015,S0=100,angles=deg_angles,fractions=[100,0,0],snr=None)
    X0=diffusivitize(S0,bvals,bvecs,invert=False)
    D0=np.sqrt(np.sum(X0**2,axis=1))
    print 'D0 shape',D0.shape
    
    
    deg_angles=rotate_needles(needles=[(0,0),(90,0),(90,90)],angles=(0,0,0))
    S1,sticks0=simulations_dipy(bvals,bvecs,d=0.0015,S0=100,angles=deg_angles,fractions=[0,100,0],snr=None)
    X1=diffusivitize(S1,bvals,bvecs,invert=False)
    D1=np.sqrt(np.sum(X1**2,axis=1))
    
    deg_angles=rotate_needles(needles=[(0,0),(90,0),(90,90)],angles=(0,0,0))
    S2,sticks0=simulations_dipy(bvals,bvecs,d=0.0015,S0=100,angles=deg_angles,fractions=[0,0,100],snr=None)
    X2=diffusivitize(S2,bvals,bvecs,invert=False)
    D2=np.sqrt(np.sum(X2**2,axis=1))
    
    
    
    deg_angles=rotate_needles(needles=[(0,0),(90,0),(90,90)],angles=(0,0,0))
    SM,sticks0=simulations_dipy(bvals,bvecs,d=0.0015,S0=100,angles=deg_angles,fractions=[33,33,33],snr=None)
    XM=diffusivitize(SM,bvals,bvecs,invert=False)
    DM=np.sqrt(np.sum(XM**2,axis=1))
    
    deg_angles=rotate_needles(needles=[(0,0),(90,0),(90,90)],angles=(0,0,0))
    SM2,sticks0=simulations_dipy(bvals,bvecs,d=0.0015,S0=100,angles=deg_angles,fractions=[50,50,0],snr=None)
    XM2=diffusivitize(SM2,bvals,bvecs,invert=False)
    DM2=np.sqrt(np.sum(XM2**2,axis=1))
    
    X012=(X0+X1+X2)/3.
    S012=(S0+S1+S2)/3.
    D012=(D0+D1+D2)/3.
    X01=(X0+X1)/2.
    
    print 'D', np.corrcoef(DM,D012)
    
    print 'X', np.corrcoef(np.sqrt(np.sum(XM**2,axis=1)), np.sqrt(np.sum(X012**2,axis=1)))
    
    r=fvtk.ren()
    fvtk.add(r,fvtk.point(X01,fvtk.yellow,1,.2,16,16))
    fvtk.add(r,fvtk.point(XM2,fvtk.red,1,.2,16,16))
    fvtk.show(r)


def codebook(bvals,bvecs,S0=100,d=0.0015,fractions=False):
            
    sticks,e,t=create_half_unit_sphere(3)    
    CB=[]
    STK=[]
    FRA=[]
    REG=[]
    
    def single_needles(CB,STK,FRA,bvals,bvecs,sticks,fraction):
        for s in sticks:    
            S=np.zeros(len(bvecs))
            for (i,g) in enumerate(bvecs[1:]):
                S[i+1]=fraction*np.exp(-bvals[i+1]*d*np.dot(s,g)**2)
                S[i+1]=S0*S[i+1] 
            S[0]=S0        
            CB.append(S)
            STK.append([s])
            FRA.append([1.])
        return              
        
    #initial codebook only with single needles
    single_needles(CB,STK,FRA,bvals,bvecs,sticks,fraction=1)
    IN=np.array(CB)
    REG.append(len(CB))
    
    #two fibers  50 50                  
    for c in combinations(range(len(sticks)),2):
        CB.append((IN[c[0]]+IN[c[1]])/2.)
        STK.append([sticks[c[0]],sticks[c[1]]])
        FRA.append([.5,.5])
    #REG.append(len(CB))
        
    if fractions:
        #two fibers 75 25
        for c in combinations(range(len(sticks)),2):
            CB.append((.75*IN[c[0]]+.25*IN[c[1]]))
            STK.append([sticks[c[0]],sticks[c[1]]])
            FRA.append([.75,.25])
        #REG.append(len(CB))        
        #two fibers 25 75
        for c in combinations(range(len(sticks)),2):
            CB.append((.25*IN[c[0]]+.75*IN[c[1]]))
            STK.append([sticks[c[0]],sticks[c[1]]])
            FRA.append([.25,.75])
        #REG.append(len(CB))
    REG.append(len(CB))
        
    #three fibers 33 33 33            
    for c in combinations(range(len(sticks)),3):
        CB.append((IN[c[0]]+IN[c[1]]+IN[c[2]])/3.)
        STK.append([sticks[c[0]],sticks[c[1]],sticks[c[2]]])
        FRA.append([1/3.,1/3.,1/3.])
    #REG.append(len(CB))

    if fractions:
        #three fibers 60 20 20            
        for c in combinations(range(len(sticks)),3):
            CB.append((0.6*IN[c[0]]+0.2*IN[c[1]]+0.2*IN[c[2]]))
            STK.append([sticks[c[0]],sticks[c[1]],sticks[c[2]]])
            FRA.append([.6,.2,.2])
        #REG.append(len(CB))
        #three fibers 20 60 20            
        for c in combinations(range(len(sticks)),3):
            CB.append((0.2*IN[c[0]]+0.6*IN[c[1]]+0.2*IN[c[2]]))
            STK.append([sticks[c[0]],sticks[c[1]],sticks[c[2]]])
            FRA.append([.2,.6,.2])
        #REG.append(len(CB))        
        #three fibers 20 20 60            
        for c in combinations(range(len(sticks)),3):
            CB.append((0.2*IN[c[0]]+0.2*IN[c[1]]+0.6*IN[c[2]]))
            STK.append([sticks[c[0]],sticks[c[1]],sticks[c[2]]])
            FRA.append([.2,.2,.6])
        #REG.append(len(CB))
    REG.append(len(CB))
    #isotropic
    CB.append(S0*np.ones(len(bvecs)))
    STK.append([])
    FRA.append([0.,0.,0.])
    REG.append(len(CB))
    return np.array(CB),STK,FRA,REG
    

def draw_needles(r,sticks0,sc=60,w=5,off=np.array([0,0,0])):
    if len(sticks0)==3:
        fvtk.add(r,fvtk.line(off+sc*np.array([-sticks0[0],sticks0[0]]),fvtk.red,linewidth=w))
        fvtk.add(r,fvtk.line(off+sc*np.array([-sticks0[1],sticks0[1]]),fvtk.green,linewidth=w))
        fvtk.add(r,fvtk.line(off+sc*np.array([-sticks0[2],sticks0[2]]),fvtk.blue,linewidth=w))
    if len(sticks0)==2:
        fvtk.add(r,fvtk.line(off+sc*np.array([-sticks0[0],sticks0[0]]),fvtk.red,linewidth=w))
        fvtk.add(r,fvtk.line(off+sc*np.array([-sticks0[1],sticks0[1]]),fvtk.green,linewidth=w))
    if len(sticks0)==1:
        fvtk.add(r,fvtk.line(off+sc*np.array([-sticks0[0],sticks0[0]]),fvtk.red,linewidth=w))
    if len([])==0:
        pass

def draw_pcaaxes(r,U,sc=120,w=5,off=np.array([0,0,0])):
    fvtk.add(r,fvtk.line(off+sc*np.array([-U[:,0],U[:,0]]),0.5*fvtk.red,linewidth=w))
    fvtk.add(r,fvtk.line(off+sc*np.array([-U[:,1],U[:,1]]),0.5*fvtk.green,linewidth=w))
    fvtk.add(r,fvtk.line(off+sc*np.array([-U[:,2],U[:,2]]),0.5*fvtk.blue,linewidth=w))
    
def more_tests_with_pca():
    data,bvals,bvecs=get_data(name='118_32')    
    Bvecs=np.concatenate([bvecs[1:],-bvecs[1:]])
    
    S0,sticks0=simulations_dipy(bvals,bvecs,d=0.0015,S0=100,angles=[(0,0),(90,0),(90,90)],fractions=[100,0,0],snr=None)
    X0=np.dot(np.diag(np.concatenate([S0[1:],S0[1:]])),Bvecs)
    
    S1,sticks1=simulations_dipy(bvals,bvecs,d=0.0015,S0=100,angles=[(0,0),(45,0),(90,90)],fractions=[0,100,0],snr=None)
    X1=np.dot(np.diag(np.concatenate([S1[1:],S1[1:]])),Bvecs)   
    
    SX,sticksX=simulations_dipy(bvals,bvecs,d=0.0015,S0=100,angles=[(0,0),(90,0),(90,90)],fractions=[50,50,0],snr=None)
    XX=np.dot(np.diag(np.concatenate([SX[1:],SX[1:]])),Bvecs)   
    
    U0,s0,V0=np.linalg.svd(np.dot(X0.T,X0),full_matrices=True)
    U1,s1,V1=np.linalg.svd(np.dot(X1.T,X1),full_matrices=True)
    UX,sX,VX=np.linalg.svd(np.dot(XX.T,XX),full_matrices=True)
    
    X01=(X0+X1)/2.
    U01,s01,V01=np.linalg.svd(np.dot(X01.T,X01),full_matrices=True)
    np.set_printoptions(2,suppress=True)
    
    print UX
    print U01
    
    #r=fvtk.ren()
    
    #draw_needles(r,sticks0)
    #draw_pcaaxes(r,U0)
    
    #draw_needles(r,sticks1)
    #draw_pcaaxes(r,U1)
    
    #fvtk.add(r,fvtk.point(X0,fvtk.yellow,1,2,16,16))
    #fvtk.add(r,fvtk.point(X1,fvtk.red,1,2,16,16))
    #fvtk.add(r,fvtk.point(XX,fvtk.green,1,2,16,16))
    #fvtk.add(r,fvtk.point(X01,fvtk.blue,1,2,16,16))
        
    #KERNEL 
    #X0K=kernelpca(X0,'cityblock',3,[1/2.])
    #X1K=kernelpca(X1,'cityblock',3,[1/2.])
    #UK0,sK0,VK0=np.linalg.svd(np.dot(X0K.T,X0K),full_matrices=True)
    #UK1,sK1,VK1=np.linalg.svd(np.dot(X1K.T,X1K),full_matrices=True)
    #fvtk.add(r,fvtk.point(20*X0K,fvtk.yellow,1,2,16,16))
    #fvtk.add(r,fvtk.point(20*X1K,fvtk.red,1,2,16,16))
    #draw_pcaaxes(r,UK0)
    #draw_pcaaxes(r,UK1)
    
    #fvtk.show(r)

def vectorize_codebook(STK,FRA,REG):
    RES=np.zeros((len(STK),13))
    for (i,s) in enumerate(STK):
        if len(s)==1:
            RES[i,:3]=s[0]
            RES[i,10]=FRA[i][0]
        if len(s)==2:
            RES[i,:3]=s[0]
            RES[i,3:6]=s[1]
            RES[i,10]=FRA[i][0]
            RES[i,11]=FRA[i][1]            
        if len(s)==3:
            RES[i,:3]=s[0]
            RES[i,3:6]=s[1]
            RES[i,6:9]=s[1]
            RES[i,10]=FRA[i][0]
            RES[i,11]=FRA[i][1]
            RES[i,12]=FRA[i][2]
    return RES

def learn_nonlinear_mapping(hid_layers=4):    
    hid_layers=3
    train=CBK[0::2,:]/100.
    SFR=vectorize_codebook(STK,FRA,REG)
    target=SFR[0::2,-3:]    
    print train.shape, target.shape
    net = mlp(train,target,hid_layers,outtype='linear')
    valid=CBK[1::4,:]/100.
    valtarg=SFR[1::4,-3:]
    print valid.shape, valtarg.shape
    #net.mlptrain(train,target,0.9,100)
    net.earlystopping(train,target,valid,valtarg,0.9)#eta was 0.25
    test=CBK[3::4,:]/100.
    testtarg=SFR[3::4,-3:]
    print test.shape, testtarg.shape
    exptest = np.concatenate((test,-np.ones((np.shape(test)[0],1))),axis=1)
    testout=net.mlpfwd(exptest)    
    print testtarg[:3,:]
    print testout[:3,:]


def match_codebook(S0,CBK,REG,type=0): 
    R2s=[]
    print 'type >>>>>>>>>>>>',type, '<<<<<<<<<<<<<'
    if type==0:                
        print 'max', 1- np.abs(np.var(np.log(S0[1:])-np.log(CBK[-1,1:]))), 'index', REG[-1]-1, 'no', 0
        R2s.append(REG[-1])          
    if type==1:
        block=range(REG[0])
    if type==2:
        block=range(REG[0],REG[1])
    if type==3:
        block=range(REG[1],REG[2]) 
    if type>0:       
        for i in block:
            SM=CBK[i]
            #XM=np.dot(np.diag(np.concatenate([SM[1:],SM[1:]])),Bvecs)  
            #R2=(np.trace(np.dot(X0,XM.T))**2)/(np.trace(np.dot(X0,X0.T))*np.trace(np.dot(XM,XM.T)))
            R2=np.sum(S0*SM)**2/(np.sum(S0**2)*np.sum(SM**2))    
            #=np.corrcoef(S0,SM)**2
            R2s.append(R2)
        R2s=np.array(R2s) 
        print 'max ',R2s.max(),' index ',R2s.argmax(),' no ',len(STK[R2s.argmax()])
        print 'sim needles', sticks0
        #print 'needles ', STK[R1s.argmax()]
        print 'fractions ',FRA[R2.argmax()]
    else:
        R2s=np.array(R2s)
        print 'fractions',[]                 
    S0M=CBK[R2s.argmax()]
    ##Linear Regression Part
    c=np.cov(np.vstack((S0M[1:],S0[1:])))
    scale=c[0,1]/c[0,0]
    isotropic=np.mean(S0[1:])-scale*np.mean(S0M[1:])
    S0MF=scale*S0M+isotropic
    print 'scale',scale,'isotropic',isotropic
    R2n = c[0,1]**2/(c[0,0]*c[1,1])
    r = np.corrcoef(S0,S0M)[0,1]
    #print R2n, r**2
    return S0MF


def codebook_invest():
        
    data,bvals,bvecs=get_data(name='118_32',par=0)
    CBK,STK,FRA,REG=codebook(bvals,bvecs,fractions=True)
    Bvecs=np.concatenate([bvecs[1:],-bvecs[1:]])
    S0,sticks0=simulations_dipy(bvals,bvecs,d=0.0015,S0=100,angles=[(0,0),(90,0),(90,90)],fractions=[0,100,0],snr=None)
    #S0=S0*bvals/1000
    #S0=S0[0]-S0
    #S0=100*S0/S0[0]
    
    SIN=CBK[:33]
    SIN_STK=STK[:33]
    #SIN=S0-SIN
    
    ASIN=np.sum(SIN,axis=1)
    
    print '#####################'
    print ASIN.min(), ASIN.argmin(), SIN_STK[ASIN.argmin()]
    print ASIN.max(), ASIN.argmax(), SIN_STK[ASIN.argmax()]
    print '#####################'
    
    #S0=np.abs(S0-SIN[ASIN.argmax()])
    
    X0=np.dot(np.diag(np.concatenate([S0[1:],S0[1:]])),Bvecs)
    
    #P0=np.fft.ifftn(X0)
    #P0=P0/np.sqrt(np.sum(P0[:,0]**2+P0[:,1]**2+P0[:,2]**2,axis=1))
    
    #S0=CBK[2000]
    #S0=S0[0]-S0
    #X0=np.dot(np.diag(np.concatenate([S0[1:],S0[1:]])),Bvecs)
    S0MF0=match_codebook(S0,CBK,REG,type=0)
    S0MF1=match_codebook(S0,CBK,REG,type=1)
    S0MF2=match_codebook(S0,CBK,REG,type=2)
    S0MF3=match_codebook(S0,CBK,REG,type=3)
    
    #S0M=S0M[0]-S0M
    
    
    X0M=np.dot(np.diag(np.concatenate([S0MF0[1:],S0MF0[1:]])),Bvecs)
    r=fvtk.ren()
    fvtk.add(r,fvtk.point(X0,fvtk.yellow,1,2,16,16))
    draw_needles(r,sticks0,100,2)
    
    
    
    l=[]
    for i in range(len(SIN)):
        S=S0-SIN[i]
        print '>>>>'
        print SIN_STK[i]
        print i, np.sum(S[1:]),np.mean(S[1:]),np.var(S[1:])
        l.append(np.var(S[1:]))
        
        """
        X=np.dot(np.diag(np.concatenate([S[1:],S[1:]])),Bvecs)
        fvtk.add(r,fvtk.point(X+(i+1-14)*np.array([200,0,0]),fvtk.green,1,2,8,8))
        XSIN=np.dot(np.diag(np.concatenate([SIN[i][1:],SIN[i][1:]])),Bvecs)
        fvtk.add(r,fvtk.point(XSIN+np.array([(i+1-14)*200,-200,0]),fvtk.yellow,1,2,8,8))
        
        draw_needles(r,sticks0,100,2,off=(i+1-14)*np.array([200,0,0]))
        """
    
    print np.array(l).argmin()
    print '#=<>'
    print np.argsort(l)
    
    #fvtk.add(r,fvtk.point(0.01*P0,fvtk.green,1,2,16,16))
    #fvtk.add(r,fvtk.point(100*P0,fvtk.green,1,2,16,16))
    #fvtk.add(r,fvtk.point(X0M,fvtk.cyan,1,2,16,16))
    fvtk.show(r)
    
    """
    i=4000
    r=fvtk.ren()
    for (i,n) in enumerate([0, 20, 60, 100, 1000, 2000, 6017]):  
        SM=CBK[n]
        SM=SM[0]-SM
        XM=np.dot(np.diag(np.concatenate([SM[1:],SM[1:]])),Bvecs)    
        #fvtk.add(r,fvtk.point(20*X0,fvtk.yellow,1,20,16,16))
        fvtk.add(r,fvtk.point(np.array([i*4000,0,0])+20*XM,fvtk.red,1,20,16,16))
        draw_needles(r,STK[n],1000,5,np.array([i*4000,0,0]))      
        R2=(np.trace(np.dot(X0,XM.T))**2)/(np.trace(np.dot(X0,X0.T))*np.trace(np.dot(XM,XM.T)))
        print R2   
    fvtk.show(r)
    """
    
if __name__=='__main__':
    
    data,bvals,bvecs=get_data('515_32')
    
    D=[0,-1,1,-2,2,-3,3,-4,4,-5,5,-6,6]
    
    cnt=0
    for a in D:
        for b in D:
            for c in D:
                if a**2+b**2+c**2<=36:
                    cnt+=1
                    
    print cnt

    for (i,g) in enumerate(bvecs[2:]):
        if np.sum((bvecs[1]+g)**2)<0.0001:
            print 2+i,g

    bvecs[0]=np.array([0,0,0])
    qtable=np.vstack((bvals,bvals,bvals)).T * bvecs
    
    r=fvtk.ren()
    fvtk.add(r,fvtk.point(qtable/qtable.max(),fvtk.red,1,0.01,6,6))
    fvtk.show(r)
