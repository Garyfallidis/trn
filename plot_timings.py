
from time import time
import numpy as np
from dipy.tracking.distances import local_skeleton_clustering
from dipy.tracking.metrics import downsample,length,winding
from dipy.tracking.vox2track import track_counts

import matplotlib.pyplot as plt

from dipy.io.dpy import Dpy
from dipy.io.pickles import load_pickle,save_pickle

def change_dtype(T,type='f4'):
    return [t.astype(type) for t in T]

def plot_timings():
    
    #dir='/home/eg309/Data/LSC_limits/full_1M'
    dir='/tmp/full_1M'
    fs=['.npy','_2.npy','_3.npy','_4.npy','_5.npy']#,'_6.npy','_7.npy','_8.npy','_9.npy','_10.npy']
    
    T=[]
    for f in fs:
        fs1=dir+f
        T+=change_dtype(list(np.load(fs1)))
    #return T
    T=T[:100000]
    
    print len(T)    
    #dists=[4.,6.,8.,10.]
    dists=[8.]
    #pts=[3,6,12,18]
    pts=[12]
    #sub=10**5
    sub=10**3
    res={}
    for p in pts:
        print p
        res[p]={}
        for d in dists:
            print d
            res[p][d]={}
            res[p][d]['time']=[]
            res[p][d]['len']=[]
            step=0
            while step <= len(T):
                print step
                Td=[downsample(t,p) for t in T[0:step+sub]]
                t1=time()
                C=local_skeleton_clustering(Td,d)
                t2=time()
                res[p][d]['time'].append(t2-t1)
                res[p][d]['len'].append(len(C))       
                step=step+sub
    
    
    save_pickle('/tmp/res.pkl',res)
    print('Result saved in /tmp/res.pkl')
    #return res
    save_pickle('/tmp/res.pkl',res)
    
    
    
    
def show_timings():
    
    res=load_pickle('/home/eg309/Data/LSC_limits/timings_100K_1M.pkl')
    
    #[ 3,  6, 12, 18]
    dists=[  4.,   6.,   8.,  10.]
    
    plt.subplots_adjust(hspace=0.4)
    
    ax1=plt.subplot(221)    
    plt.title('3 point tracks')
    for d in dists:    
        ax1.plot(res[3][d]['time'][:-2],label=str(d*2.5)+' mm')        
    ax1.set_ylim((0,6000))
    ax1.set_xlim((0,9))    
    ax1.set_xticklabels(['100K','200K','300K','400K','500K','600K','700K','800K','900K','1M'])
    ax1.set_ylabel('Seconds')
    ax1.set_xlabel('Number of tracks')
    plt.legend()

    ax2=plt.subplot(222)
    plt.title('6 point tracks')
    for d in dists:    
        ax2.plot(res[6][d]['time'][:-2],label=str(d*2.5)+' mm')        
    ax2.set_ylim((0,6000))
    ax2.set_xlim((0,9))    
    ax2.set_xticklabels(['100K','200K','300K','400K','500K','600K','700K','800K','900K','1M'])
    ax2.set_ylabel('Seconds')
    ax2.set_xlabel('Number of tracks')
    plt.legend()
    
    ax3=plt.subplot(223)
    plt.title('12 point tracks')
    for d in dists:    
        ax3.plot(res[12][d]['time'][:-2],label=str(d*2.5)+' mm')        
    ax3.set_ylim((0,6000))
    ax3.set_xlim((0,9))    
    ax3.set_xticklabels(['100K','200K','300K','400K','500K','600K','700K','800K','900K','1M'])
    ax3.set_ylabel('Seconds')
    ax3.set_xlabel('Number of tracks')
    plt.legend()
    
    ax4=plt.subplot(224)
    plt.title('18 point tracks')
    for d in dists:    
        ax4.plot(res[18][d]['time'][:-2],label=str(d*2.5)+' mm')        
    ax4.set_ylim((0,6000))
    ax4.set_xlim((0,9))    
    ax4.set_xticklabels(['100K','200K','300K','400K','500K','600K','700K','800K','900K','1M'])
    ax4.set_ylabel('Seconds')
    ax4.set_xlabel('Number of tracks')
    plt.legend(loc=2)       
    
    plt.show()
    
def show_timing_vs_others():
    
    res=load_pickle('/home/eg309/Data/LSC_limits/timings_1K_100K.pkl')
    
    ax=plt.subplot(111)
    
    times=res[12][8]['time'][:-1]
    print len(times)
    x=np.arange(10**3,10**5+10**3,10**3)
    print len(x)
    
    #ax.set_xticklabels(['1K','20K','40K','60K','70K','80K','90K','100K'])    
    ax.plot(x,times,label='LSC')
    ax.plot([1000,60000],[30,14400],"*",label='Wang')
     
    #ax.set_ylim((0,40))
    #ax.set_xlim((1000,100000))
    
    plt.legend(loc=0)
    
    plt.show()

#show_timings()
show_timing_vs_others()
    
    
    
    
    
