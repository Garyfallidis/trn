import numpy as np
import itertools
from dipy.core import performance as pf
from dipy.core import track_learning as tl
from dipy.core import track_metrics as tm
from dipy.viz import fos
import pbc
import cPickle

import cProfile as profile
import pstats

path='/home/eg01/Data/PBC/pbc2009icdm'
G,hdr,R=pbc.load_training_set(path)

tracks=pbc.load_approximate_tracks(path,1,1)

#tracks=[t for (i,t) in enumerate(tracks) if i%25==0]

def test(bundle_list, divergence_threshold_list=[0.25], fibre_weight_list=[0.8],index_lists=False):
    
    comments = open('/home/ian/tractarian/commentary.txt','w')
    
    #reduced_hits = []

    #for b in [1,2,3,4,5,6,7,8]:
    for b in bundle_list:
            
        #print 'Starting ...'   
        refindex = G[b]['indices'].index([R[b]])
        ref = G[b]['tracks'][refindex]
        
        #print 'Bundle %d (%s)' % (b,G[b]['label_name'])
        print >> comments, 'Bundle %d (%s)' % (b,G[b]['label_name'])
        
        #print 'Removing far tracks ...'
        nearbundle,nearbundleindices = tl.rm_far_tracks(ref,tracks)

        #print 'Entering cut planes ...'
        hitdata = pf.cut_plane(nearbundle,ref)
        
        for divergence_threshold in divergence_threshold_list:
            for fibre_weight in fibre_weight_list:

                #print 'Reducing hit data ...'
                reduced_hitdata,heavy_weight_fibres = \
                    tl.threshold_hitdata(hitdata,divergence_threshold=divergence_threshold,fibre_weight=fibre_weight)

                #print 'Starting ...'
                #pbc.show_cut_color(reduced_hitdata,ref,bundle=G[b]['tracks'])

                #reduced_hits += [reduced_hitdata]
                
                green = set(G[b]['indices'])
                blue = set([nearbundleindices[i] for i in heavy_weight_fibres])
                
                nGB = len(green.intersection(blue))
                nG  = len(green.difference(blue))
                nB  = len(blue.difference(green))
                #print 'DivThresh %f; FibWt %f' % (divergence_threshold, fibre_weight)
                print >> comments, 'DivThresh %f; FibWt %f; Green %d; Blue %d; Green and blue %d; missed Green %d; stray Blue %d' % (divergence_threshold, fibre_weight, len(green),len(blue),nGB,nG,nB)
                #print >> comments, 'DivThresh %f; FibWt %f' % (divergence_threshold, fibre_weight)
                #print >> comments, 'Green %d; Blue %d; Green and blue %d; missed Green %d; stray Blue %d' \
                #    % (len(green),len(blue),nGB,nG,nB)
                if index_list == True:
                    return green, blue
    comments.close()
'''
if __name__=="__main__":
    
    profile.run('test()','teststat')'''
    
    