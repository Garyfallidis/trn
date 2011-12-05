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

reference_indices = [
    (0,0,0),
    (1, 197816, 2223),
    (2, 15009, 47),
    ( 3, 157189, 6888),
    (4, 64423, 464),
    (5, 118191, 516),
    (6, 168055, 225),
    (7, 123041, 432),
    (8, 88647, 116)]


tracks=pbc.load_approximate_tracks(path,1,1)

#tracks=[t for (i,t) in enumerate(tracks) if i%25==0]

def test(bundle, divergence_threshold=0.25, fibre_weight=0.8, index_lists=False):
    
    comments = open('/home/ian/tractarian/commentary.txt','a')
    
    b = bundle

    #print 'Starting ...'   
    refindex = G[b]['indices'].index([R[b]])
    ref = G[b]['tracks'][refindex]
    
    #print 'Bundle %d (%s)' % (b,G[b]['label_name'])
    print >> comments, 'Bundle %d (%s)' % (b,G[b]['label_name'])
    
    #print 'Removing far tracks ...'
    nearbundle,nearbundleindices = tl.rm_far_tracks(ref,tracks)

    #print 'Entering cut planes ...'
    hitdata = pf.cut_plane(nearbundle,ref)
    

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

    if index_lists == True:
        return blue,green,reduced_hitdata
    comments.close()
'''
if __name__=="__main__":
    
    profile.run('test()','teststat')'''
    
    