import numpy as np
from dipy.core import track_metrics as tm
import dipy.core.performance as pf
import numpy.linalg as npla
import pbc

'''
path='/home/eg01/Data/PBC/pbc2009icdm'

G,hdr,R,relR,ref = pbc.load_training_set_plus(path)

hits = pbc.load_pickle(path+'/hits.pkl')

b=8

plane_hits=hits[b][0]
'''

def concs(plane_hits,ref):
    '''
    calculates the log detrminant of the concentration matrix for the hits in planehits    
    '''
    dispersions = [np.prod(np.sort(npla.eigvals(np.cov(p[:,0:3].T)))[1:2]) for p in plane_hits]
    index = np.argmin(dispersions)
    log_concentration = -np.log2(dispersions[index])
    centre = ref[index+1]
    return index, centre, log_concentration

def refconc(brain, ref):
    '''
    given a reference fibre locates the parallel fibres in brain (tracks)
    with threshold_hitdata applied to cut_planes output then follows
    with concentration to locate the locus of a neck
    '''
    
    hitdata = pf.cut_plane(brain, ref)
    reduced_hitdata, heavy_weight_fibres = tl.threshold_hitdata(hitdata, divergence_threshold=0.2, fibre_weight=0.9)
    index, centre, conc = concs(reduced_hitdata, ref)
    return heavy_weight_fibres, index, centre
    

