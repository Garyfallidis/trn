
import pbc
import numpy as np
import pbc1109
from pbc1109 import track_volumes as tv
from dipy.viz import fos
from dipy.core import track_metrics as tm
from dipy.core import track_learning as tl
from dipy.core import performance as pf
from itertools import combinations

#---------------------------------------------------------------------------------------------------------------------------------------------

path='/home/eg01/Data/PBC/pbc2009icdm'

#---------------------------------------------------------------------------------------------------------------------------------------------
ids=tl.emi_atlas()

#---------------------------------------------------------------------------------------------------------------------------------------------
def load_template_tes_and_tracks(path,brain,scan):
    
    volpath=path+'/ICBM_WMPM_tweaked_'+str(brain) +'_'+str(scan)+'.nii'    
    print volpath    
    template,voxsz,aff=pbc.loadvol(volpath)    
    tracks=pbc.load_approximate_tracks(path,brain,scan)
    print 'template shape', template.shape    
    tcs,tes = tv.track_counts(tracks, template.shape, vox_sizes=(1,1,1), return_elements=True)
    print 'tcs shape', tcs.shape    
    return template,tcs,tes,tracks

#---------------------------------------------------------------------------------------------------------------------------------------------
def for_a_value_in_template(template,value,tes,tracks):
    ''' For a specific value in the ICBM atlas return the tracks that cross the region that has the same value. 
    '''

    ind=np.where(template==value)
    indices=set([])

    for i in range(len(ind[0])):
        try:
            tmp=tes[(ind[0][i], ind[1][i], ind[2][i])]
            indices=indices.union(set(tmp))
        except:
            pass
    
    bundle=[tracks[i] for i in list(indices)]        
    return bundle,list(indices)

#---------------------------------------------------------------------------------------------------------------------------------------------
def show_specific_bundles(r,template,ids,tes,tracks):
    
    for i in ids:
        vs=ids[i]['value']
        color=ids[i]['color']
        for v in vs:    
            
            bundle,indices=for_a_value_in_template(template,v,tes,tracks)            
            fos.add(r,fos.line(bundle,color,opacity=0.9))            

#--------------------------------------------------------------------------------------------------------------------------------------------
#9to20
''''
References=[[109309],[144023,235741,69257,219905],[56022],[154373,25104,31959],[197404],[64917,66270],
[33418,115381,230360],[177742],[45579,179716,196524],[88291,89525],[114248],[202092]]
'''

#---------------------------------------------------------------------------------------------------------------------------------------------
def show():    

    #brains=[(1,1),(1,2),(2,1),(3,1),(3,2)]
    brains=[(1,1),(2,1),(3,1)]

    ids=tl.emi_atlas()    
    
    #print ids.keys()
    
    for (b,s) in brains:
        
        ids2 = pbc.load_pickle(path+'/Relabelling_8_sc1_'+str(b)+'_'+str(s)+'.pkl')
        #'/Relabelling_8_sc1_'+str(b)+'_'+str(s)+'.pkl'
        print b,s, ids2.keys()
        tracks = pbc.load_approximate_tracks(path,b,s)
        
        for i in ids:
            
            if i >0:
                
                r=fos.ren()
                
                color=np.array(ids[i]['color'])                
                indices=ids2[i]['indices']                
                
                bundle=[tracks[ind] for ind in indices]      
                fos.add(r,fos.line(bundle,color,opacity=0.9))      
                
                print 'Bundle_name',i,ids[i]['bundle_name']            
                
                fos.show(r,title=ids[i]['bundle_name'][0])
            

#---------------------------------------------------------------------------------------------------------------------------------------------
def print_results():
    
    #brains=[(1,1),(1,2),(2,1),(3,1),(3,2)]
    brains=[(1,1),(2,1),(3,1)]

    for (b,s) in brains:
        
        fname=path+'/eg309_brain'+str(b)+'_scan'+str(s)+'_ch1.txt'
        f=open(fname,'wt')        
        subm=np.zeros((250000,2)).astype(int)    
        ids = pbc.load_pickle(path+'/Relabelling_'+str(b)+'_'+str(s)+'.pkl')
        
        for i in ids:            
            indices=ids[i]['indices']            
            for ind in indices:                
                subm[ind,0]=ind+1
                subm[ind,1]=i
            
        for i in range(subm.shape[0]):                        
            print >> f, '%d\t%d' % (i,subm[i,1]) 
                    
        f.close()
        

#---------------------------------------------------------------------------------------------------------------------------------------------
def make_ref_dic():
    '''
        Creates a dictionary containing reference fibres, names, and template values
        for the original 20 tracts in the training dataset
    '''
    path='/home/eg01/Data/PBC/pbc2009icdm'
    #name_list=['Unassigned', 'Arcuate L', 'Cingulum L','Corticospinal R', 'Forceps Major','Fornix','Inferior Occipitofrontal Fasciculus L','Superior Longitudinal Fasciculus L','Uncinate R']
    name_list=['Not Assigned']+[ids[i]['bundle_name'][0] for i in range(1,9)]
    #name_list=['Not Assigned']+[ids[i]['bundle_name'][0] for i in range(1,21)]
    #value_list=[0,41,35,9,5,6,31,41,48]
    #value_list= [[0]]+[ids[i]['value'] for i in range(1,21)]
    
    value_list= [[0]]+[ids[i]['value'] for i in range(1,9)]

    #corr = pbc.load_pickle(path+'/corr_20.pkl')
    corr = pbc.load_pickle(path+'/corr_8_sc1.pkl')
    
    #brainscan_keys = [(1,1),(1,2),(2,1),(3,1),(3,2)]
    brainscan_keys = [(1,1),(2,1),(3,1)]

    refdic={}
    
    #for (i, (b,s)) in enumerate([(1,1)]):
    for (i, (b,s)) in enumerate(brainscan_keys):
        
        refdic[(b,s)] = {}
        refdic[(b,s)]['reference_indices']=[[]]+[list(set(c)) for c in corr[i]]        
        refdic['names'] = name_list        
        refdic['template_values'] = value_list        
        template,tcs,tes,tracks = load_template_tes_and_tracks(path,b,s)
        
        #refdic[(b,s)]['reference_tracks'] =[[]]+ [[tracks[r] for r in l] for l in refdic[(b,s)]['reference_indices'][1:]]
        
        for (j,l) in enumerate(refdic[(b,s)]['reference_indices']):
            if j==0:
                refdic[(b,s)]['reference_tracks'] = [[]]
            else:
                refdic[(b,s)]['reference_tracks'].append( [tracks[t] for t in l])
                
        for (j,values) in enumerate(value_list):           
            if j==0:
                refdic[(b,s)]['template_tracks'] =[[]]
                refdic[(b,s)]['template_indices'] = [[]]
            else:
                all_template_tracks = []
                all_template_indices = []
                for value in values:
                    #template_tracks, template_indices = tracks_and_indices_for_a_value_in_template(template,value,tes,tracks)
                    template_tracks, template_indices = for_a_value_in_template(template,value,tes,tracks)
                    all_template_tracks += template_tracks
                    all_template_indices += template_indices
                refdic[(b,s)]['template_tracks'] += [all_template_tracks]
                refdic[(b,s)]['template_indices'] += [all_template_indices]
        
    pbc.save_pickle(path+'/refdic_8_sc1.pkl',refdic)
            
    #return refdic

#---------------------------------------------------------------------------------------------------------------------------------------------
def measure_internal_overlaps(refs):
    overlaps = np.zeros((len(refs),len(refs)), dtype=np.int)
 
    for b1 in range(2,len(refs)):
        ind1 = set(refs[b1])
        leftover = set(refs[b1])
        for b2 in range(1,b1):
            ind2 = set(refs[b2])
            overlaps[b1,b2]=len(ind1.intersection(ind2))
            overlaps[b2,b1]=overlaps[b1,b2]
            leftover=leftover.difference(ind2)
        overlaps[0,b1]=len(leftover)
        overlaps[b1,0]=overlaps[0,b1]
        
    return overlaps

#---------------------------------------------------------------------------------------------------------------------------------------------
def print_all_overlaps():
    brainscan_keys = [(1,1),(1,2),(2,1),(3,1),(3,2)]
    refdic = pbc.load_pickle('/home/eg01/Data/PBC/pbc2009icdm/refdic.pkl')

    for bs in brainscan_keys:
        refs=refdic[bs]['template_indices']
        print 'Overlaps for brainscan', bs
        print measure_internal_overlaps(refs)

#---------------------------------------------------------------------------------------------------------------------------------------------
def map_bundle_to_refs(bundle, bundle_indices, refs, ref_indices, refclass):
 
    min_ref_class = []
    for track in bundle:
        d = []
        for ref in refs:
            #d.append(pf.zhang_distances(track,ref,metric='min'))
            d.append(pf.zhang_distances(track,ref,metric='avg'))
        #print np.argmin(d), len(refclass), len(ref_indices), len(refs)
        min_dist=np.argmin(d)        
        if min_dist>15:
            min_ref_class.append(0)
        else:    
            min_ref_class.append(refclass[np.argmin(d)])

    assignment={}
    
    #for c in range(1,21):        
    for c in range(0,9):
    
        assignment[c] = []        
        
    for (i,bi) in enumerate(bundle_indices):            
        assignment[min_ref_class[i]].append(bi)
    
    return assignment
  
#---------------------------------------------------------------------------------------------------------------------------------------------
def relabel_them_all():
    
    path='/home/eg01/Data/PBC/pbc2009icdm'

    #refdic = pbc.load_pickle(path + '/refdic.pkl')
    refdic = pbc.load_pickle(path + '/refdic_8_sc1.pkl')
    
    #brains = [(1,1),(1,2),(2,1),(3,1),(3,2)]    
    brains = [(1,1),(2,1),(3,1)]    
    
    for (b,s) in brains:
        print b,s
        
        tA= refdic[(b,s)]['template_tracks']
        iA= refdic[(b,s)]['template_indices']
        
        tR = [] 
        iR = []         
        cR=[]
        
        #for bundle in range(1,21):
        for bundle in range(1,9):
            
            tmp=refdic[(b,s)]['reference_tracks'][bundle]             
            tR+=tmp            
            iR+=refdic[(b,s)]['reference_indices'][bundle]                 
            #print len(tmp),np.repeat(bundle,len(tmp)).astype(int)            
            cR+=list(np.repeat(bundle,len(tmp)).astype(int))       

        relabelling = {}

        #for bundle in range(1,21):
        #for bundle in range(1,9):
        for bundle in range(0,9):
            print 'Relabelling', bundle
            #print len(tR),len(iR),len(cR)
            relabelling[bundle] =  map_bundle_to_refs(tA[bundle], iA[bundle], tR, iR, cR)        
                
        IDS=ids                
        #for bundle1 in range(1,21):
        for bundle1 in range(1,9):
            print 'Remapping', 
            #for bundle2 in range(1,21):
            for bundle2 in range(1,9):
            
                try: 
                    IDS[bundle2]['indices']+=relabelling[bundle1][bundle2]                    
                except :
                    IDS[bundle2]['indices']=relabelling[bundle1][bundle2]
        
        print 'Saving',b,s            
        pbc.save_pickle(path+'/Relabelling_8_sc1_'+str(b)+'_'+str(s)+'.pkl',IDS)

#relabel_them_all()
#show()
    
    
    
    