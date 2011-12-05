
import pbc
import numpy as np
import pbc1109
from pbc1109 import track_volumes as tv
from dipy.viz import fos
from dipy.core import track_metrics as tm
from dipy.core import track_learning as tl
from dipy.core import performance as pf

#---------------------------------------------------------------------------------------------------------------------------------------------

path='/home/eg01/Data/PBC/pbc2009icdm'

#---------------------------------------------------------------------------------------------------------------------------------------------
ids={

1:{'name':['Arcuate L'],'value':[41],'color':fos.red},
2:{'name':['Cingulum L'],'value':[35],'color':fos.blue},
3:{'name':['Corticospinal R','Cerebral peduncle R'],'value':[9,17],'color':fos.yellow},
4:{'name':['Forceps Major'],'value':[5],'color':fos.green},
5:{'name':['Fornix'],'value':[6],'color':fos.indigo},
6:{'name':['Inferior Occipitofrontal Fasciculus (Sagittal stratum) L','Inferior Occipitofrontal Fasciculus L'],'value':[31,45],'color':fos.lime},
7:{'name':['Superior Longitudinal Fasciculus L'],'value':[41],'color':fos.gray},
8:{'name':['Uncinate R'],'value':[48],'color':fos.cyan},

9:{'name':['Cingulum R'],'value':[36],'color':fos.blue},
10:{'name':['Corticospinal L','Cerebral peduncle L'],'value':[8,16],'color':fos.yellow},
11:{'name':['Forceps Minor'],'value':[3],'color':fos.green},
12:{'name':['Corpus Callosum Body'],'value':[4],'color':fos.dark_red},
13:{'name':['Inferior Occipitofrontal Fasciculus (Sagittal stratum) R','Inferior Occipitofrontal Fasciculus R'],'value':[32,46],'color':fos.lime},
14:{'name':['Superior Longitudinal Fasciculus R'],'value':[42],'color':fos.gray},
15:{'name':['Uncinate L'],'value':[47],'color':fos.cyan},

16:{'name':['Middle cerebellar peduncle'],'value':[1],'color':fos.hot_pink},
17:{'name':['Medial lemniscus R'],'value':[11],'color':fos.aquamarine},
18:{'name':['Medial lemniscus L'],'value':[10],'color':fos.aquamarine},
19:{'name':['Tapatum R'],'value':[50],'color':fos.azure},
20:{'name':['Tapatum L'],'value':[49],'color':fos.azure}

#21:{'name':['Optic Radiation R'],'value':[30],'color':fos.coral},
#22:{'name':['Optic Radiation L'],'value':[29],'color':fos.coral}

}

    
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

    ind=np.where(template==value)
    indices=set([])

    for i in range(len(ind[0])):
        try:
            tmp=tes[(ind[0][i], ind[1][i], ind[2][i])]
            indices=indices.union(set(tmp))
        except:
            pass
            
    return [tracks[i] for i in indices],list(indices)

#---------------------------------------------------------------------------------------------------------------------------------------------
def tracks_and_indices_for_a_value_in_template(template,value,tes,tracks):

    ind=np.where(template==value)
    indices=set([])

    for i in range(len(ind[0])):
        try:
            tmp=tes[(ind[0][i], ind[1][i], ind[2][i])]
            indices=indices.union(set(tmp))
        except:
            pass
            
    return [tracks[i] for i in indices], indices

#---------------------------------------------------------------------------------------------------------------------------------------------
def show_specific_bundles(r,template,ids,tes,tracks):

    
    for i in ids:
        vs=ids[i]['value']
        color=ids[i]['color']
        for v in vs:    
            
            bundle,indices=for_a_value_in_template(template,v,tes,tracks)            
            fos.add(r,fos.line(bundle,color,opacity=0.9))            


#---------------------------------------------------------------------------------------------------------------------------------------------
def bundle_center_of_mass(bundle):
    
    cm=np.array([tm.center_of_mass(t) for t in bundle])            
    return np.mean(cm,axis=0)    

#---------------------------------------------------------------------------------------------------------------------------------------------    
def euclidean(p1,p2): 
    return np.sqrt(np.sum((p2-p1)**2))

#---------------------------------------------------------------------------------------------------------------------------------------------
def assign_intersecting_bundles(template,ids,tes,tracks):

    for i in ids:
        vs=ids[i]['value']                 
        ids[i]['indices']=[]
        ids[i]['tracks']=[]
              
        for v in vs:    
                        
            tracks,indices=for_a_value_in_template(template,v,tes,tracks)            
            ids[i]['indices']=ids[i]['indices']+indices
            ids[i]['tracks']=ids[i]['tracks']+indices
    
    no_ids=len(ids)
    
    #for pair in combinations(range(no_ids),2):
        
            
        
    
    return ids

#---------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------
def make_ref_dic():
    '''
        Creates a dictionary containing reference fibres, names, and template values
        for the original 20 tracts in the training dataset
    '''

    path='/home/eg01/Data/PBC/pbc2009icdm'
    #name_list=['Unassigned', 'Arcuate L', 'Cingulum L','Corticospinal R', 'Forceps Major','Fornix','Inferior Occipitofrontal Fasciculus L','Superior Longitudinal Fasciculus L','Uncinate R']
    name_list=['Not Assigned']+[ids[i]['name'][0] for i in range(1,21)]
    #value_list=[0,41,35,9,5,6,31,41,48]
    value_list= [[0]]+[ids[i]['value'] for i in range(1,21)]

    corr = pbc.load_pickle(path+'/corr_20.pkl')


    brainscan_keys = [(1,1),(1,2),(2,1),(3,1),(3,2)]

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
                    template_tracks, template_indices = tracks_and_indices_for_a_value_in_template(template,value,tes,tracks)
                    all_template_tracks += template_tracks
                    all_template_indices += template_indices
                refdic[(b,s)]['template_tracks'] += [all_template_tracks]
                refdic[(b,s)]['template_indices'] += [all_template_indices]
                
    return refdic

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
    refdic = pbc.load_pickle('/home/ian/tractarian/refdic.pkl')

    for bs in brainscan_keys:
        refs=refdic[bs]['template_indices']
        print 'Overlaps for brainscan', bs
        print measure_internal_overlaps(refs)
#---------------------------------------------------------------------------------------------------------------------------------------------
'''
r=fos.ren()

template,tcs,tes,tracks=load_template_tes_and_tracks(path,3,2)     
       
show_specific_bundles(r,template,ids,tes,tracks)

#bundle=[np.random.rand(10,3),np.random.rand(20,3)]
#bundle=[np.array([[1,0,0],[2,0,0],[3,0,0]]),np.array([[1,1,0],[2,1,0],[3,1,0]])]



fos.show(r)
'''
'''
ids11=pbc.load_pickle(path+'IDS_1_1.pkl')
refs11=[[]]+[ids11[i]['indices'] for i in ids11.keys()]
ids11da=pbc.load_pickle(path+'/IDS_DONT_ASSIGN_1_1.pkl')
refs11da=[[]]+[ids11da[i]['indices'] for i in ids11.keys()]

ids12=pbc.load_pickle(path+'IDS_1_1.pkl')
refs12=[[]]+[ids12[i]['indices'] for i in ids12.keys()]
ids12da=pbc.load_pickle(path+'/IDS_DONT_ASSIGN_1_1.pkl')
refs12da=[[]]+[ids12da[i]['indices'] for i in ids12.keys()]

ids21=pbc.load_pickle(path+'IDS_1_1.pkl')
refs21=[[]]+[ids21[i]['indices'] for i in ids21.keys()]
ids21da=pbc.load_pickle(path+'/IDS_DONT_ASSIGN_1_1.pkl')
refs21da=[[]]+[ids21da[i]['indices'] for i in ids21.keys()]

ids31=pbc.load_pickle(path+'IDS_1_1.pkl')
refs31=[[]]+[ids31[i]['indices'] for i in ids31.keys()]
ids31da=pbc.load_pickle(path+'/IDS_DONT_ASSIGN_1_1.pkl')
refs31da=[[]]+[ids31da[i]['indices'] for i in ids31.keys()]

ids31=pbc.load_pickle(path+'IDS_1_1.pkl')
refs31=[[]]+[ids31[i]['indices'] for i in ids31.keys()]
ids31da=pbc.load_pickle(path+'/IDS_DONT_ASSIGN_1_1.pkl')
refs31da=[[]]+[ids31da[i]['indices'] for i in ids31.keys()]
'''

#---------------------------------------------------------------------------------------------------------------------------------------------
def amalgamate(reference_list,tract_numbers):
    refsplus = set([])
    if len(reference_list) != len(tract_numbers):
        print 'mismatched list lengths'
    else:
        for i in range(len(reference_list)):
            for r in reference_list[i]:
                refsplus.add((r,tract_numbers[i]))    
        refsplus=list(refsplus)
        allrefs=[]
        allclass=[]
        for (r,c) in refsplus:
            allrefs.append(r)
            allclass.append(c)
        return allrefs, allclass
    
    remapped={}
    for tn in tract_numbers:
        try:
            remapped[tn] += [r]
        except:
            remapped[tn] == [r]
    return remapped

#---------------------------------------------------------------------------------------------------------------------------------------------
def map_bundle_to_refs(bundle, bundle_indices, refs, ref_indices, refclass):
 
    min_ref_class = []
    for track in bundle:
        d = []
        for ref in refs:
            d.append(pf.zhang_distances(track,ref,metric='min'))
        #print np.argmin(d), len(refclass), len(ref_indices), len(refs)
        min_ref_class.append(refclass[np.argmin(d)])

    assignment={}
    
    for c in range(1,21):
        
        assignment[c] = []        
        
    for (i,bi) in enumerate(bundle_indices):
            
        assignment[min_ref_class[i]].append(bi)
    
    return assignment
  

def relabel_them_all(no):
    
    path='/home/eg01/Data/PBC/pbc2009icdm'

    refdic = pbc.load_pickle('/home/ian/tractarian/refdic.pkl')
    
    brains = [(1,1),(1,2),(2,1),(3,1),(3,2)]
    
    
    for (b,s) in [brains[no]]:
        print b,s
        
        tA= refdic[(b,s)]['template_tracks']
        iA= refdic[(b,s)]['template_indices']
        
        tR = [] 
        iR = []         
        cR=[]
        
        for bundle in range(1,21):
            
            tmp=refdic[(b,s)]['reference_tracks'][bundle] 
            
            tR+=tmp
            
            iR+=refdic[(b,s)]['reference_indices'][bundle]     
            
            #print len(tmp),np.repeat(bundle,len(tmp)).astype(int)
            
            cR+=list(np.repeat(bundle,len(tmp)).astype(int))       

        relabelling = {}

        for bundle in range(1,21):
            print 'Relabelling', bundle
            #print len(tR),len(iR),len(cR)
            relabelling[bundle] =  map_bundle_to_refs(tA[bundle], iA[bundle], tR, iR, cR)
        
                
        IDS=ids
        
        for bundle1 in range(1,21):
            print 'Remapping', 
            for bundle2 in range(1,21):
            
                try:
                
                    IDS[bundle2]['indices']+=relabelling[bundle1][bundle2]
                    
                except :
                    IDS[bundle2]['indices']=relabelling[bundle1][bundle2]
        
        print 'Saving',b,s            
        pbc.save_pickle(path+'/Relabelling_'+str(b)+'_'+str(s)+'.pkl',IDS)
    
                    
'''        
tA11 = refdic[(1,1)]['template_tracks']

iA11 = refdic[(1,1)]['template_indices']

tR11_1_and_7 = refdic[(1,1)]['reference_tracks'][1] +refdic[(1,1)]['reference_tracks'][7]

iR11_1_and_7, cR11_1_and_7 = \
    amalgamate( [refdic[(1,1)]['reference_indices'][1], refdic[(1,1)]['reference_indices'][7]], [1,7])


relabelling =  map_bundle_to_refs(tA11[1], iA11[1], tR11_1_and_7, iR11_1_and_7, cR11_1_and_7)

print relabelling.keys()

print len(relabelling[1])
print len(relabelling[7])
'''