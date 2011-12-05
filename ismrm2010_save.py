import numpy as np
from dipy.viz import fos
from dipy.core import track_metrics as tm
from dipy.core import track_learning as tl
from dipy.io import trackvis as tv
import dipy.core.performance as pf

def corresponding_bundles(indices,tracks1,tracks2):
	''' Detect similar bundles
	'''

		

	pass

def corresponding_tracks(indices,tracks1,tracks2):
	''' Detect similar tracks in different brains
	'''
    
	li=len(indices)
	track2track=np.zeros((li,3))
	cnt=0
	for i in indices:        
        
		rt=[pf.zhang_distances(tracks1[i],t,'avg') for t in tracks2]
		rt=np.array(rt)               

		track2track[cnt-1]=np.array([cnt,i,rt.argmin()])        
		cnt+=1
        
	return track2track.astype(int)


#br1path='/backup/Data/Eleftherios/CBU090133_METHODS/20090227_145404/Series_003_CBU_DTI_64D_iso_1000/dtk_dti_out/dti_RK2.trk'
#br2path='/backup/Data/Eleftherios/CBU090134_METHODS/20090227_154122/Series_003_CBU_DTI_64D_iso_1000/dtk_dti_out/dti_RK2.trk'

#br1path='/backup/Data/Eleftherios/CBU090133_METHODS/20090227_145404/Series_003_CBU_DTI_64D_iso_1000/dtk_dti_out/dti_FACT.trk'
#br2path='/backup/Data/Eleftherios/CBU090134_METHODS/20090227_154122/Series_003_CBU_DTI_64D_iso_1000/dtk_dti_out/dti_FACT.trk'

br1path='/backup/Data/Eleftherios/CBU090133_METHODS/20090227_145404/Series_003_CBU_DTI_64D_iso_1000/dtk_dti_out/dti_FACT.trk'
br2path='/backup/Data/Eleftherios/CBU090134_METHODS/20090227_154122/Series_003_CBU_DTI_64D_iso_1000/dtk_dti_out/dti_FACT.trk'
br3path='/backup/Data/Eleftherios/CBU090133_METHODS/20090227_145404/Series_003_CBU_DTI_64D_iso_1000/dtk_dti_out/dti_RK2.trk'



min_len=20
down=20
rand_tracks=-1 #default 10
min_search_len=70
max_search_len=140

'''
corr_mat_demo=np.array([[ 1, 10560,  3609],[ 2, 17872, 15377],[ 3,  6447,  3897],
 			[4, 18854,  6409], [ 5, 14416,  4515], [ 6,  6071, 12155], [ 7,  9956, 13913],
 			[8, 10853, 15572], [ 9, 13280,  8461], [ 0, 11275,  9224]])
'''

corr_mat_demo=np.array([[ 1, 10560,  3609],[ 2, 17872, 15377],[ 3,  6447,  3897],
 			[4, 18854,  6409], [ 5, 14416,  4515], [ 7,  9956, 13913],
 			[8, 10853, 15572], [ 9, 13280,  8461], [ 0, 11275,  9224]])



print 'Minimum track length', min_len, 'mm'
print 'Number of segments for downsampling',down
print 'Number of tracks for detection',rand_tracks
print 'Minimum searched track length', min_search_len, 'mm'
print 'Maximum searched track length', max_search_len, 'mm'

tracks1,hdr1=tv.read(br1path)
tracks2,hdr2=tv.read(br2path)
tracks3,hdr3=tv.read(br3path)

#Load only track points, no scalars or parameters.

tracks1=[t[0] for t in tracks1]
tracks2=[t[0] for t in tracks2]
tracks3=[t[0] for t in tracks3]

print 'Before thresholding'

print len(tracks1)
print len(tracks2)
print len(tracks3)

print hdr1['dim']
print hdr2['dim']
print hdr3['dim']

#Apply thresholds

tracks1=[t for t in tracks1 if tm.length(t) > min_len]
tracks2=[t for t in tracks2 if tm.length(t) > min_len]
tracks3=[t for t in tracks3 if tm.length(t) > min_len]

print 'After thresholding'

print len(tracks1)
print len(tracks2)
print len(tracks3)

print 'Downsampling'

tracks1z=[tm.downsample(t,down) for t in tracks1] 
tracks2z=[tm.downsample(t,down) for t in tracks2] 
tracks3z=[tm.downsample(t,down) for t in tracks3] 

print 'Detecting random tracks'

lt1=len(tracks1)
lt2=len(tracks2)
lt3=len(tracks3)

if rand_tracks==-1:
	#use already stored indices
	t_ind=corr_mat_demo[:,1]
else:
	
	#find the size in number of tracks of the smallest dataset
	mlt=np.min(lt1,lt2)

	#find some random tracks
	rt=0
	t_ind=[]
	while rt < rand_tracks:
	
		ind=mlt*np.random.rand()

		#indices of random fibers
		ind=int(round(ind))

		if tm.length(tracks1[ind]) >= min_search_len and tm.length(tracks1[ind]) <= max_search_len:
			t_ind.append(ind)
			rt+=1
				
	t_ind=np.array(t_ind)

print 'Indices of tracks for detection', t_ind

print 'Finding corresponding tracks'
track2track=corresponding_tracks(t_ind,tracks1z,tracks2z)
track2track2=corresponding_tracks(t_ind,tracks1z,tracks3z)

print 'First Correspondance Matrix'
print track2track
print 'Second Correspondance Matrix'
print track2track2

print 'Fos loading'


#fos.add(r,fos.line(tracks1,fos.red,opacity=0.01))
#fos.add(r,fos.line(tracks2,fos.cyan,opacity=0.01))

tracks1zshift=[t+np.array([-70,0,0]) for t in tracks1z]
tracks2zshift=[t+np.array([70,0,0]) for t in tracks2z]
tracks3zshift=[t+np.array([210,0,0]) for t in tracks3z]

import pbc
pbc.save_pickle('t1zs.pkl',tracks1zshift)
pbc.save_pickle('t2zs.pkl',tracks2zshift)
pbc.save_pickle('t3zs.pkl',tracks3zshift)

pbc.save_pickle('t2t.pkl',track2track)
pbc.save_pickle('t2t2.pkl',track2track2)

'''
r=fos.ren()

fos.add(r,fos.line(tracks1zshift,fos.red,opacity=0.02))
fos.add(r,fos.line(tracks2zshift,fos.cyan,opacity=0.02))
fos.add(r,fos.line(tracks3zshift,fos.blue,opacity=0.02))

print 'Show track to track correspondence br1 br2'
for i in track2track:
	fos.add(r,fos.line(tracks1zshift[i[1]],fos.yellow,opacity=0.5,linewidth=3))
	fos.label(r,str(i[0]),tracks1zshift[i[1]][0],(4,4,4),fos.white)

	fos.add(r,fos.line(tracks2zshift[i[2]],fos.yellow,opacity=0.5,linewidth=3))
	fos.label(r,str(i[0]),tracks2zshift[i[2]][0],(4,4,4),fos.white)

print 'Show track to track correspondence br1_FACT and br2_RK2'
for i in track2track2:
	fos.add(r,fos.line(tracks3zshift[i[2]],fos.yellow,opacity=0.5,linewidth=3))
	fos.label(r,str(i[0]),tracks3zshift[i[2]][0],(4,4,4),fos.white)

fos.show(r)

'''

