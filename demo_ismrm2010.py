import pbc
from dipy.viz import fos

path='/home/eg309/Data/'

tracks1zshift=pbc.load_pickle(path+'t1zs.pkl')
tracks2zshift=pbc.load_pickle(path+'t2zs.pkl')
tracks3zshift=pbc.load_pickle(path+'t3zs.pkl')

track2track=pbc.load_pickle(path+'t2t.pkl')
track2track2=pbc.load_pickle(path+'t2t2.pkl')

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