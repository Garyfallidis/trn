import pylab
from pylab import matplotlib
import time
import numpy as np

pylab.ion()
n=30
m=1
gn = 100
step = 1
spacing = 1
iterations = 50
centre = np.array([gn/2,gn/2],).astype(int)
#colour =  zip(np.arange(0,1.,1./(n*m)),np.arange(1,0,-1./(m*n)),np.arange(0.5,0.9,0.4/(n*m)))

tstart = time.time()		# for profiling

ogrid = np.lib.index_tricks.nd_grid(sparse=False)

grid = ogrid[0:n,0:n].astype(int)

#print '* %d %d %d %d' % (np.min(grid[0].flatten()), np.max(grid[0].flatten()), np.min(grid[1].flatten()), np.max(grid[1].flatten()))

x = np.tile(grid[0].flatten(),m)
y = np.tile(grid[1].flatten(),m)

#print '* %d %d %d %d' % (np.min(x), np.max(x), np.min(y), np.max(y))

x = centre[0]+x*spacing-spacing*n/2
y = centre[1]+y*spacing-spacing*n/2

#print '* %d %d %d %d' % (np.min(x), np.max(x), np.min(y), np.max(y))

blobs = []

#pylab.figure()

#pylab.ioff()

for i in range(len(x)):
	if np.abs(x[i]-centre[0])+np.abs(y[i]-centre[1]) <= n/2:
		#blob, = pylab.plot(x[i:i+1],y[i:i+1], marker='o', color=colour[i % len(colour) ], alpha=0.2)
		blob, = pylab.plot(x[i:i+1],y[i:i+1], marker='o')
		blob.set_color((1,0,0))
	else:
		blob, = pylab.plot(x[i:i+1],y[i:i+1], marker='o')
		blob.set_color((0,0,1))
	blobs.append(blob)
	#print '%d %d %d %d %d' % (i, min(x), max(x), min(y), max(y))

pylab.axis([0,100,0,100],'equal')
#pylab.ion()
pylab.draw()

print "Press return to continue ..."
raw_input()

for j in np.arange(iterations):
	pylab.ioff()
	x = x+step*(2*np.random.binomial(1,0.5,n*n)-1)
	y = y+step*(2*np.random.binomial(1,0.5,n*n)-1)
	#print '%d %d %d %d %d' % (j, np.min(x), np.max(x), np.min(y), np.max(y))
#~ #	pylab.clf()
	for i in np.arange(len(x)):
		blobs[i].set_xdata([x[i:i+1]])
		blobs[i].set_ydata([y[i:i+1]])
	pylab.title('Iteration '+str(j+1))
	pylab.ion()
	pylab.draw()			# redraw the canvas

print "Press return to continue ..."
raw_input()

