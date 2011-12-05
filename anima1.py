import pylab
from pylab import matplotlib
import time

pylab.ion()

tstart = time.time()		# for profiling
x = 50
y = 50
blob, = pylab.plot([x],[y],'bo')
pylab.axis([0,100,0,100],'equal')
for i in pylab.arange(1,200):
	x += 2*pylab.randint(0,2)-1
	y += 2*pylab.randint(0,2)-1
	blob.set_xdata([x])
	blob.set_ydata([y])
#	plot([x],[y],'bo')
	pylab.draw()			# redraw the canvas

