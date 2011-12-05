#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
    Author: Eleftherios Garyfallidis & Ian Nimmo-Smith
    Info:
        2d image, 2d plotting, camera and 3d plotting through mayavi.mlab
        
    Example:
    
    x=sp.rand(30), pylab.plot(x), pylab.hist(x),
    pylab.imshow(255*sp.rand(40,40)), pylab.show()

'''

try:
    #import matplotlib
    from matplotlib import pyplot, mpl
except:
    print('matplotlib is not installed.')

try:
    import PIL
except:
    print('PIL is not installed.')
    
try:
    import opencv
except:
    print('Opencv is not installed')

try:
    from enthought.mayavi.mlab import *
except:
    print('Mayavi is not installed')

import form
import scipy as sp

def testinterp():
    
    xp = [1, 2, 3]
    fp = [3, 2, 0]
    sp.interp(2.5, xp, fp)
    
    sp.interp([0, 1, 1.5, 2.72, 3.14], xp, fp)
    
    UNDEF = -99.0
    sp.interp(3.14, xp, fp, right=UNDEF)
    
    #Plot an interpolant to the sine function:
    
    x = sp.linspace(0, 2*sp.pi, 10)
    y = sp.sin(x)
    xvals = sp.linspace(0, 2*sp.pi, 50)
    yinterp = sp.interp(xvals, x, y)
    
    pyplot.plot(x, y, 'o')
    pyplot.plot(xvals, yinterp, '-x')
    pyplot.show()

def testhist():
    
    fname='/home/eg309/Data/AquaTermi_lowcontrast.JPG'
    #fname='/home/eg01/Data/AquaTermi_lowcontrast.JPG'
    
    #imhist,bins = sp.histogram(im.flatten(),nbr_bins,normed=True,new=True)
    
    im=form.loadpic(fname)
    
    print im.shape
    
    nbr_bins=256

    #Return the hist
    imhist,bins = sp.histogram(im.flatten(),nbr_bins,normed=True,new=True)
    
    #pylab.hist(im.flatten(),nbr_bins,normed=True,new=True)
    
    pyplot.figure(1)
    
    print 'imhist',imhist.shape
    pyplot.plot(imhist)
    
    pyplot.figure(2)
    #fig.add_subplot(2,1,1)
    
    pyplot.imshow(im, cmap=mpl.cm.gray)
    
    pyplot.show()
    
    pyplot.figure(3)    
    
    cdf = imhist.cumsum() #cumulative distribution function
    cdf = 255 * cdf / cdf[-1] #normalize

    print 'bins.shape',bins.shape
    print 'cdf.shape',cdf.shape
    #use linear interpolation of cdf to find new pixel values
    im2 = sp.interp(im.flatten(),bins[10:-1],cdf[10:])    
    
    print 'im.min',im.min()
    print 'im.max',im.max()
    
    print 'im.min',im2.min()
    print 'im.max',im2.max()
        
    pyplot.imshow(im2.reshape(im.shape), cmap=mpl.cm.gray)
    
    pyplot.show()
    
def testsubplot():
    
    print 'Test'
    pyplot.figure(1)                # the first figure
    pyplot.subplot(211)             # the first subplot in the first figure
    pyplot.plot([1,2,3])
    pyplot.subplot(212)             # the second subplot in the first figure
    pyplot.plot([4,5,6])


    pyplot.figure(2)                # a second figure
    pyplot.plot([4,5,6])            # creates a subplot(111) by default

    pyplot.figure(1)                # figure 1 current; subplot(212) still current
    pyplot.subplot(211)             # make subplot(211) in figure1 current
    pyplot.title('Easy as 1,2,3')   # subplot 211 title

def animationexample():
    from pylab import *
    import time

    ion()

    tstart = time.time()               # for profiling
    x = arange(0,2*pi,0.01)            # x-array
    line, = plot(x,sin(x))
    for i in arange(1,200):
        line.set_ydata(sin(x+i/10.0))  # update the data
        draw()                         # redraw the canvas

    print 'FPS:' , 200/(time.time()-tstart)


if __name__ == "__main__":

    testhist()
    #testsubplot()
    
    