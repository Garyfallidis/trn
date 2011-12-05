#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Author: Eleftherios Garyfallidis
Description: Implementation of different methods using Markov Random Fields.
'''

import form

#from draw import pyplot as plt
import pylab as plt

import scipy as sp


    
    
def denoise2DICM():
    '''
    Example taken from Bishop's book p.389
    Iterated Conditional Modes (ICM) Discrete Case
    Just a cross neighbourhood is allowed here. Other configurations are easy to apply as well.
    '''
    
    im=form.loadpic('mrf2.png')
    
    im=sp.asarray(im,dtype='float64')    
    im[im>127]=255
    im[im<=127]=0
    print im.min(),im.max()
    plt.matshow(im)
    plt.show()

    #observed noisy image
    Y=sp.asarray(im,dtype='float64')
    Y[Y==255]=1
    Y[Y==0]=-1
    
    #state of pixel 
    #X=sp.ones(Y.shape)
    X=Y
        
    beta,eta,h=1,1,0

    E=sp.zeros(Y.shape)
    
    converge=False
    
    while(not converge):
        
        Eprev=sp.sum(E)
        for r in xrange(1,Y.shape[0]-1):
            for c in xrange(1,Y.shape[1]-1):
            
                #calculate energy for positive 
                x=1
                
                Ep=-eta*x*Y[r,c]-beta*x*X[r,c+1]-beta*x*X[r,c-1] - beta*x*X[r-1,c]-beta*x*X[r+1,c]
                
                #calculate energy for negative
                x=-1
                En=- eta*x*Y[r,c] - beta*x*X[r,c+1] - beta*x*X[r,c-1] - beta*x*X[r-1,c] - beta*x*X[r+1,c]
                
                if En<Ep:
                    X[r,c]=-1
                    E[r,c]=En
                else:
                    X[r,c]=1
                    E[r,c]=Ep
                
                #E[r,c]=-eta*X[r,c]*Y[r,c]-beta*X[r,c]*X[r-1,c]-beta*X[r,c]*X[r+1,c]
                
                
        #plt.matshow(E)       
        if Eprev-sp.sum(E)==0:
            converge=True
            
        plt.matshow(X)       
    
    return E,Y,X

def denoise1Dicm(noise='poisson',prior='quadratic',lambd=1,noise_amp=.5):
    
    '''
    Iterated Conditional Modes (ICM) Continuous Case
    
    From Komodakis presentation use 
    
    
    gaussian & quadratic
    
    1/2 sum((y_i-x_i)^2) + lambd*sum(x_i-x_i-1)**2
    
    poisson & quadratic
    
    1/2 sum((x_i-y_i*log(x_i))^2) + lambd*sum(x_i-x_i-1)**2

    gaussian & linear prior
    
    1/2 sum((y_i-x_i)^2) + lambd*abs(x_i-x_i-1)
        
    
    '''
        
    N=4
    bins=1000
    l=lambd
    
    u=sp.linspace(0,N-1,bins)
    print u
    
    y=sp.cos(2*sp.pi*u)
    print y
    
    plt.plot(u,y,'.')
    plt.show()
    
    if noise=='gaussian':
        y=y+ noise_amp*sp.random.normal(size=len(y))+2
    
    if noise=='poisson':
        y=y+noise_amp*sp.random.poisson(size=len(y))+2
    

    #plt.figure()
    line,=plt.plot(u,y,'.')
    #plt.show()
    #return
    
    x=sp.zeros(len(y))
    #x[0]=

    x=y
    #plt.ion()
    count=0
    while(True):
    
        tmpx=x
        if noise=='gaussian' and prior=='quadratic':
            
            print noise, prior
        
            for i in xrange(1,len(x)-1):
            
                x[i]=(y[i]+2*l*(x[i-1]+x[i+1]))/(1+4*l)
                
            x[0]=(y[0]+2*l*x[1])    /(1+2*l)
            x[-1]=(y[-1]+2*l*x[-1])    /(1+2*l)
            
        if noise=='poisson' and prior=='quadratic':

            print noise, prior
            
            for i in xrange(1,len(x)-1):
         
                x[i]=(-1+l*x[i]*(x[i-1]+x[i+1])+sp.sqrt((x[i-1]+x[i+1])**2+16*l*y[i]))/(8*l)
                
            x[0]=(-1+l*x[0]*x[1]+sp.sqrt(x[1]**2+8*l*y[0]))/(4*l)
            x[-1]=(-1+l*x[-1]*x[-2]+sp.sqrt(x[-2]**2+8*l*y[-1]))/(4*l)
        
        if noise=='gaussian' and prior=='linear':
            print 'not implemented yet'
            pass
            
        plt.plot(u,x)
        #line.set_ydata(x)
        #plt.draw()
        
        #'''
        count=count+1
        if count==100:
            break
        #'''
        #if sum(x)-sum(tmpx) < 0.0001:
        #    break
        
    
    plt.figure()
    plt.plot(u,x)

if __name__ == "__main__":
    
    #experICM()
    #denoise1D()
    pass
    