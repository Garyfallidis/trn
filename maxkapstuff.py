#import os

#os.system('rm maxkapS1.o')
#os.system('rm maxkapS1.so')
#os.system('gcc -O2 -c -fPIC -l/usr/include/python2.6/ maxkapS1.c')
#os.system('gcc -shared maxkapS1.o -o maxkapS1.so')

from time import time
import itertools

import ctypes
max_kappa=ctypes.cdll.LoadLibrary('./maxkapS1.so')

import numpy as np
import numpy.random as rand

'''
data = np.array([[0, 4], [1, 3], [0, 4], [2, 1], [1, 4],
        [4, 2], [1, 3], [3, 4], [2, 3], [3, 3],
        [3, 4], [0, 3], [0, 0], [1, 3], [3, 3],
        [4, 4], [2, 3], [2, 0], [2, 0], [2, 3],
        [3, 0], [2, 1], [1, 4], [3, 2], [3, 1], 
        [0, 4], [4, 2], [0, 3], [2, 0], [0, 4],
        [3, 3], [0, 0], [4, 4], [4, 0], [4, 1],
        [2, 4], [0, 1], [3, 3], [0, 2], [3, 2],
        [0, 1], [4, 3], [2, 0], [0, 1], [1, 2],
        [1, 0], [1, 4], [2, 3], [1, 4], [1, 3],
        [3, 1], [4, 4], [4, 3], [2, 3], [3, 4],
        [4, 1], [1, 0], [2, 3], [0, 1], [3, 4],
        [3, 0], [3, 3], [1, 2], [3, 4], [3, 3],
        [4, 0], [3, 1], [3, 1], [2, 2], [0, 0],
        [1, 0], [4, 3], [2, 0], [4, 3], [3, 4],
        [2, 3], [3, 4], [1, 2], [1, 1], [2, 4],
        [2, 3], [1, 1], [3, 3], [2, 1], [4, 2],
        [3, 1], [4, 2], [2, 1], [3, 4], [3, 2],
        [2, 4], [0, 1], [0, 4], [3, 4], [0, 2],
        [1, 4], [4, 1], [0, 3], [4, 2], [1, 0]])
'''

def corresp(data):
    m = max(data[:,0])+1
    n = max(data[:,1])+1
    M = np.max(m,n)
    N = np.zeros((M,M))
    for i in range(len(data)):
        if data[i,0]>=M or data[i,1]>=M:
            print i,data[i,0],data[i,1]
            break
        N[data[i,0],data[i,1]]+=1

    return N

#N=np.array([[88, 10,  2],[14, 40, 6],[18, 10,12]],dtype='float')
def kappa(N):
    #print N
    rowsum = np.sum(N,axis=1)
    #print 'row sums:', rowsum
    colsum = np.sum(N,axis=0)
    #print 'col sums:', colsum
    total = np.sum(N)
    #print 'total', total
    agree = np.sum([N[i,i] for i in range(min(N.shape))])/total
    #print 'agree', agree
    #print rowsum.shape, colsum.shape
    chance = np.sum(colsum*rowsum)/total**2
    #print 'chance', chance
    kappa = (agree-chance)/(1-chance)
    #print 'kappa', kappa
    return kappa

'''
perm = [1,0,3,4,2]
N = corresp(data)
print kappa(N),kappa(N[:,perm])
'''


'''
for y,v in results:
    print y,v 
'''

def maxKappa(x,y):
    newdata = np.row_stack((x,y))
    newdata = newdata.astype('f4')
    newdata=newdata+1 #add one to labels ... so they start at 1 in C-fashion
    maxclasses = np.max([np.max(x),np.max(y)])+1
    nclasses = np.array(maxclasses).astype('i4')
    nunits = np.array(data.shape[0]).astype('i4')
    nmeths = np.array(data.shape[1]).astype('i4')
    #print 'setting up size: ', (nclasses+1)*nmeths*(nmeths-1)/2
    ans=np.zeros((nclasses+1)*nmeths*(nmeths-1)/2).astype('f4')
    ncl=np.array([nclasses],'i4')
    order = np.array(np.arange(nclasses),'i4')
    max_kappa.clustComp(ans.ctypes.data,newdata.ctypes.data,nunits.ctypes.data,nmeths.ctypes.data,ncl.ctypes.data,order.ctypes.data)
    maxkappa = ans[0]
    maxorder = np.argsort(ans[1:(nclasses+1)])
    return maxkappa, maxorder


import sys
sys.path += ['build/lib.linux-i686-2.2/']
import hungarian

from munkres import Munkres, print_matrix

for classes in np.arange(50,4200,100):
    print '\n>>>>>> Working with ', classes, ' classes'
    '''
    data=rand.random_integers(0,classes-1,size=(10**3,2))
    x = data[:,0]
    y = data[:,1]
    if classes <= 10:
        t = time()
        maxkappa, maxorder = maxKappa(x,y)
        t = time()-t
        print '\n>>> Annealing approximation ...'
        print 'time      ', t
        print 'maxkappa  ', 100*maxkappa
        #print 'max order ', maxorder
    N = corresp(data)
    '''
    N = np.array(np.diag(classes*[10]),'i2')
    q = np.random.permutation(classes)
    N = N[:,q]
    if classes <=9:
        m = itertools.permutations(range(classes))
        t = time()
        results = [(y,kappa(N[:,y])) for y in m]
        t = time()-t
        vals = [v for y,v in results]
        imax = np.argmax(vals)
        pmax, vmax = results[imax]
        print '\n>>> Exhaustive search ...'
        print 'time      ', t
        print 'maxkappa  ', 100*vmax
        #print 'max order ', pmax
    '''
    print '\n>>> Hungarian matching (1) ...'
    t=time()
    my_munkres = Munkres()
    mapping = my_munkres.compute(-N)
    t=time()-t
    print '>>> time elapsed =', t
    #print_matrix(N, msg='Maximum matching:')
    total = 0
    map = np.zeros((len(mapping)),'i4')
    for row, column in mapping:
        map[row]=column
        value = N[row,map[row]]
        total += value
        #print '(%d, %d) -> %d' % (row, column, value)
    print 'percent agreements: ', 100*total/np.sum(N)
    #print 'total agreements (percent): %d' % 100*np.sum(np.diag(N[:,map]))/np.sum(N)
    #print np.sum(np.diag(N[:,map]))
    #print 'optimal assignment', map
    print 'kappa', 100*kappa(N[:,map])
    #print map
    print '\n>>> Hungarian matching (2) ...'
    t=time()
    Nnegintlist = [[int(-c) for c in b] for b in N]
    print 'time to build list of list of integers =', time()-t
    t=time()
    h = hungarian.HungarianSolver(Nnegintlist)
    h.solve()
    t=time()-t
    print '>>> time elapsed =', t
    p = h.get_assignment()
    total = 0
    for row in range(min(N.shape)):
        total += N[row,p[row]]
    print 'percent agreements: ', 100*total/np.sum(N)
    print 'kappa', 100*kappa(N[:,p])
    print p
    '''

    print '\n>>> Hungarian matching (APC: Lawler - implemented by G. CARPANETO, S. MARTELLO, P. TOTH) ...'
    import hung_APC 
    t=time()
    mapping, cost, errorcode  = hung_APC.apc(-N)
    if errorcode != 0:
        print 'APC error code %d: need to increase MAXSIZE in APC.f to handle this problem' % (errorcode)
    t=time()-t
    print '>>> time elapsed =', t
    total=np.sum(np.diag(N[:,mapping-1]))
    #total = -cost
    print 'cost', cost, 'perm length',len(mapping)     
    print 'percent agreements: ', 100*total/np.sum(N)
    print 'kappa', 100*kappa(N[:,mapping-1])
        
