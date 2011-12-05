from time import time
import numpy as np
from numpy import array as RR
import test_cython_pointers as tcp

A=np.random.rand(3,2,2,2)
I=RR([1,1,1,1])
K=RR([2,0,0,1])
L=RR([2,1,0,1])
print I, A[1,1,1,1]
print I, tcp.get_element(I,A)
print K, A[2,0,0,1]
print K, tcp.get_element(K,A)
print L, A[2,1,0,1]
print L, tcp.get_element(L,A)


t1=time()
A=np.random.rand(10,10,10,10)
t2=time()
print t2-t1, A.sum()
res=tcp.sum_array(A)
t3=time()
print t3-t2,res
res2=A.ravel().sum()
t4=time()
print t4-t3,res2
s=0
for i in range(10):
    s+=tcp.sum_array(A)
t5=time()
print t5-t4,s

A=np.random.rand(10,1,1,2)
A2=A.copy()
print '----------------'
print A-tcp.addone(A)
print '----------------'
print A2-tcp.addone(A)
print '----------------'
print A2-tcp.addone(A2)
#A=np.random.rand(10,1,1,2)
#print A

print '#################'
a=np.array([1,1,0,1])
A=np.ones((2,2,2,2))
B=np.ones((2,2,2,2))

C=tcp.process_array(a.copy(),A,B)
print C

print '[[[[[[[[[]]]]]]]]]'
l=[1,2,3]
it=10**7
t1=time()
print len(tcp.process_list(l,it))
l=[1,2,3]
t2=time()
print t2-t1

for j in xrange(it):
    l.append(j)

t3=time()
print len(l)
print t3-t2

print 'ratio',(t3-t2)/(t2-t1)

#print tcp.fast_array_return()
t4=time()
for i in range(10**2):
    a=tcp.fast_array_return(100)
t5=time()
print t5-t4

    
 





