
cimport cython

import numpy as np
cimport numpy as cnp

cdef inline long offset(long *indices,long *strides,\
                             int lenind, int typesize) nogil:
    cdef int i
    cdef long summ=0
    for i from 0<=i<lenind:
        summ+=strides[i]*indices[i]        
    summ/=<long>typesize
    return summ

cdef double _addone(double *a,int lena):
    cdef int i
    cdef double tmp
    for i from 0<=i<lena:
        tmp=a[i]
        a[i]=a[i]+1
        #print tmp    
    return 1

def addone(cnp.ndarray[double ,ndim=4] A):
    cdef double *pA=<double *>A.data
    _addone(pA,A.size)
    return A

def get_element(cnp.ndarray[long, ndim=1] iA,\
                    cnp.ndarray[double, ndim=4] A):    
    cdef double *pA=<double *>A.data
    #addinarray(<double *>A.data,A.size)
    return pA[offset(<long*>iA.data,<long *>A.strides,A.ndim,8)]

cdef double _direct_sum(double *pA,long size) nogil:
    cdef long i,j
    cdef double summ=0, summ2=0
    for i from 0<=i<size:
        summ+=pA[i]
    return summ

#@cython.boundscheck(False)
#@cython.wraparound(False)
cdef double _sum_array(long *iA, long* sA,\
                                  double* pA,\
                                  int s0,int s1,\
                                  int s2,int s3) nogil:
    cdef int i,j,k,l
    cdef double summ
    for i from 0<=i<s0:
        for j from 0<=j<s1:
            for k from 0<=k<s2:
                for l from 0<=l<s3:
                    iA[0]=i
                    iA[1]=j
                    iA[2]=k
                    iA[3]=l
                    summ+=pA[offset(iA,sA,4,8)]
    return summ
   
def sum_array(cnp.ndarray[double, ndim=4] A):
    cdef:
        int i,j,k,l
        double summ=0
        int s0=A.shape[0]
        int s1=A.shape[1]
        int s2=A.shape[2]
        int s3=A.shape[3]
        double *pA=<double *>A.data
        long iA[4]
        long *sA=<long *>A.strides    

    #return _sum_array(iA,sA,pA,s0,s1,s2,s3)
    return _direct_sum(pA,A.size)    
    
def process_array(cnp.ndarray[long, ndim=1] a,\
                      cnp.ndarray[double, ndim=4] A,\
                      cnp.ndarray[double, ndim=4] B):
    cdef:
        double *pA=<double *>A.data
        double *pB=<double *>B.data
        long *pa=<long *>a.data
        cnp.ndarray[double, ndim=4] C=np.zeros((2,2,2,2))
        double *pC=<double *>C.data
        long *sA=<long *>A.strides
        int i=0
          
    for i in range(A.size):
        pC[i]=pA[i]+pB[i]

    print(A.strides[0],A.strides[1],A.strides[2],A.strides[3])
    print(sA[0],sA[1],sA[2],sA[3])
    print(a)
    print(pa[0],pa[1],pa[2],pa[3])
    print(offset(pa,sA,4,8))

    pC[offset(pa,sA,A.ndim,8)]=pA[offset(pa,sA,A.ndim,8)] +\
        2*pB[offset(pa,sA,B.ndim,8)]

    
    return C
    
def process_list(object A,long it):

    cdef long i

    #print A.data

    for i from 0<=i<it:
        A.append(i)
    return A

@cython.boundscheck(False)
@cython.wraparound(False)
def fast_array_return(long it):

    cdef double A[30000]

    cdef long i

    

    with nogil:

        for i from 0<=i<it:
            A[i]=<double>i

    cdef cnp.ndarray[double,ndim=1] a=np.empty(it)

    with nogil:

        for i from 0<=i<it:
            a[i]=A[i]
        
    return a
        
