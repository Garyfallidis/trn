#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Author: Eleftherios Garyfallidis
Description: General Usage Python functions e.g. for exception handling, system information etc.
'''

import sys
import traceback
import platform
import time
import array
import os

def timing(f, n, a):
    '''
    Time a function f with argument a n times
    '''
    print f.__name__,
    r = range(n)
    t1 = time.clock()
    for i in r:
        f(a); f(a); f(a); f(a); f(a); f(a); f(a); f(a); f(a); f(a)
    t2 = time.clock()
    print round(t2-t1, 3)


def exceptinfo(maxTBlevel=5):
    
    cla, exc, trbk = sys.exc_info()
    excName = cla.__name__
    try:
        excArgs = exc.__dict__["args"]
    except KeyError:
        excArgs = "<no args>"
        excTb = traceback.format_tb(trbk, maxTBlevel)
    return (excName, excArgs, excTb)

def systeminfo():

    arch=platform.architecture()
    release=platform.release()
    uname=platform.uname()
    plat=sys.platform
    pyversion=sys.version.split(' ')[0]

    #print(sys.getwindowsversion())
    
    info='This is a '+arch[0]+' '+uname[0]+' os.' 
    if uname[0]=='Linux':
        info2='Exact os version is ' + uname[2]+ '.'
    else:
        info2='Exact os version is ' + uname[2] +' '+ uname[3]+'.'

    info3='The name of this computer is ' + uname[1]+'.'    
    
    info4='The version of Python is '+pyversion+'.\n'
    

    print(info)
    print(info2)
    print(info3)
    print(info4)


    if uname[0]=='Windows':

        try:
            from ctypes.wintypes import windll
       
            class MEMORYSTATUS(Structure):

                _fields_ = [
                    ('dwLength', DWORD),
                    ('dwMemoryLoad', DWORD),
                    ('dwTotalPhys', DWORD),
                    ('dwAvailPhys', DWORD),
                    ('dwTotalPageFile', DWORD),
                    ('dwAvailPageFile', DWORD),
                    ('dwTotalVirtual', DWORD),
                    ('dwAvailVirtual', DWORD),
                ]
    
            x = MEMORYSTATUS()
            windll.kernel32.GlobalMemoryStatus(byref(x))    
            print('%d MB physical RAM left.' % (x.dwAvailPhys/1024**2))
            print('%d MB physical RAM in total.' % (x.dwTotalPhys/1024**2))
            print('%d MB total virtual memory.' % (x.dwTotalVirtual/1024**2))

        except:
            print('Ctypes.wintypes module is not installed.')

    if uname[0]=='Linux':

        import re
        re_meminfo_parser = re.compile(r'^(?P<key>\S*):\s*(?P<value>\d*)\s*kB')


        result = dict()
        for line in open('/proc/meminfo'):
            match = re_meminfo_parser.match(line)
            if not match:
                continue  # skip lines that don't parse
            key, value = match.groups(['key', 'value'])
            result[key] = int(value)
        
        print('The system has %d MB total memory.' %(result['MemTotal']/1024))
        
        '''
        for cpu usage see /proc/stat
        for loadvg  see /proc/loadavg
        '''

def number_of_processors():
    ''' number of virtual processors on the computer '''
    # Windows
    if os.name == 'nt':
        return int(os.getenv('NUMBER_OF_PROCESSORS'))
    # Linux
    elif sys.platform == 'linux2':
        retv = 0
        with open('/proc/cpuinfo','rt') as cpuinfo:
            for line in cpuinfo:
                if line[:9] == 'processor': retv += 1
        return retv
        # Please add similar hacks for MacOSX, Solaris, Irix,
        # FreeBSD, HPUX, etc.
    else:
        raise RuntimeError, 'unknown platform'



def listint2str(list):
    '''
    The fastest way to convert a list of integers into a string, presuming that the integers are ASCII values.
    http://www.python.org/doc/essays/list2str.html
    '''    
    return array.array('B',list).tostring()

def str2listint(string):
    '''
    The fastest way to create a list of integer ASCII values from a string
    http://www.python.org/doc/essays/list2str.html
    '''
    return array.array('b',string).tolist()


def usingswig():
    '''
    nano example.c
    nano example.i
    swig -python example.i
    gcc -c example.c example_wrap.c -I/usr/include/python2.6 -fPIC
    ld -shared example.o example_wrap.o -o _example.so
    /* File : example.c from www.swig.org/tutorial.html*/

     #include <time.h>
     double My_variable = 3.0;

     int fact(int n) {
         if (n <= 1) return 1;
         else return n*fact(n-1);
     }

     int my_mod(int x, int y) {
         return (x%y);
     }

     char *get_time()
     {
         time_t ltime;
         time(&ltime);
         return ctime(&ltime);
     }


    
    '''

def loadsharedobjectexample():
    
    '''
    Example of how to compile with g++ and load a shared library in python

    Lets assume that we have the following fibonacci.c file
    
    Then compile it using  g++ -o fib.so -shared fibonacci.c -fPIC
    
    This example is similar to http://www.scipy.org/Cookbook/Ctypes with some 
    changes to play with scipy rather than numpy.
    
    #ifdef  __cplusplus
    extern "C" {
    #endif

    #include <stdio.h>

    typedef struct Weather_t {
        int timestamp;
        char desc[12];
    } Weather;



    typedef void*(*allocator_t)(int, int*);

    /* Function prototypes */
    int fib(int a);
    void fibseries(int *a, int elements, int *series);
    void fibmatrix(int *a, int rows, int columns, int *matrix);
    void fibmatrix2(float** data, int len);
    void fibmatrix3(float** data, int rows, int columns);
    void foo(allocator_t allocator);
    void print_weather(Weather* w, int nelems);

    int fib(int a)
    {
        if (a <= 0) /*  Error -- wrong input will return -1. */
            return -1;
        else if (a==1)
            return 0;
        else if ((a==2)||(a==3))
            return 1;
        else
            return fib(a - 2) + fib(a - 1);
    }

    void fibseries(int *a, int elements, int *series)
    {
        int i;
        for (i=0; i < elements; i++)
        {
        series[i] = fib(a[i]);
        }
    }

    void fibmatrix(int *a, int rows, int columns, int *matrix)
    {
        int i, j;
        for (i=0; i<rows; i++)
            for (j=0; j<columns; j++)
            {
                matrix[i * columns + j] = fib(a[i * columns + j]);
            }
    }

    void fibmatrix2(float** data, int len) {
            float** x = data;
            for (int i = 0; i < len; i++, x++) {
                /* do something with *x */
            }
        }

    void fibmatrix3(float** data, int rows, int columns) {
        float** x = data;
        //for (int i = 0; i < len; i++, x++) {
            /* do something with *x */
        for (int i=0; i<rows; i++)
                for (int j=0; j<columns; j++)
            {
                x[i][j]=0;
        
                }
    }

    void foo(allocator_t allocator) {
       int dim = 2;
       int shape[] = {2, 3};
       float* data = NULL;
       int i, j;
       printf("foo calling allocator\n");
       data = (float*) allocator(dim, shape);
       printf("allocator returned in foo\n");
       printf("data = 0x%p\n", data);
       for (i = 0; i < shape[0]; i++) {
          for (j = 0; j < shape[1]; j++) {
             *data++ = (i + 1) * (j + 1);
          }
       }
    }

    void print_weather(Weather* w, int nelems)
    {
        int i;
        for (i=0;i<nelems;++i) {
            printf("timestamp: %d\ndescription: %s\n\n", w[i].timestamp, w[i].desc);
        }
    }

    #ifdef  __cplusplus
    }
    #endif

    '''
    
    import ctypes as ct
    import scipy as sp
   
    #name and directory of the shared lib
    lib = sp.ctypeslib.load_library('fib', '/home/eg01/Devel/BirchLGB/birch_py')
    
    #dealing with fib
    lib.fib.argtypes=[ct.c_int]
    lib.fib.restype=ct.c_int

    a=20
    print 'The '+str(a) +' fibonacci number is :'
    print lib.fib(int(a))

    #dealing with fibseries
    lib.fibseries.argtypes = [sp.ctypeslib.ndpointer(dtype = sp.intc), ct.c_int, sp.ctypeslib.ndpointer(dtype = sp.intc)]                                    
    lib.fibseries.restype  = ct.c_void_p

    b=[1,2,3,4,5,6,7,8,9]
    
    b = sp.asarray(b, dtype=sp.intc)
    result = sp.empty(len(b), dtype=sp.intc)
 
    lib.fibseries(b, len(b), result)
    print 'Series ',result
    
    #dealing with fibmatrix
    lib.fibmatrix.argtypes = [sp.ctypeslib.ndpointer(dtype = sp.intc),ct.c_int,ct.c_int,sp.ctypeslib.ndpointer(dtype = sp.intc)]
    lib.fibmatrix.restype= ct.c_void_p
    
    A=sp.array([[1,2,3],[4,5,6],[7,8,9]])
    
    rows=A.shape[0]
    cols=A.shape[1]
    
    A=sp.asarray(A,dtype=sp.intc)
    Mat=sp.empty_like(A)
    
    lib.fibmatrix(A,int(rows),int(cols),Mat)

    print 'Matrix'
    print Mat.reshape(rows,cols)
    
    #dealing with double pointers
    '''
    x = sp.array([[1,2,3], [4,5,6], [7,8,9]], 'i4')
    i4ptr = ct.POINTER(ct.c_int)
    D = (i4ptr*len(x))(*[row.ctypes.data_as(i4ptr) for row in x])
    print D
    
    '''
    
    

    #dealing with structures
    
    dat = [[1126877361,'sunny'], [1126877371,'rain'], [1126877385,'damn nasty'], [1126877387,'sunny']]

    dat_dtype = sp.dtype([('timestamp','i4'),('desc','|S12')])
    arr = sp.rec.fromrecords(dat,dtype=dat_dtype)
    
    lib.print_weather.restype = None
    lib.print_weather.argtypes = [sp.ctypeslib.ndpointer(dat_dtype, flags='aligned, contiguous'), ct.c_int]
   
    lib.print_weather(arr, arr.size)
    print arr
    


    '''
    void foo(float** data, int len) {
        float** x = data;
        for (int i = 0; i < len; i++, x++) {
            /* do something with *x */
        }
    }

    You can create the necessary structure from an existing 2-D NumPy array using the following code:

    Toggle line numbers

    x = N.array([[10,20,30], [40,50,60], [80,90,100]], 'f4')
    f4ptr = POINTER(c_float)
    data = (f4ptr*len(x))(*[row.ctypes.data_as(f4ptr) for row in x])

    '''
    
    #dealing with function pointers
    '''
    allocated_arrays = []
    def allocate(dim, shape):
        print 'allocate called'
        x = N.zeros(shape[:dim], 'f4')
        allocated_arrays.append(x)
        ptr = x.ctypes.data_as(ct.c_void_p).value
        print hex(ptr)
        print 'allocate returning'
        return ptr
    
    
    ALLOCATOR = ct.CFUNCTYPE(ct.c_long, ct.c_int, ct.POINTER(ct.c_int))
    lib.foo.argtypes = [ALLOCATOR]
    lib.foo.restype = ct.c_void_p

    print 'calling foo'
    lib.foo(ALLOCATOR(allocate))
    print 'foo returned'

    print allocated_arrays[0]
    '''
    
    return

if __name__ == "__main__":

    try:
        x = x + 1
    except:
        print exceptinfo()

    '''    
    try:
        x=x+1
    except:
        print sys.exc_type #non thread_safe
        print sys.exc_value #non thread_safe
        print sys.exc_info() #thread_safe
    '''    
