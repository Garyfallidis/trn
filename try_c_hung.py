import sys
sys.path += ['build/lib.linux-i686-2.2/']
import hungarian
import numpy as np

'''
'A = [[1,2,3],[4,5,6],[7,8,9]]
#B = [[1,2,3],[4,5,6],[7,8,9],[10,11,12]]
C = [[1,2,3,10],[4,5,6,11],[7,8,9,12]]

h = hungarian.HungarianSolver(A)
print A
print h.print_rating()

#h = hungarian.HungarianSolver(B)

h = hungarian.HungarianSolver(C)
print C
print h.print_rating()
'''

for s in range(10000,10001):
    #a = 100*np.eye(s,dtype='i4')
    a = np.random.randint(0, high=100, size=(s,s))
    aa = [[int(c) for c in b] for b in a]
    #print aa
    h = hungarian.HungarianSolver(aa)
    #print h.print_rating()
    #print a
    