#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Author: Eleftherios Garyfallidis
Description: Python library for Research and Development in Artificial Intelligence aka Art.

At the moment we are using only mdp, pymvpa, Pycluster and numpy/scipy for array manipulation.  

Other possible software that we should use is pybrain, scikits, mlpy, pyml, shogun, pycgal(comp. geom.), elefant, orange, weka.
'''
try:
    import mdp
except:
    print('MDP is not installed')

try:
    import mvpa
except:    
    print('MVPA is not installed.')

try:
    import Pycluster as pcl    
except:
    
    print('Pycluster is not installed look http://bonsai.ims.u-tokyo.ac.jp/~mdehoon/software/cluster/software.htm')

try:
    import openopt as oo
    import FuncDesigner as fd
    import DerApproximator as da
except ImportError:
    print('openopt or funcdesigner or derapproximator are not installed')
    print('Go to http://openopt.org')

try:    
    import scipy as sp
except:
    print('Scipy is not installed')


def mdptest():

  print('MDP Testing')

  #mdp.test()

def mvpatest():
    
    from mvpa.datasets import Dataset
    from mvpa.datasets.masked import MaskedDataset
    
    #2d only
    data=Dataset(samples=sp.randn(10,5),labels=1)
    
    print('labels',data.labels)
    print('chunks',data.chunks)
    print('samples',data.samples)
    
    #nd
    mdata = MaskedDataset(samples=sp.random.normal(size=(5,3,4)),labels=[1,2,3,4,5])
    print('labels',mdata.labels)
    print('nfeatures',mdata.nfeatures)
    print('samples',mdata.samples) 
    print('mapforward',mdata.mapForward(sp.arange(12).reshape(3,4)))
    print('mapreverse',mdata.mapReverse(sp.arange(mdata.nfeatures)))

def automaticdiffertest():
    
    #from FuncDesigner import *

    a, b, c = fd.oovars('a', 'b', 'c')

    f1, f2 = fd.sin(a) + fd.cos(b) - fd.log2(c) + fd.sqrt(b), fd.sum(c) + c * fd.cosh(b) / fd.arctan(a) + c[0] * c[1] + c[-1] / (a * c.size)

    f3 = f1*f2 + 2*a + fd.sin(b) * (1+2*c.size + 3*f2.size)

    f = 2*a*b*c + f1*f2 + f3 + fd.dot(a+c, b+c)

    point = {a:1, b:2, c:[3, 4, 5]} # however, you'd better use numpy arrays instead of Python lists

    print(f(point))

    print(f.D(point))

    print(f.D(point, a))

    print(f.D(point, [b]))

    print(f.D(point, fixedVars = [a, c]))


    

def kdtree():
    #use scipy.spatial
    #and read info here http://folk.uio.no/sturlamo/python/multiprocessing-tutorial.pdf
    #and http://www.scipy.org/Cookbook/KDTree
    #read also the wikipedia article
    #we could use a kdtree or the birch algorithm to initialize k-means
    
    pass

def pyclustertest():
    
    data=sp.rand(100,4)
    cid,e,n=pcl.kcluster(data)
    centroids,cmask=pcl.clustercentroids(D,clusterid=cid)
    
    print data    
    print centroids
    
def fastpdexample():
    '''
    g++ maxflow.cpp graph.cpp Fast_PD.cpp -o Fast_PD.so -shared -fPIC
    
    '''
    
    import ctypes as ct
    import scipy as sp
    import form
   
    #name and directory of the shared lib
    lib = sp.ctypeslib.load_library('Fast_PD.so', '/home/eg01/Devel/Discrete_Optimization/FastPD/FastPD_ElfVersion')
    
    lib.test.argtypes = [ct.c_void_p]
                                            
    lib.test.restype= ct.c_void_p
       
    lib.test(None)
    
    
def birchexample():
    
    '''
    Vassilis Implementation +
    http://xin.cz3.nus.edu.sg/group/personal/cx/modules/DM/birch.ppt
    http://people.cs.ubc.ca/~rap/teaching/504/2005/slides/Birch.pdf
    '''
    
    #void birch_py(double *data,int mrows, int DIMENSION, int K_, int B_, int L_, double wp, double wd, double T_, double *codeBook, double *codeIndices)
    '''
    compile with gcc -o birch.so -shared birch.c -fPIC
    
    run: [codeBook, codeIndices] = birch(reflectionCoefficients, numCodeVectors, B, L, wp, wd, T); */
    B, L: design parameters for the data structure give B = 10, L = 10 */
    wp, wd: mixing parameters give wp = 5, wd = 0.2 */
    T: threshold give 0 or a small value specific to the problem */ 
    '''    
    
    import ctypes as ct
    import scipy as sp
    import form
   
    #name and directory of the shared lib
    lib = sp.ctypeslib.load_library('birch.so', '/home/eg01/Devel/BirchLGB/birch_py')
    
    #A=100*sp.rand(200,3)
    
    R=form.loadmat('ref.mat')
    A=R['reflectionCoefficients']
    
    
    rows=A.shape[0]
    cols=A.shape[1]
    
    A=sp.arange(rows*cols).reshape((rows,cols))
    print A
    #return

    A=sp.asarray(A,dtype=sp.double)
    
    Mat=sp.empty_like(A)

    lib.birch_py.argtypes = [sp.ctypeslib.ndpointer(dtype = sp.double),
                                            ct.c_int, ct.c_int, ct.c_int, ct.c_int, ct.c_int,
                                            ct.c_double,ct.c_double,ct.c_double,
                                            sp.ctypeslib.ndpointer(dtype = sp.double),
                                            sp.ctypeslib.ndpointer(dtype = sp.intc)]
                                            
    lib.birch_py.restype= ct.c_void_p
    
    #K,B,L,wp,wd,T=3,10,10,5,0.2,0
    K,B,L,wp,wd,T=128,10,10,5,0.2,0.01
    
    codeBook=sp.zeros(cols*K)
    codeIndices=sp.ones(rows)
    
    codeBook=sp.asarray(codeBook,dtype=sp.double)    
    codeIndices=sp.asarray(codeIndices,dtype=sp.intc)
    
    #lib.birch_py(A,int(rows),int(cols),int(K),int(B),int(L),wp,wd,T,codeBook,codeIndices)
    lib.birch_py(A,int(rows),int(cols),int(K),int(B),int(L),float(wp),float(wd),float(T),codeBook,codeIndices)
    
    #print 'A:',A
    print 'codeBook',codeBook.reshape(K,cols)
    print 'codeIndices',codeIndices

    
def graphexample():
    '''
    We have this simple graph
    A -> B,    A -> C,    B -> C,    B -> D,    C -> D,    D -> C,    E -> F,    F -> C
    We are going to use dict to create this graph
    '''
    graph={'A':['B','C'],'B':['C','D'],'C':['D'],'D':['C'],'E':['F'],'F':['C']}
    
    def find_path(graph,start,end, path=[]):
        
        path=path+[start]
        if start == end:
            return path
        if not graph.has_key(start):
            return None
        
        for node in graph[start]:
            if node not in path:
                newpath=find_path(graph,node,end,path)
                if newpath: 
                    return newpath
                
        return None
    
    def find_all_paths(graph, start, end, path=[]):
        path = path + [start]
        if start == end:
            return [path]
        if not graph.has_key(start):
            return []
        paths = []
        for node in graph[start]:
            if node not in path:
                newpaths = find_all_paths(graph, node, end, path)
                for newpath in newpaths:
                    paths.append(newpath)
        return paths

    def find_shortest_path(graph, start, end, path=[]):
        path = path + [start]
        if start == end:
            return path
        if not graph.has_key(start):
            return None
        shortest = None
        for node in graph[start]:
            if node not in path:
                newpath = find_shortest_path(graph, node, end, path)
                if newpath:
                    if not shortest or len(newpath) < len(shortest):
                        shortest = newpath
        return shortest


    print 'Entire graph'
    print graph

    print 'All root nodes'
    for node in graph:
        print node
    
    print 'Nodes under A'
    for node in graph['A']:
        print node
    
    
    print 'Find Path A->D',find_path(graph,'A','D')
    print 'Find Path A->F',find_path(graph,'A','F')
    print 'Find all Paths A->D',find_all_paths(graph,'A','D')
    print 'Find shortest Path A->D',find_shortest_path(graph, 'A', 'D')

    print 'Some tests'
    testg={'A':['B',3,'C',4], 'B':['C',3]}
    
    print testg
    print testg['A']

def networkxexample():
    #Example is from
    #http://networkx.lanl.gov/tutorial/tutorial.html
    import networkx as nx

    G=nx.Graph()
    print(G)
    
    G.add_node(1)
    print(G)
    G.is_directed()
    G.add_nodes_from([2,3])
    G.add_nodes_from(nx.path_graph(10))
    G.add_edge(1,2)
    e=(2,3)
    G.add_edge(*e)
    G.add_edges_from(nx.path_graph(10).edges())
    G.number_of_nodes()
    G.number_of_edges()
    G.nodes()
    G.edges()
    G.neighbors(1)
    H=nx.DiGraph(G)
    H.edges()
    H2=nx.Graph({0:[1,2,3], 1:[0,3], 2:[0], 3:[0]})
    H3=nx.Graph()
    H3.add_edge(1,2,color='red')
    H3.add_edges_from([(1,3,{'color':'blue'}), (2,0,{'color':'red'}), (0,3)])
    H.edges()
    H3.nodes()
    H3.edges(data=True)    
    H3.add_edge(0,3,color='green')
    H3.edges()
    H3.edges(data=True)
    H3.nodes()
    H3[0]
    H3[0][2]
    H3[0][3]
    H3.add_edge(1,3)
    H3[1][3]['color']='blue'
    FG=nx.Graph()
    FG.add_weighted_edges_from([(1,2,0.125),(1,3,0.75),(2,4,1.2),(3,4,0.375)])

    #Fast examination of all edges

    for n, nbrs in FG.adjacency_iter():
        for nbr, eattr in nbrs.iteritems():
            data=eattr['weight']
            if data < 0.5: print (n,nbr, data)
       

    FG.adjacency_iter()
    FG.adjacency_iter().next()
    FG
    FG.nodes()
    FG.edges()
    FG.adjacency_iter().next()

    FG[3][4]
    FG[3][4]['weight']
    DG=nx.DiGraph()
    DG.add_weighted_edges_from([(1,2,0.5), (3,1,0.75)])
    DG.out_degree(1,weighted=True)
    DG.degree(1,weighted=True)
    DG.edges()
    DG.nodes()
    DG.successors(1)
    DG.neighbors(1)
    DG.neighbors(3)
    UG=G.to_undirected()
    UG2=nx.Graph(G)
    
    MG=nx.MultiGraph()
    MG.add_weighted_edges_from([(1,2,.5), (1,2,.75), (2,3,.5)])
    MG.degree(weighted=True, with_labels=True)
    GG=nx.Graph()
    for n,nbrs in MG.adjacency_iter():
        for nbr,edict in nbrs.iteritems():
            
            minvalue=min(edict.values())
            GG.add_edge(n,nbr,minvalue)
            

    nx.shortest_path(GG,1,3)
    nx.connected_components(G)
    sorted(nx.degree(G))
    G.nodes()
    nx.clustering(G)

    import matplotlib.pyplot as plt

    nx.draw(G)
    plt.show()
    nx.draw_random(G)
    nx.draw_circular(G)
    nx.draw_spectral(G)
    nx.draw_spring(G)
    nx.draw_spring(G)
    plt.savefig('path.png')



if __name__ == "__main__":    

    #mdptest()
    #mvpatest()
    #pyclustertest()
    #graphexample()
    #networkxexample()
    automaticdiffertest()
    
