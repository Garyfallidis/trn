import numpy as np

import regreg.api as R
import sphere_tools

import matplotlib.pyplot as plt  
from dipy.viz import fvtk
import dipy.data
from dipy.core.geometry import sphere2cart
import nipy
from dipy.data import get_data

import nibabel as nib
from dipy.utils.spheremakers import sphere_vf_from

from cvxopt import matrix
from cvxopt import solvers

parameters={'101_32':['/home/ian/Data/PROC_MR10032/101_32',
                    '1312211075232351192010121313490254679236085ep2dadvdiffDSI10125x25x25STs004a001','.bval','.bvec','.nii'],
            '118_32':['/home/ian/Data/PROC_MR10032/118_32',\
                      '131221107523235119201012131413348979887031ep2dadvdiffDTI25x25x25STEAMs011a001','.bval','.bvec','.nii'],
            '64_32':['/home/ian/Data/PROC_MR10032/64_32',\
                     '1312211075232351192010121314035338138564502CBUDTI64InLea2x2x2s005a001','.bval','.bvec','.nii']}
       

def get_data(name='64_32'):    
    bvals=np.loadtxt(parameters[name][0]+'/'+parameters[name][1]+parameters[name][2])
    bvecs=np.loadtxt(parameters[name][0]+'/'+parameters[name][1]+parameters[name][3]).T
    img=nib.load(parameters[name][0]+'/'+parameters[name][1]+parameters[name][4])     
    return img.get_data(),bvals,bvecs


def signal_1_stick(S0,bvals,d,orientation,gradients):
    return S0*np.array([np.exp(-bvals[i]*d*np.dot(orientation,gradient)**2) \
                      for i,gradient in enumerate(gradients)])

def matrix_rank(M):
   S = np.linalg.svd(M, compute_uv=False)
   tol = S.max() * np.finfo(S.dtype).eps
   return np.sum(S > tol)

#siem64 =  nipy.load_image('/home/ian/Devel/dipy/dipy/core/tests/data/small_64D.gradients.npy')

#data102,affine102,bvals102,dsi102=dcm.read_mosaic_dir('/home/ian/Data/Frank_Eleftherios/frank/20100511_m030y_cbu100624/08_ep2d_advdiff_101dir_DSI')

#bvals102=bvals102.real
#dsi102=dsi102.real

#v362,f362 = sphere_vf_from('symmetric362')
#print v362.shape()

#v642,f642 = sphere_vf_from('symmetric642')
def qpfit1(design, S0,bvals,d,test_direction,gradients,lambd):
    Y = signal_1_stick(S0,bvals[1:],d,test_direction,gradients)
    n, m = design.shape
    augmented_gradients = np.vstack((gradients,np.zeros(3)))
    distances = np.dot(augmented_gradients,augmented_gradients.T)**2
    distances[:,m-1]=0.
    distances[m-1,:]=0.
    sum_cost = lambd
    sim_cost = 0.
    P = matrix(np.dot(design.T,design)-sim_cost*distances)
    penalty = sum_cost*np.ones(m)-np.dot(design.T,Y)
    penalty[m-1]=0
    q = matrix(penalty)
    G = matrix(-np.eye(m))
    h = matrix(np.zeros(m))
    A = None
    b = None
    initvals=None
    solvers.options['show_progress'] = False
    solution = solvers.qp(P, q, G, h, A, b, initvals)
    x = np.array(solution['x']).flatten()
    fitted = np.dot(design,x)
    residuals = Y-fitted
    rss = np.sum(residuals**2)
    maxresid = np.max(np.abs(residuals))
    xi = np.argsort(-x[:-1])
    top_inds = xi[x[xi]>0.05]
    top_coeffs = x[top_inds]
    top_grads = augmented_gradients[top_inds,:]
    nearness = np.dot(top_grads,test_direction)**2    
    test_distances = np.dot(gradients,test_direction)**2
    nearest = np.argmax(test_distances)
    #print 'top_gradient indices', top_inds
    #print 'top coefficients', top_coeffs
    #print 'nearness', nearness
    #print 'nearest', nearest
    return {'top': list(top_inds), 'nearest': nearest, 'actual': test_direction, 'rss': rss, 'maxresid': maxresid}

if __name__ == '__main__':

    #data, bvals, bvecs = get_data('64_32')
    # Load data from the dipy standard data
    
    data, bvals, bvecs = dipy.data.get_data('small_64D')
    bvals64 = np.load(bvals)
    bvecs64 = np.load(bvecs)
    
    data, bvals, bvecs = dipy.data.get_data('small_101D')
    bvals101 = np.loadtxt(bvals)
    bvecs101 = np.loadtxt(bvecs).T
    
    d = 0.0015
    S0 = 100
    b = 1200

    # Non zero gradients (first gradient is 0)
    gradients = bvecs64[1:,:]

    # model fibers in orientation of gradients
    orientations = bvecs101[1:,:]
    templates = [signal_1_stick(S0,bvals64[1:],d,orientation,gradients)
                 for i, orientation in enumerate(orientations)]
    # add isotropic component as last column
    templates += [np.ones(gradients.shape[0])]

    design = np.array(templates).T

    # Make pure fiber data to test against
    #test_direction = np.array([1,2,3])/np.sqrt(14.)

    #print np.sum(test_direction**2)
    #unit vector for test signal direction

    #test_direction = gradients[15,:]
    test_directions = sphere_tools.random_uniform_on_sphere(10**2)
    sparsity = []
    lambd = 2.**18.1
    for test_direction in test_directions:
        sparsity += [qpfit1(design, S0,bvals,d,test_direction,gradients,lambd)]

    angles =[np.abs(np.dot(s['actual'],gradients[s['top']].T)) for s in sparsity] 
    max_angles =[np.max(np.abs(np.dot(s['actual'],gradients[s['top']].T))) for s in sparsity]
    lengths = np.bincount([len(s['top']) for s in sparsity])
    
    '''
    Y = signal_1_stick(S0,bvals[1:],d,test_direction,gradients)

    # Set up model according to Jonathan's recipe
    n, m = design.shape
    '''
    '''
    # Select up to (not including) last column to penalize
    #penalized_coefs = R.selector(slice(0, m-1), m)
    weights = np.ones(m)
    weights[-1] = 0
    penalty = R.nonnegative.linear(m, lagrange=5, linear_term=weights)
    #penalty = R.constrained_positive_part.linear(penalized_coefs, lagrange=5)
    # 'coef' becomes 'legrange' in Brad's branch
    loss = R.l2normsq.affine(design,-Y, coef=0.5)
    problem = R.container(loss, penalty)
    solver = R.FISTA(problem.composite())
    '''
    '''
    augmented_gradients = np.vstack((gradients,np.zeros(3)))
    distances = np.dot(augmented_gradients,augmented_gradients.T)**2
    distances[:,m-1]=0.
    distances[m-1,:]=0.
    sum_cost = 1000.
    sim_cost = 0.
    #sim_cost = 1.1*10**-1.9
    #sim_cost = -1000.
    #sim_cost = 0.
    from cvxopt import matrix
    from cvxopt import solvers
    P = matrix(np.dot(design.T,design)-sim_cost*distances)
    print matrix_rank(P)
    penalty = sum_cost*np.ones(m)-np.dot(design.T,Y)
    penalty[m-1]=0
    q = matrix(penalty)
    G = matrix(-np.eye(m))
    h = matrix(np.zeros(m))
    '''
    '''
    A = matrix(np.zeros((1,m)))
    b = matrix(np.zeros(1))
    '''
    '''
    A = None
    b = None
    print matrix_rank(np.hstack((P,G)))
    initvals=None
    solvers.options['show_progress'] = False
    solution = solvers.qp(P, q, G, h, A, b, initvals)
    x = np.array(solution['x']).flatten()
    xi = np.argsort(x)[-1:0:-1]
    top_inds = xi[x[xi]>0.05]
    top_coeffs = x[top_inds] 
    top_grads = augmented_gradients[top_inds,:]
    nearness = np.dot(top_grads,test_direction)**2
    
    test_distances = np.dot(gradients,test_direction)**2
    nearest = np.max(test_distances)
    
    print 'top_gradient indices', top_inds
    print 'top coefficients', top_coeffs
    print 'nearness', nearness
    print 'nearest', nearest
    '''
'''
X = np.random.standard_normal((64,200))
X_s = np.random.standard_normal((64,1))
design = np.hstack([X, X_s])
Y = np.random.standard_normal(64)

penalized_coefs = R.selector(slice(0,64), design.shape[1:])
penalty = R.constrained_positive_part.linear(penalized_coefs, lagrange=5)

loss = R.l2normsq.affine(design,-Y, coef=0.5)
problem = R.container(loss, penalty)
solver = R.FISTA(problem.composite())

solver.fit()
coefs = solver.composite.coefs
'''

