""" Negotiating with regreg """
import numpy as np

import dipy.data
import regreg.api as rr


def signal_1_stick(S0,bvals,d,orientation,gradients):
    return S0*np.array([np.exp(-bvals[i]*d*np.dot(orientation,gradient)**2) \
                      for i,gradient in enumerate(gradients)])


if __name__ == '__main__':
    # Load data from the dipy standard data
    data, bvals, bvecs = dipy.data.get_data('small_64D')
    bvals = np.load(bvals)
    bvecs = np.load(bvecs)

    d = 0.0015
    S0 = 100
    b = 1200

    # Non zero gradients (first gradient is 0)
    gradients = bvecs[1:,:]

    # model fibers in orientation of gradients
    orientations = bvecs[1:,:]
    templates = [signal_1_stick(S0,bvals[1:],d,orientation,gradients)
                 for i, orientation in enumerate(orientations)]
    # add isotropic component as last column
    templates += [np.zeros(gradients.shape[0])]
    design = np.array(templates).T
    # mean center fibers
    X = design - design.mean(axis=0)
    # X = design
    X[:,-1] = 1
    # Make pure fiber data to test against
    test_direction = np.array([1,2,3])/np.sqrt(14.)
    #print np.sum(test_direction**2)
    #unit vector for test signal direction
    Y = signal_1_stick(S0,bvals[1:],d,test_direction,gradients)
    # Y = design[:,3]

    # Set up model according to Jonathan's recipe
    N, P = X.shape
    # 'coef' becomes 'lagrange' in Brad's branch
    loss = rr.l2normsq.affine(X,-Y, coef=0.5)
    # Select up to (not including) last column to penalize
    lagrange = 500
    weights = np.ones(P) * lagrange
    weights[-1] = 0
    penalty = rr.nonnegative(P, lagrange=1, linear_term=weights)
    # Neighborhood weightings
    coses2 = orientations.dot(orientations.T) ** 2
    nearests = np.argmax(coses2 - np.eye(P-1), axis=1)
    edges = {}
    for i, nn in enumerate(nearests):
        if nn > i:
            tup = (i, nn)
        else:
            tup = (nn, i)
        if tup not in edges:
            edges[tup] = 1
    rows, cols = zip(*edges.keys())
    n_edges = len(rows)
    angle_D = np.zeros((n_edges, P))
    angle_D[range(n_edges), rows] = 1
    angle_D[range(n_edges), cols] = -1
    angle_penalty = rr.l1norm.linear(angle_D, lagrange=500)
    # Set starting estimate
    initial = np.zeros(P)
    initial[-1] = 60
    # composite_form = rr.composite(loss.smooth_objective,
    #                              penalty.nonsmooth_objective,
    #                              penalty.proximal,
    #                              initial,
    #                             )
    problem = rr.container(loss, penalty, angle_penalty)
    solver = rr.FISTA(problem.composite())
    # solver = rr.FISTA(composite_form)
    solver.fit()
    coefs = solver.composite.coefs

