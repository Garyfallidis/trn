# sphere_tools - for tools to help with stuff on spheres

import numpy as np
import dipy.core.geometry

def random_uniform_on_sphere(n=1,coords='xyz'):
    u = np.random.normal(0,1,(n,3))
    u = u/np.sqrt(np.sum(u**2,axis=1)).reshape(n,1)
    if coords=='xyz':
        return u
    else:
        angles = np.zeros((n,2))
        for (i,xyz) in enumerate(u):
            angles[i,:]=dipy.core.geometry.cart2sphere(*xyz)[1:]
        if coords=='radians':
            return angles
        else:
            return (180./np.pi)*angles