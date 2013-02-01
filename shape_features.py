import numpy as np

from dipy.core.geometry import vec2vec_rotmat
from dipy.tracking.metrics import frenet_serret


def vec2vec_proj(u, v):
    return np.dot(v, u) * v / np.linalg.norm(v)


def vec2plane(u, v, w):
    """ http://www.cs.oberlin.edu/~bob/cs357.08/VectorGeometry/VectorGeometry.pdf
    """
    n = np.cross(v, w)
    return u - np.dot(u, n) * n / np.linalg.norm(n)


def rotation_matrix(axis, theta_degree):
    """ Create rotation matrix for angle theta along axis

    Parameters
    ----------
    axis : array, shape (3, )
            rotation axis
    theta_degree : float,
            rotation angle in degrees

    Returns
    -------
    mat : array, shape (3, 3)

    """
    theta = 1. * theta_degree * np.pi / 180.
    axis = 1. * axis / np.sqrt(np.dot(axis, axis))
    a = np.cos(theta / 2)
    b, c, d = - axis * np.sin(theta / 2)
    mat = np.array([[a * a + b * b - c * c - d * d, 2 * (b * c - a * d), 2 * (b * d + a * c)],
                    [2 * (b * c + a * d), a * a + c * c - b * b - d * d, 2 * (c * d - a * b)],
                    [2 * (b * d - a * c), 2 * (c * d + a * b), a * a + d * d - b * b - c * c]])
    return mat




def invariant_angles(S):

    the_phi = []

    for i in range(2, len(S)):

        u = S[i] - S[i - 1]
        u = u / np.linalg.norm(u)

        v = S[i - 1] - S[i - 2]
        v = v / np.linalg.norm(v)

        c = np.cross(u, v)

        theta = np.arccos(np.dot(u, v)) * 180 / np.pi

        if i == 2:
            phi = 0
        else:
            phi = np.arccos(np.dot(c_prev, c)) * 180 / np.pi

        # print theta
        # print phi
        the_phi.append((theta, phi))
        c_prev = c

    return np.array(the_phi)

#dummy
sq = np.sqrt(2.) / 2.

S = np.array([[0, 0, 0],
              [sq, sq, 0],
              [sq + 1, sq, 0],
              [2 * sq + 1, 0, 0]])

S = S * 3

print 'Dummy standard'
print invariant_angles(S)

Sinit = S.copy()

Rot = rotation_matrix(np.array([0, 1, 0]), 30)

S = np.dot(Rot, S.T).T + np.array([1, 0, 0])
print 'Dummy translated'
print invariant_angles(S)

from dipy.viz import fvtk
r = fvtk.ren()
fvtk.add(r, fvtk.line(Sinit, fvtk.red))
fvtk.add(r, fvtk.line(S, fvtk.green))
fvtk.add(r, fvtk.axes())
fvtk.show(r)

# helix
theta = 2 * np.pi * np.linspace(0, 2, 100)
x = np.cos(theta)
y = np.sin(theta)
z = theta / (2 * np.pi)
Shel = np.vstack((x, y, z)).T

from dipy.tracking.metrics import downsample
Shel = downsample(Shel, 12)

print 'Helix standard'
print invariant_angles(Shel)
Shel2 = np.dot(Rot, Shel.T).T

print 'Helix translated'
print invariant_angles(Shel2)

