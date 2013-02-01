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
    mat = np.array([[a*a + b*b - c*c - d*d, 2*(b*c - a*d), 2*(b*d + a*c)],
                     [2*(b*c + a*d), a*a + c*c - b*b - d*d, 2*(c*d - a*b)],
                     [2*(b*d - a*c), 2*(c*d + a*b), a*a + d*d - b*b - c*c]])
    return mat


sq = np.sqrt(2.) / 2.

S = np.array([[0, 0, 0],
              [sq, sq, 0],
              [sq + 1, sq, 0],
              [2 * sq + 1, 0, 0]])

S = S * 3

Rot = rotation_matrix(np.array([1, 0, 0]), 45)

S = np.dot(Rot, S.T).T

# T, N, B, k, t = frenet_serret(S)

from dipy.viz import fvtk
r = fvtk.ren()
fvtk.add(r, fvtk.line(S, fvtk.red))
#fvtk.add(r, fvtk.line(S2, fvtk.green))
fvtk.show(r)

t_ext = np.vstack((S[0], S[0] + np.array([1, 0, 0])))
b_ext = np.vstack((S[0], S[0] + np.array([0, 1, 0])))
n_ext = np.vstack((S[0], S[0] + np.array([0, 0, 1])))

fvtk.add(r, fvtk.line(t_ext, fvtk.dark_red))
fvtk.add(r, fvtk.line(b_ext, fvtk.green))
fvtk.add(r, fvtk.line(n_ext, fvtk.blue))


t = (S[1] - S[0]) / np.linalg.norm(S[1] - S[0])
R = vec2vec_rotmat(np.array([1, 0, 0]), t)
b = np.dot(R, np.array([0, 1, 0]))
n = np.dot(R, np.array([0, 0, 1]))

for i in range(2, len(S)): 

	t_ext = np.vstack((S[i-1], S[i-1] + t))
	b_ext = np.vstack((S[i-1], S[i-1] + b))
	n_ext = np.vstack((S[i-1], S[i-1] + n))

	fvtk.add(r, fvtk.line(t_ext, fvtk.dark_red))
	fvtk.add(r, fvtk.line(b_ext, fvtk.green))
	fvtk.add(r, fvtk.line(n_ext, fvtk.blue))

	tnew = (S[i] - S[i-1]) / np.linalg.norm(S[i] - S[i-1])

	print 'theta', np.arccos(np.dot(tnew, n)) * 180 / np.pi
	tnew_proj = vec2plane(tnew, t, b)
	print 'phi', np.arccos(np.dot(tnew_proj, t)) * 180 / np.pi

	#print 'theta', np.arccos(np.dot(tnew, t)) * 180 / np.pi
	#tnew_proj = vec2plane(tnew, b, n)
	#print 'phi', np.arccos(np.dot(tnew_proj, n)) * 180 / np.pi

	R = vec2vec_rotmat(t, tnew)
	b = np.dot(R, b)
	n = np.dot(R, n)
	t = tnew

"""
T = (S[1] - S[0]) / np.linalg.norm(S[1] - S[0])
R = vec2vec_rotmat(np.array([1, 0, 0]), T)
B = np.dot(R, np.array([0, 1, 0]))
N = np.dot(R, np.array([0, 0, 1]))

Tnew = (S[2] - S[1]) / np.linalg.norm(S[2] - S[1])
print 'theta', np.arccos(np.dot(Tnew, T)) * 180 / np.pi

Tnew_proj = vec2plane(Tnew, B, N)
print 'phi', np.arccos(np.dot(Tnew_proj, N)) * 180 / np.pi
"""
