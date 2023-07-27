import numpy as np
import scipy.sparse as sp
from psbody.mesh import Mesh
from psbody.mesh.topology.connectivity import get_vert_connectivity
from sklearn.preprocessing import normalize
from chumpy import Ch
import chumpy as ch
import cv2

EPS = 10e-9

def distance_computation(verts, mesh):

    tree = mesh.compute_aabb_tree()
    closest_points = tree.nearest(verts)[1]

    dists = ch.sqrt(ch.sum((closest_points - verts) ** 2, axis=1))
    assert (len(dists) == len(closest_points))

    return dists

class Rodrigues(Ch):
    """
    Chumpy class for Rodrigues.
    """

    dterms = ['rt']

    def compute_r(self):
        return cv2.Rodrigues(self.rt.r)[0]

    def compute_dr_wrt(self, wrt):
        if wrt is self.rt:
            return cv2.Rodrigues(self.rt.r)[1].T


class PlaneFrame:
    """Class to store the frame of refrence attached to a plane"""
    def __init__(self, center=None, center_idx=None, frame=None, rms=0, point_dist=None):
        self.center = center
        self.center_idx = center_idx # plane center index in mesh
        self.frame = frame
        self.rms = rms
        self.point_dist = point_dist


def rotate(v, angles):
    '''
    This function rotates the vertices along each axis step by step
    :param v: 2D numpy array of size (nx3), where m is number of vertices and n is 3. "v" represents the set of 3D vertices.
    :param angles: list of 3 angles(x,y,z) in radians
    :return: rotated set of vertices
    '''

    for idx, angle in enumerate(angles):
        # theta is in radians
        theta = [0.] * 3
        theta[idx] = angle
        rodr_matrix, _ = cv2.Rodrigues(np.array(theta))
        v = v.dot(rodr_matrix)

    return v


def rigid_alignment_analytic(current_p, target_p):
    """
    function [aligned, R, T] = rigid_alignment (current_p, target_p)
    Find rotation and translation (in the least squares sense), such that
        min_(R,T) | pm_i - current_p * R + T |^2

    An implementation of
     Least-Squares Fitting of Two 3-D Point Sets
     K.S. Arun, T.S. Huang and S.D. Blostein
     PAMI-9, No 5, 1987
    """

    # Step 1:
    p = np.mean(current_p, axis=0)
    pm = np.mean(target_p, axis=0)
    q_i = current_p - p
    qm_i = target_p - pm

    # Step 2:
    H = np.dot(q_i.T, qm_i)

    # Step 3:
    U, S, V = np.linalg.svd(H)

    # Step 4:
    R = np.dot(U, V)
    T = (pm - np.dot(R.T, p)).reshape((1, 3))

    unposed = np.dot(target_p - T, R.T)

    return unposed, R, T


def rigid_alignment_analytic_weighted(current_p, target_p, weight_vector):
        """
        function [aligned, R, T] = rigid_alignment (current_p, target_p)
        Find rotation and translation (in the least squares sense), such that
            min_(R,T) | pm_i - current_p * R + T |^2

        An implementation of
         Least-Squares Fitting of Two 3-D Point Sets
         K.S. Arun, T.S. Huang and S.D. Blostein
         PAMI-9, No 5, 1987

         https://igl.ethz.ch/projects/ARAP/svd_rot.pdf
        """

        assert weight_vector.shape[0] == target_p.shape[0]
        assert current_p.shape[0] == target_p.shape[0]

        # normalize the weight vector
        weight = weight_vector * 1.0 / np.sum(weight_vector)

        # Step 1:
        p_m = np.sum(current_p * weight[:, np.newaxis], axis=0)
        q_m = np.sum(target_p * weight[:, np.newaxis], axis=0)
        p = current_p - p_m
        q = target_p - q_m

        # Step 2:
        W = np.diag(weight)
        H = p.T.dot(W).dot(q)

        # Step 3:
        U, S, V = np.linalg.svd(H)

        # Step 4:
        D = np.diag(np.ones(3))
        D[-1, -1] = np.linalg.det(V.T.dot(U.T))

        R = V.T.dot(D).dot(U.T)
        T = (q_m - np.dot(p_m, R.T)).reshape((1, 3))

        unposed_verts = np.dot(current_p, R.T) + T

        return unposed_verts, R, T


def fit_plane(point_cloud):
    """ For a point cloud nx3, approximate a plane with least mean squares.
    Return:
     center
     normal
     frobenius distance from the point cloud to the plane
     absolute dist for each point to the plane nx3
     """

    center = np.mean(point_cloud, axis=0)
    X = point_cloud - center

    U, S, V = np.linalg.svd(X, full_matrices=False)

    i = V[0,:]
    j = V[1,:]
    n = V[2,:]

    point_dist = np.abs(np.matmul(X, np.array(n)))
    global_rms_dist = np.linalg.norm(point_dist) #frobenius distance

    plane_frame = PlaneFrame(center=center, frame=[i,j,n], rms=global_rms_dist, point_dist=point_dist)

    return plane_frame


def mesh_closest_point(p, mesh):
    index = np.argmin(np.linalg.norm(mesh.v-p, axis=1))
    coord = mesh.v[index]
    return index, coord


def mesh_normal(point_index, mesh):
    # find a triangle containing this point index
    f = mesh.f[np.where(mesh.f == point_index)[0][0]]
    assert(len(f)>0)

    v1 = mesh.v[f[0]] - mesh.v[f[2]]
    v2 = mesh.v[f[1]] - mesh.v[f[2]]
    n = ch.cross(v1,v2)
    n = n*1./ch.linalg.norm(n)
    return n


def triangle_normal(face, verts):
    v1 = verts[face[0]] - verts[face[2]]
    v2 = verts[face[1]] - verts[face[2]]
    n = ch.cross(v1,v2)
    n_normed = n*1./ch.linalg.norm(n)
    return n_normed


def is_normed(np_vect):
    return (np.linalg.norm(np_vect)-1)<EPS

    
def laplacian(part_mesh):
    """ Compute laplacian operator on part_mesh. This can be cached.
    """

    connectivity = get_vert_connectivity(part_mesh)
    # connectivity is a sparse matrix, and np.clip can not applied directly on
    # a sparse matrix.
    connectivity.data = np.clip(connectivity.data, 0, 1)
    lap = normalize(connectivity, norm='l1', axis=1)
    lap = sp.eye(connectivity.shape[0]) - lap

    return lap

def get_submesh(verts, faces, verts_retained=None, faces_retained=None, min_vert_in_face=2):
    '''
        Given a mesh, create a (smaller) submesh
        indicate faces or verts to retain as indices or boolean

        @return new_verts: the new array of 3D vertices
                new_faces: the new array of faces
                bool_faces: the faces indices wrt the input mesh
                vetex_ids: the vertex_ids wrt the input mesh
        '''

    if verts_retained is not None:
        # Transform indices into bool array
        if verts_retained.dtype != 'bool':
            vert_mask = np.zeros(len(verts), dtype=bool)
            vert_mask[verts_retained] = True
        else:
            vert_mask = verts_retained

        # Faces with at least min_vert_in_face vertices
        bool_faces = np.sum(vert_mask[faces.ravel()].reshape(-1, 3), axis=1) > min_vert_in_face

    elif faces_retained is not None:
        # Transform indices into bool array
        if faces_retained.dtype != 'bool':
            bool_faces = np.zeros(len(faces_retained), dtype=bool)
        else:
            bool_faces = faces_retained

    new_faces = faces[bool_faces]
    # just in case additional vertices are added
    vertex_ids = list(set(new_faces.ravel()))

    oldtonew = -1 * np.ones([len(verts)])
    oldtonew[vertex_ids] = range(0, len(vertex_ids))

    new_verts = verts[vertex_ids]
    new_faces = oldtonew[new_faces].astype('int32')

    return (new_verts, new_faces, bool_faces, vertex_ids)



def move_along_smoothed_normal(v, f, n, displacement, smoothing_iteration=100):
    """
    Move vertices along their normal on a distance equal to displacement
    @param v: vertices
    @param f: faces
    @param n: normals
    @param displacement:
    @param smoothing_iteration: how many smooth iterations on the normal
    @return: moved vertices
    """

    lap = laplacian(Mesh(v=v, f=f))

    for i in range(smoothing_iteration):
        n = n - lap * n

    v = v + displacement * n
    return v


def symmetrize_mesh(mesh):
    """ Symmetrize a mesh by flipping it along the z axis and flipping the faces"""
    mesh.v[:, 2] = -mesh.v[:, 2]
    mesh.flip_faces()
    return mesh
