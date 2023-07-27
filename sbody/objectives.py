__all__ = ['landmark_function', 'scan_to_mesh_squared_function',
           'mesh_to_scan_squared_function', 'scan_to_mesh_function',
           'mesh_to_scan_function', 'full_mesh_to_scan_function', 'vecdiff_function']


from scipy.sparse import csc_matrix
from numpy import array
import scipy as sy
import scipy.sparse as sp
import numpy as np
# import body.mesh_distance as mesh_distance
# import mesh_distance_lazy
import random
import copy

from sbody.matlab import *
from psbody.mesh import Mesh
# from body.chained.chained import Chained
# from body.chained.chained import *
# from body.geometry.squared_length import SquaredLength
# from body.geometry.tri_normals import TriNormals


# def sample_mesh_distance_squared(sample_mesh, sample_spec, reference_mesh, reference_mesh_normals=None):
#
#     # Faster
#     result = mesh_distance.SampleMeshDistanceSquared(sample_mesh, sample_spec, reference_mesh)
#
#     # seems that faster code has occasional issues with m2s. We should fix this!
#     if np.isnan(np.float(result.dr_reference_mesh.sum())):
#         print('using slower mesh distance function')
#         # Slower
#         return mesh_distance_lazy.SampleMeshDistanceSquared(sample_mesh, sample_spec, reference_mesh, reference_mesh_normals)
#     return result
#
#
# class Orig_to_Dest(Chained):
#
#     def __init__(self, x, scan, anchor_template, num_samples, sample_spec, s2m, scan_search_tree=None, scan_normals=None):
#
#         self.x = self.differentiable_wrt(x)
#
#         self.scan = scan
#         self.scan_normals = scan_normals
#         self.anchor_template = anchor_template
#         self.sample_spec = sample_spec
#         self.s2m = s2m
#         self.scan_search_tree = scan_search_tree
#
#     def compute_r(self):
#         return self.result.r
#
#     def compute_dr_wrt(self, wrt):
#         return self.result.dr_reference_mesh if self.s2m else self.result.dr_sample_mesh
#
#     @cached_property
#     def result(self):
#
#         deformed_template = Mesh(f=self.anchor_template.f, v=self.x.r.reshape(-1, 3))
#         if self.s2m:
#             return sample_mesh_distance_squared(
#                 self.scan,
#                 self.sample_spec,
#                 deformed_template)
#         else:
#             if self.scan_search_tree is not None:
#                 self.scan.tree = self.scan_search_tree
#
#             return sample_mesh_distance_squared(
#                 deformed_template,
#                 self.sample_spec,
#                 self.scan,
#                 self.scan_normals)
#
#
# def scan_to_mesh_squared_function(scan, anchor_template, num_samples=1e+4, sampling_strategy='uniformly-from-vertices'):
#     """ Returns a function which measures distance from the scan to a deformed template.
#     The integration is over the scan surface, and is approximate (uses sampling)
#
#     Args:
#         scan: a "Mesh" object with the scan
#         anchor_template: a "Mesh" object with the anchor template (or a deformed template,
#                 as we just use the topology)
#     """
#     sample_spec = sample_from_mesh(scan, sampling_strategy, num_samples)
#     return lambda x: Orig_to_Dest(x, scan, anchor_template, num_samples, sample_spec, s2m=True).set_label('Scan to Mesh Distance')
#
#
# def scan_to_mesh_function(scan, anchor_template, num_samples=1e+4, rho=None, sampling_strategy='uniformly-from-vertices', normalized=True):
#     sample_spec = sample_from_mesh(scan, sampling_strategy, num_samples)
#
#     # we take square root (raise to .5) because OrigToDest returns squared distance
#     if rho:
#         rho_used = lambda x: rho(x ** 0.5)
#     else:
#         rho_used = lambda x: x ** 0.5
#
#     if 'point2sample' in sample_spec:
#         num_samples = sample_spec['point2sample'].shape[0] / 3
#
#     if normalized:
#         return lambda x: (rho_used(Orig_to_Dest(x, scan, anchor_template, num_samples, sample_spec, s2m=True)) / (num_samples ** 0.5)).set_label('ScanToMesh')
#     else:
#         return lambda x: (rho_used(Orig_to_Dest(x, scan, anchor_template, num_samples, sample_spec, s2m=True))).set_label('ScanToMesh')
#
#
# def mesh_to_scan_squared_function(scan, anchor_template, num_samples=1e+4, sampling_strategy='uniformly-from-vertices', vertex_indices_to_sample=None):
#     """ Returns a function which measures distance from the deformed template to the scan.
#     The integration is over the template surface, and is approximate (uses sampling)
#
#     Args:
#         scan: a "Mesh" object with the scan
#         anchor_template: a "Mesh" object with the anchor template (or a deformed template,
#                 as we just use the topology)
#     """
#     scan.tree = scan.compute_aabb_tree()
#     scan_normals = TriNormals(scan.v, scan.f)
#     sample_spec = sample_from_mesh(anchor_template, sampling_strategy, num_samples, vertex_indices_to_sample)
#     return lambda x: Orig_to_Dest(x, scan, anchor_template, num_samples, sample_spec, s2m=False, scan_normals=scan_normals).set_label('Mesh to Scan Distance')
#
#
# def mesh_to_scan_function(scan, anchor_template, num_samples=1e+4, rho=None, sampling_strategy='uniformly-from-vertices', normalized=True, vertex_indices_to_sample=None):
#     sample_spec = sample_from_mesh(anchor_template, sampling_strategy, num_samples, vertex_indices_to_sample)
#
#     # we take square root (raise to .5) because OrigToDest returns squared distance
#     if rho:
#         rho_used = lambda x: rho(x ** 0.5)
#     else:
#         rho_used = lambda x: x ** 0.5
#
#     scan.tree = scan.compute_aabb_tree()
#     scan_normals = TriNormals(scan.v, scan.f)
#
#     if 'point2sample' in sample_spec:
#         num_samples = sample_spec['point2sample'].shape[0] / 3
#
#     if normalized:
#         return lambda x: (rho_used(Orig_to_Dest(x, scan, anchor_template, num_samples, sample_spec, s2m=False, scan_normals=scan_normals)) / (num_samples ** 0.5)).set_label('MeshToScan')
#     else:
#         return lambda x: (rho_used(Orig_to_Dest(x, scan, anchor_template, num_samples, sample_spec, s2m=False, scan_normals=scan_normals))).set_label('MeshToScan')
#
#
# def points_to_points_function(point_cloud, rho=None, normalized=True, use_cgal=False, vertex_indices=None):
#     point_cloud.closest_point_tree = point_cloud.compute_closest_point_tree(use_cgal)
#     rho_used = (lambda x: rho(x ** 0.5)) if rho else (lambda x: x ** 0.5)
#     select = (lambda x: x) if (vertex_indices is None) else (lambda x: Select(x, np.vstack([3 * vertex_indices, 3 * vertex_indices + 1, 3 * vertex_indices + 2]).T.flatten()))
#
#     if normalized:
#         return lambda x: (rho_used(SquaredLength((select(x) - Cw(point_cloud.closest_point_tree.nearest_vertices(select(x).r.reshape(-1, 3)).flatten())))) / ((len(select(x).r) / 3) ** 0.5)).set_label('PointsToPoints')
#     else:
#         return lambda x: (rho_used(SquaredLength(select(x) - Cw(point_cloud.closest_point_tree.nearest_vertices(select(x).r.reshape(-1, 3)).flatten())))).set_label('PointsToPoints')
#
#
# def points_to_plane_function(plane_coefficients=np.array([0.0, 1.0, 0.0, -1.0]), normalized=True, rho=None, vertex_indices=None):
#     rho_used = (lambda x: rho(x ** 0.5)) if rho else (lambda x: x ** 0.5)
#     select = (lambda x: x) if (vertex_indices is None) else (lambda x: Select(x, np.vstack([3 * vertex_indices, 3 * vertex_indices + 1, 3 * vertex_indices + 2]).T.flatten()))
#
#     if plane_coefficients is not None:
#         normal_vector = plane_coefficients[:3] / (np.sum(plane_coefficients[:3] ** 2.0) ** 0.5)
#         if normalized:
#             return lambda x: (rho_used((VecsDotVec(select(x), normal_vector) + Cw(plane_coefficients[3])) ** 2.0) / ((len(select(x).r) / 3) ** 0.5)).set_label('PointsToPlane')
#         else:
#             return lambda x: rho_used((VecsDotVec(select(x), normal_vector) + Cw(plane_coefficients[3])) ** 2.0).set_label('PointsToPlane')
#     else:
#         first_three_values = lambda coefficients: Select(coefficients, np.array([0, 1, 2]))
#         last_value = lambda coefficients: Select(coefficients, np.array([3]))
#         normal_vector = lambda plane_coefficients: first_three_values(plane_coefficients) / (SumOf(first_three_values(plane_coefficients) ** 2.0) ** 0.5)
#         if normalized:
#             return lambda x, plane_coefficients: (rho_used((VecsDotVec(select(x), normal_vector(plane_coefficients)) + Cw(last_value(plane_coefficients))) ** 2.0) / ((len(select(x).r) / 3) ** 0.5)).set_label('PointsToPlane')
#         else:
#             return lambda x, plane_coefficients: rho_used((VecsDotVec(select(x), normal_vector(plane_coefficients)) + Cw(last_value(plane_coefficients))) ** 2.0).set_label('PointsToPlane')
#
#
# class GTZero(Chained):
#
#     def __init__(self, x, sigma=1.0):
#         self.x = self.differentiable_wrt(x)
#         self.sigma = sigma
#
#     def compute_r(self):
#         return ((self.x.r / self.sigma) ** 2.0) * (self.x.r > 0)
#
#     def compute_dr_wrt(self, wrt):
#         diag = (2.0 * (self.x.r / self.sigma)) * (self.x.r > 0) + (1 / (self.sigma - self.x.r)) * (self.x.r < 0)
#         return sp.diags(diag, 0, format='csc')
#
#
# def min_point_to_point_distance_function(min_distances, vertex_indices_1=None, vertex_indices_2=None, normalized=True, sigma=1.0):
#     select1 = (lambda v: v) if (vertex_indices_1 is None) else (lambda v: Select(v, np.vstack([3 * np.array(vertex_indices_1), 3 * np.array(vertex_indices_1) + 1, 3 * np.array(vertex_indices_1) + 2]).T.flatten()))
#     select2 = (lambda v: v) if (vertex_indices_2 is None) else (lambda v: Select(v, np.vstack([3 * np.array(vertex_indices_2), 3 * np.array(vertex_indices_2) + 1, 3 * np.array(vertex_indices_2) + 2]).T.flatten()))
#     if normalized:
#         return lambda v1, v2: GTZero(Cw(min_distances) - (SquaredLength(select1(v1) - select2(v2))), sigma) / Cw(len(min_distances) ** 0.5)
#     else:
#         return lambda v1, v2: GTZero(Cw(min_distances) - (SquaredLength(select1(v1) - select2(v2))), sigma)
#
#
# def full_mesh_to_scan_function(scan, anchor_template, rho=None, normalized=True):
#     return mesh_to_scan_function(scan=scan, anchor_template=anchor_template, rho=rho,
#                                  sampling_strategy='vertices', normalized=normalized)
#
#
# def part_mesh_to_scan_function(scan, anchor_template, anchor_segmentation, parts, rho=None, normalized=True):
#     verts = np.unique(anchor_template.f[sum([anchor_segmentation[p] for p in parts], [])])
#     if scan.f.size:
#         return mesh_to_scan_function(scan=scan, anchor_template=anchor_template, rho=rho,
#                                      sampling_strategy='vertices', normalized=normalized, vertex_indices_to_sample=verts)
#     else:
#         return points_to_points_function(scan, rho=None, normalized=normalized, use_cgal=True, vertex_indices=verts)
#
#
# def deformed_model_function(scape_model):
#     """ Returns a funciton which outputs the flattened vertices of a deformed scape model where the model
#     has the specified trans, pose and shape, and vertices are deformed by the values in offsets"""
#
#     return lambda trans, pose, shape, offsets: scape_model.coords_for(trans, pose, shape) + offsets
#
#
# def landmark_function(template_landmarks, scan_landmarks, num_template_verts, difftype='euclidean'):
#     """ Returns a function which measures landmark error for input vertices,
#     assuming a known mapping between N landmarks and their vertex indices on a template.
#
#     Args:
#         scan_landmarks: a dict mapping landmark names to 3d locations
#         template_landmarks: a dict mapping landmark names to indices on the template
#         num_template_total: number of verts in the template
#         difftype: can be 'euclidean', 'squared-euclidean', or 'coordinate'
#
#     Note that if difftype is euclidean or squared-euclidean, the number of elements in "r" will
#     be the number of vertices, while if difftype is "coordinate", it will be (3 * # vertices)
#     """
#
#     lm_names = list((set(template_landmarks.keys()) & set(scan_landmarks.keys())))
#     num_landmarks = len(lm_names)
#
#     if num_landmarks < len(template_landmarks.keys()) or num_landmarks < len(scan_landmarks.keys()):
#         print 'warning: these landmarks are mismatched between template and scan:'
#         aa = set(template_landmarks.keys())
#         bb = set(scan_landmarks.keys())
#         print aa.difference(bb)
#         print bb.difference(aa)
#
#     # collect the template indices and scan 3d positions into lists
#     template_lm_idxs = np.array([template_landmarks[lm_name] for lm_name in lm_names])
#     scan_lm_pts = np.concatenate([scan_landmarks[lm_name] for lm_name in lm_names]).reshape(-1, 3) if lm_names else np.array([])
#
#     # construct a sparse matrix that converts between the landmark pts and all
#     # the pts, with height (# landmarks * 3) and width (# vertices * 3)
#     IS = np.arange(num_landmarks * 3)
#     JS = np.hstack(([col(template_lm_idxs * 3 + i) for i in range(3)]))
#     VS = np.ones(len(IS))
#     mtx = sparse(IS, JS, VS, num_landmarks * 3, num_template_verts * 3) if num_landmarks else sparse(IS, JS, VS, 1, num_template_verts * 3)
#
#     # This class will compute the difference (but not the distance)
#     class LandmarkDifference(Chained):
#
#         def __init__(self, x, mtx, scan_lm_pts):
#             self.x = self.differentiable_wrt(x)
#             self.mtx = mtx
#             self.scan_lm_pts = scan_lm_pts
#
#         def compute_r(self):
#             return ((self.mtx * col(self.x.r)).reshape(-1, 3) - scan_lm_pts) if len(self.scan_lm_pts) else np.array([0.0])
#
#         def compute_dr_wrt(self, obj):
#             return self.mtx
#
#     if num_landmarks == 0:
#         return lambda x: LandmarkDifference(x, mtx, scan_lm_pts)
#
#     if difftype == 'euclidean':
#         return lambda x: SquaredLength(LandmarkDifference(x, mtx, scan_lm_pts)) ** 0.5
#     elif difftype == 'squared-euclidean':
#         return lambda x: SquaredLength(LandmarkDifference(x, mtx, scan_lm_pts))
#     elif difftype == 'coordinate':
#         return lambda x: LandmarkDifference(x, mtx, scan_lm_pts)
#
#
# def normalized_landmark_function(template_landmarks, scan_landmarks, num_template_verts, rho=None, difftype='euclidean'):
#     landmark_func = landmark_function(template_landmarks, scan_landmarks, num_template_verts)
#
#     lm_names = list((set(template_landmarks.keys()) & set(scan_landmarks.keys())))
#     rho = rho if rho else lambda x: x
#
#     return lambda x: (rho(landmark_func(x)) / (len(lm_names) ** 0.5)).set_label('Normalized Landmark Errors')
#
#
# def edgediff_function(A3):
#
#     class EdgeDifference(Chained):
#
#         def __init__(self, xp, A3, edge_dr):
#             self.x = self.differentiable_wrt(xp)
#             self.A3 = A3
#             self._dr = edge_dr
#
#         def compute_r(self):
#             x1 = self.x.r[0:len(x) / 2]
#             x2 = self.x.r[len(x) / 2:]
#             x1e = self.A3 * x1
#             x2e = self.A3 * x2
#             return x1e - x2e
#
#         def compute_dr_wrt(self, wrt):
#             return self._dr
#
#     dr = sp.hstack((A3.T, -A3.T))
#     # return lambda xp : SquaredLength(EdgeDifference(xp,A3, dr)) ** 0.5
#     return lambda xp: EdgeDifference(xp, A3, dr)
#
#
# def vecdiff_function(vec_length):
#
#     class VecDifference(Chained):
#
#         def __init__(self, x_both, dr):
#             self.x = self.differentiable_wrt(x_both)
#             self._dr = dr
#
#         def compute_r(self):
#             x1 = self.x.r[0:len(self.x.r) / 2]
#             x2 = self.x.r[len(self.x.r) / 2:]
#             return self.x1 - self.x2
#
#         def compute_dr_wrt(self, wrt):
#             x1 = self.x.r[0:len(self.x.r) / 2]
#             x2 = self.x.r[len(self.x.r) / 2:]
#             return self._dr
#
#     dr = sp.hstack((sp.eye(vec_length, vec_length), -sp.eye(vec_length, vec_length)))
#     return lambda x: VecDifference(x, dr)
#
#
def co3(x):
    return bsxfun(np.add, row(np.arange(3)), col(3 * (x)))


def triangle_area(v, f):
    return np.sqrt(np.sum(np.cross(v[f[:, 1], :] - v[f[:, 0], :], v[f[:, 2], :] - v[f[:, 0], :]) ** 2, axis=1)) / 2


def sample_categorical(samples, dist):
    a = np.random.multinomial(samples, dist)
    b = np.zeros(int(samples), dtype=int)
    upper = np.cumsum(a)
    lower = upper - a
    for value in range(len(a)):
        b[lower[value]: upper[value]] = value
    np.random.shuffle(b)
    return b


def sample_from_mesh(mesh, sample_type='edge_midpoints', num_samples=10000, vertex_indices_to_sample=None, seed=0):

    # print 'WARNING: sample_from_mesh needs testing, especially with edge-midpoints and uniformly-at-random'
    if sample_type == 'vertices':
        if vertex_indices_to_sample is None:
            sample_spec = {'point2sample': sy.sparse.eye(mesh.v.size, mesh.v.size)}  # @UndefinedVariable
        else:
            sample_ind = vertex_indices_to_sample
            IS = co3(array(range(0, sample_ind.size)))
            JS = co3(sample_ind)
            VS = np.ones(IS.size)
            point2sample = sparse(IS.flatten(), JS.flatten(), VS.flatten(), 3 * sample_ind.size, 3 * mesh.v.shape[0])
            sample_spec = {'point2sample': point2sample}
    elif sample_type == 'uniformly-from-vertices':
        # Note: this will never oversample: when num_samples is greater than number of verts,
        # then the vert indices are all included (albeit shuffled), and none left out
        # (because of how random.sample works)

        #print("SEED IS", seed, 'SIZE is', mesh.v.shape[0], '#elements is', int(min(num_samples, mesh.v.shape[0])))
        random.seed(seed)  # XXX uncomment when not debugging
        np.random.seed(seed)
        sample_ind = np.array(random.sample(range(mesh.v.shape[0]), int(min(num_samples, mesh.v.shape[0]))))
        #print("FIRST ELEMENTS ARE", sample_ind[:100])
        IS = co3(array(range(0, sample_ind.size)))
        JS = co3(sample_ind)
        VS = np.ones(IS.size)
        point2sample = sparse(IS.flatten(), JS.flatten(), VS.flatten(), 3 * sample_ind.size, 3 * mesh.v.shape[0])
        sample_spec = {'point2sample': point2sample}
    else:
        if sample_type == 'edge-midpoints':
            tri = np.tile(array(range(0, mesh.f.size[0])).reshape(-1, 1), 1, 3).flatten()
            IS = array(range(0, tri.size))
            JS = tri
            VS = np.ones(IS.size) / 3
            area2weight = sparse(IS, JS, VS, tri.size, mesh.f.shape[0])
            bary = np.tile([[.5, .5, 0], [.5, 0, .5], [0, .5, .5]], 1, mesh.f.shape[0])

        elif sample_type == 'uniformly-at-random':
            random.seed(seed)  # XXX uncomment when not debugging
            np.random.seed(seed)
            tri_areas = triangle_area(mesh.v, mesh.f)
            tri = sample_categorical(num_samples, tri_areas / tri_areas.sum())
            bary = np.random.rand(tri.size, 3)
            flip = np.sum(bary[:, 0:1] > 1)
            bary[flip, :2] = 1 - bary[flip, 1::-1]
            bary[:, 2] = 1 - np.sum(bary[:, :2], 1)
            area2weight = sy.sparse.eye(tri.size, tri.size)  # @UndefinedVariable
        else:
            raise('Unknown sample_type')

        IS = []
        JS = []
        VS = []
        S = tri.size
        V = mesh.v.size / 3
        for cc in range(0, 3):
            for vv in range(0, 3):
                IS.append(np.arange(cc, 3 * S, 3))
                JS.append(cc + 3 * mesh.f[tri, vv])
                VS.append(bary[:, vv])

        IS = np.concatenate(IS)
        JS = np.concatenate(JS)
        VS = np.concatenate(VS)

        point2sample = sparse(IS, JS, VS, 3 * S, 3 * V)
        sample_spec = {'area2weight': area2weight, 'point2sample': point2sample, 'tri': tri, 'bary': bary}
    return sample_spec
