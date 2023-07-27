"""
Copyright©2023 Max-Planck-Gesellschaft zur Förderung
der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
for Intelligent Systems. All rights reserved.

Author: Marilyn Keller
See LICENCE.txt for licensing and contact information.
"""

import pickle
import logging
import numpy as np
from psbody.mesh import Mesh

import humplate.utils.draw as draw
import humplate.utils.geometry as geo
import humplate.config as cg
from humplate.utils.draw import get_annotation

logging.basicConfig(level=logging.WARNING)

#------------------------Parameters

head_color= [0, 0, 1]
midd_color= [1, 1, 1]
tail_color= [1, 0, 0]

zone_colors = [
head_color,
midd_color,
tail_color,
]

class BoneAnnotation():

    def __init__(self):

        # Load template
        self.bone_template_annotated = Mesh(filename=cg.bone_template_annotated)

        zone_masks = get_annotation(self.bone_template_annotated, zone_colors)

        self.head_mask = zone_masks[0]
        self.midd_mask = zone_masks[1]
        self.tail_mask = zone_masks[2]

        #Check that all the points are annotated
        self.nb_annot_verts = np.where(self.head_mask)[0].shape[0] + np.where(self.midd_mask)[0].shape[0] + np.where(self.tail_mask)[0].shape[0]
        self.nb_verts = self.bone_template_annotated.v.shape[0]

        if self.nb_annot_verts < self.nb_verts:
            print("WARNING: {} vertex(-ices) are not annotated, ie painted in red, blue or white.".format(self.nb_verts-self.nb_annot_verts))

    def annotate_bone(self, bone_mesh, middle_color = 0.7*np.ones(3)):

        if bone_mesh.v.shape[0] == self.head_mask.shape[0]:
            bone_mesh.vc = np.ones(bone_mesh.v.shape)
            bone_mesh.set_vertex_colors(head_color,   self.head_mask)
            bone_mesh.set_vertex_colors(middle_color, self.midd_mask)
            bone_mesh.set_vertex_colors(tail_color,   self.tail_mask)

        else:
            #Interpolate the colors
            ref_mesh = self.bone_template_annotated
            tree = ref_mesh.compute_aabb_tree()
            f = self.bone_template_annotated.f
            f_idx = tree.nearest(bone_mesh.v)[0][0]
            face_verts = f[f_idx]
            vc = np.zeros_like(bone_mesh.v)
            for i in range(vc.shape[0]):
                color_triplet = ref_mesh.vc[face_verts[i,:],:] # 3 lines, one per face vertex. One color on each line
                vc[i,:] = np.mean(color_triplet, axis=0)
            bone_mesh.vc = vc

    def annotated_mean_mesh(self):
        bone_template = Mesh(filename=cg.humerus_template, vc=[1, 1, 1])
        self.annotate_bone(bone_template)
        return bone_template

COLOR_LIST = [[1, 0.5, 0.5], [0.5, 0.5, 1]]


def apply_annotations(mesh, annotation_filename):
    """ Load a annotation file and color the mesh with it.
    Input :
         mesh: psbody.Mesh
         annotation_filename: path to a pickle path containing a dictionary with arrays of indices,
     which are the indices of the points of zones in the template.
     for example annotation_filename = {'zone1' : np.array([10,11,15,18,120,1247])}
    Output :
        dictionnary with the same keys as the input one but with the coordinates of the points stored
        in an array instead of an index.

     """

    logging.info("Applying annotations from file " + annotation_filename + " to mesh.")
    annotations = pickle.load(open(annotation_filename, 'rb'), encoding='latin1')
    annotated_points = {}
    assert(len(annotations) <= len(COLOR_LIST))
    for i, point_zone in enumerate(annotations.values()):
        mesh.set_vertex_colors(COLOR_LIST[i], point_zone)

    for key in annotations:
        annotated_points[key] = mesh.v[annotations[key]]

    return annotated_points

def annotate_heatmap(mesh, point_dictionary, heatmap_dictionary, max_value=5.0):
    for point_zone_key, point_zone in point_dictionary.items():
        heatmap_dictionary[point_zone_key] = heatmap_dictionary[point_zone_key] / max_value
        for i, p in enumerate(point_zone.tolist()):
            index = np.where((mesh.v[:,0] == p[0]) * (mesh.v[:,1] == p[1]) * (mesh.v[:,2] == p[2]))
            mesh.set_vertex_colors([heatmap_dictionary[point_zone_key][i]], [index])


def compute_local_frame_lines(point_dict):
    """ Return a list of psbody.Mesh.Lines that draw local frames of the tangeant plane for each of the point group
    Input:
     dict  { key: nx3 matrix} """
    errors_dict = {}
    lines = []
    for points_group_key, points_group in point_dict.items():
        plane_frame = geo.fit_plane(points_group)
        # print "plane params ", center, (i,j,n), global_rms_dist

        errors_dict[points_group_key] = plane_frame.point_dist
        center = plane_frame.center
        [i,j,n] = plane_frame.frame

        k=15

        line_i = draw.createLines([center-k*i, center+k*i])  # pairs of consecutive vertices (start, end)
        line_i.set_edge_colors([1,0,0])

        line_j = draw.createLines([center-k*j, center+k*j])  # pairs of consecutive vertices (start, end)
        line_j.set_edge_colors([0,1,0])

        line_n = draw.createLines([center-2*k*n, center+2*k*n])  # pairs of consecutive vertices (start, end)
        line_n.set_edge_colors([0, 0, 1])

        lines+=[line_n,line_i,line_j]
    return lines, errors_dict
