"""
Copyright©2023 Max-Planck-Gesellschaft zur Förderung
der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
for Intelligent Systems. All rights reserved.

Author: Marilyn Keller
See LICENCE.txt for licensing and contact information.
"""

"""Segment the plate_template in different zones"""


import numpy as np
from psbody.mesh import Mesh
import humplate.data_paths as bd
from humplate.utils.draw import get_annotation
import humplate.config as cg

#------------------------Parameters
normal_length = 25
zone_type_list = ["x_lim", "painted"]

head_color= [0, 0, 1]
midd_color= [1, 1, 1]
tail_color= [1, 0, 0]

zone_colors = [
head_color,
midd_color,
tail_color,
]

head_center_index = 349
tail_center_index = 1776 #1591
#------------------------------------

class PlateAnnotation():

    def __init__(self, zone_type='painted'):

        plate_template = Mesh(filename=cg.plate_template_contoured)

        assert(zone_type in zone_type_list)

        if zone_type == "x_lim":
            head_x_min = 120 #166 175 183
            tail_x_max = 0 #100 #76
            self.head_mask = plate_template.v[:, 0] >= head_x_min
            self.midd_mask = np.logical_and(tail_x_max < plate_template.v[:, 0], plate_template.v[:, 0] < head_x_min)
            self.tail_mask = plate_template.v[:, 0] <= tail_x_max

        elif zone_type == "painted":

            plate_template_annotated = Mesh(filename=cg.plate_template_contoured_annotated)

            zone_masks = get_annotation(plate_template_annotated, zone_colors)

            self.head_mask = zone_masks[0]
            self.midd_mask = zone_masks[1]
            self.tail_mask = zone_masks[2]

        self.nb_annot_verts = np.where(self.head_mask)[0].shape[0] + np.where(self.midd_mask)[0].shape[0] + np.where(self.tail_mask)[0].shape[0]
        self.nb_verts = plate_template.v.shape[0]

        if self.nb_annot_verts < self.nb_verts:
            print("WARNING: {} vertex(-ices) are not annotated, ie painted in red, blue or white.".format(self.nb_verts-self.nb_annot_verts))

    def annotate_plate(self, plate_mesh, middle_color = 0.7*np.ones(3)):
        plate_mesh.set_vertex_colors(head_color, self.head_mask)
        plate_mesh.set_vertex_colors(middle_color, self.midd_mask)
        plate_mesh.set_vertex_colors(tail_color, self.tail_mask)

    def get_landmarks(plate_mesh, normal_length=25):
        plate0_n = plate_mesh.estimate_vertex_normals()
        landm = {}
        landm["head_center"] = plate_mesh.v[head_center_index]
        landm["head_normal"] = landm["head_center"] + normal_length * plate0_n[head_center_index]
        landm["tail_center"] = plate_mesh.v[tail_center_index]
        landm["tail_normal"] = landm["tail_center"] + normal_length * plate0_n[tail_center_index]
        return landm

    def annotated_mean_mesh(self):
        m = Mesh(filename=cg.plate_template_contoured, vc=[1, 1, 1])
        self.annotate_plate(m)
        return m
