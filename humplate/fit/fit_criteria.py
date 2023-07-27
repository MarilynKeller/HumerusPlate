import numpy as np
import humplate.config as cg

from sbody.ch.mesh_distance import PtsToMesh, sample_from_mesh
from humplate.templates.plate_annotation import PlateAnnotation

""" Class to store the fit info of a plate set on a bone set"""

def compute_fit_score(dist_fit_rate, nb_intersection):
    no_intersect = (nb_intersection == 0)
    fit_score = (dist_fit_rate + dist_fit_rate * no_intersect) * 0.5
    return fit_score

def fit_criteria(dist_fit_rate, nb_intersection, thresh):
    fit_score = compute_fit_score(dist_fit_rate, nb_intersection)
    return fit_score > thresh


class FitData:
    """
    Contains info for the fit of a plate:bone pair
    """
    def __init__(self, plate, bone, cost=None, R=None, T=None):

        # Plate to bone distance
        bone_mesh_sampler = sample_from_mesh(bone, sample_type='uniformly-from-vertices', num_samples=len(bone.v))
        plate2bone = PtsToMesh(
            sample_verts=plate.v, reference_verts=bone.v, reference_faces=bone.f,
            reference_template_or_sampler=bone_mesh_sampler,
            signed=True, normalize=False)
        dists = plate2bone.r

        # ---------------------------------------------------
        # Check plate to bone distance
        # --------------------------------------------------
        pa = PlateAnnotation()

        # Get per part distances
        head_dists = dists[pa.head_mask]
        midd_dists = dists[pa.midd_mask]
        tail_dists = dists[pa.tail_mask]

        # Get per part points numbers
        head_pts_nb = np.count_nonzero(pa.head_mask)
        midd_pts_nb = np.count_nonzero(pa.midd_mask)
        tail_pts_nb = np.count_nonzero(pa.tail_mask)
        assert ((head_pts_nb + midd_pts_nb + tail_pts_nb) == pa.nb_annot_verts)
        if pa.nb_annot_verts < dists.shape[0]:
            print("WARNING: Some points of the plate are not annotated")

        # Get per part fitting points numbers
        head_fit_nb = np.count_nonzero(head_dists < cg.h_dist)
        midl_fit_nb = np.count_nonzero(midd_dists < cg.m_dist)
        tail_fit_nb = np.count_nonzero(tail_dists < cg.t_dist)
        total_fit_nb = head_fit_nb + midl_fit_nb + tail_fit_nb
        assert (total_fit_nb <= pa.nb_annot_verts)

        # Get fitting rate
        glob_fit_rate = total_fit_nb * 1.0 / pa.nb_annot_verts
        head_fit_rate = head_fit_nb * 1.0 / head_pts_nb
        midl_fit_rate = midl_fit_nb * 1.0 / midd_pts_nb
        tail_fit_rate = tail_fit_nb * 1.0 / tail_pts_nb

        # Check intersection
        intersection_mask = (dists < -cg.intersection_dist).astype(int) #
        nb_intersection = np.count_nonzero(intersection_mask)

        # Store results
        self.plate = plate
        self.dists = dists
        self.cost = cost
        if R is not None:
            self.R = R
        if T is not None:
            self.T = T

        self.glob_fit_rate = glob_fit_rate
        self.head_fit_rate = head_fit_rate
        self.midl_fit_rate = midl_fit_rate
        self.tail_fit_rate = tail_fit_rate
        self.intersection_mask = intersection_mask
        self.nb_intersection = nb_intersection
        
    def get_fit_score(self):
        return compute_fit_score(self.glob_fit_rate, self.nb_intersection)

    def do_fit(self, thresh):
        return fit_criteria(self.glob_fit_rate, self.nb_intersection, thresh)

    def __str__(self):
        
        fit_score = self.do_fit(cg.fit_thresh)
        s = ""
        # s += f"Plate fits : {fit_score > cg.fit_thresh} /n"
        s += "Fit score : {}\n".format(self.get_fit_score())
        s += "Global distance fit rate : {}\n".format(self.glob_fit_rate * 100)
        s += "  Head : {:0.2f}     \n".format(self.head_fit_rate * 100)
        s += "  Middle : {:.2f}    \n".format(self.midl_fit_rate * 100)
        s += "  Tail : {:.2f}      \n".format(self.tail_fit_rate * 100)
        s += "Number of intersection points : {:.2f}      \n".format(self.nb_intersection * 100)
        return s
