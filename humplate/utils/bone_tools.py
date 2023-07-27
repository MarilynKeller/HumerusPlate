"""
Copyright©2023 Max-Planck-Gesellschaft zur Förderung
der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
for Intelligent Systems. All rights reserved.

Author: Marilyn Keller
See LICENCE.txt for licensing and contact information.
"""

import numpy as np
from psbody.mesh import Mesh
import humplate.utils.geometry as geo
from humplate.utils.geometry import symmetrize_mesh, Rodrigues
from pickle import load
import chumpy as ch


def load_pca(pca_filename, num_betas=-1):
    ''' Takes a pkl file with the mean, PCA, and faces
        return a chumpy object parametrized  by t, r, betas
        and set at the mean'''

    pca_dict = load(open(pca_filename, 'rb'), encoding='latin1')

    B = pca_dict['B'][:, :, :num_betas]
    M = pca_dict['M']
    faces = pca_dict['f']

    trans = ch.zeros(3)
    rot_angles = ch.zeros(3)
    rot = Rodrigues(rt=rot_angles)
    betas = ch.zeros(B.shape[2])
    verts = ch.dot(M + ch.dot(B, betas), rot) + trans

    return trans, rot_angles, betas, verts, faces

def print_dist_stats(dist_array):
    print(f"Mean dist: {np.mean(dist_array.r):.2f} mm")
    print(f"Min dist:  {np.min(dist_array.r):.2f} mm")
    print(f"Max dist:  {np.max(dist_array.r):.2f} mm")


def save_right_side_ply(path):  
    """
    Load the mesh at the path, symmetrize it and save it under the path <previous path>_right.ply
    @param path: .ply file path
    @return: None
    """
    mesh = Mesh(filename=path)
    mesh = symmetrize_mesh(mesh)
    path.replace(".ply","_right.ply")
    
    right_side_path = path.replace(".ply","_right.ply")
    mesh.write_ply(right_side_path)
    print("Right side mesh saved as {}".format(right_side_path))



def get_plane_frames_from_humerus(point_dict, mesh):
    """
    Fit platnes to the fixation areas of the humerus and return the frames of reference attached to eahc plane as a dictionary
    """
    frame_dict = {}

    for points_group_key, points_group in point_dict.items():
        plane_frame = geo.fit_plane(points_group)

        # update centers of the frame so that it's a point on the mesh
        c_index, center_on_mesh = geo.mesh_closest_point(plane_frame.center, mesh)
        plane_frame.center = center_on_mesh
        plane_frame.center_idx = c_index

        # get normal and correct normal orientation to outside the mesh direction
        mesh_normal = geo.mesh_normal(c_index, mesh)
        plane_normal = plane_frame.frame[2]
        if np.dot(plane_normal, mesh_normal)<0:
            plane_frame.frame[2]=-plane_normal

        frame_dict[points_group_key] = plane_frame

    # Get the head - distal axe
    head_to_distal_direction = frame_dict["distal_plane"].center - frame_dict["head_plane"].center
    head_to_distal_direction = head_to_distal_direction * 1/np.linalg.norm(head_to_distal_direction)

    # orient properly the vectors i,j of the frames
    for points_group_key, points_group in point_dict.items():

        # flip the i vector to follow the head-distal direction
        i = frame_dict[points_group_key].frame[0]
        assert(geo.is_normed(i))
        dot_prod = np.dot(i,head_to_distal_direction.T)
        if dot_prod < 0:
            frame_dict[points_group_key].frame[0] = -i

        #flip j to build a direct frame with i and n
        frame_dict[points_group_key].frame[1] = np.cross(frame_dict[points_group_key].frame[2],\
                                                   frame_dict[points_group_key].frame[0])

    return frame_dict


