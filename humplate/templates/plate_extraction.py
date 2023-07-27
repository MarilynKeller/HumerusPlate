import humplate.utils.geometry as geo
import numpy as np
import humplate.config as cg
import copy
from psbody.mesh import Mesh

def B2P(b_coord, new_faces, bone):

    first_v = bone.v[bone.f[new_faces][:, 0]]
    second_v = bone.v[bone.f[new_faces][:, 1]]
    third_v = bone.v[bone.f[new_faces][:, 2]]

    new_v = first_v * b_coord[:, 0].reshape((-1, 1)) + second_v * b_coord[:, 1].reshape((-1, 1)) + third_v * b_coord[:, 2].reshape((-1, 1))

    return new_v


def B2P_normal(b_coord, new_faces, bone):
    bone_n = bone.estimate_vertex_normals()

    first_v =  bone_n[bone.f[new_faces][:, 0]]
    second_v = bone_n[bone.f[new_faces][:, 1]]
    third_v =  bone_n[bone.f[new_faces][:, 2]]

    new_n = first_v * b_coord[:, 0].reshape((-1, 1)) + second_v * b_coord[:, 1].reshape((-1, 1)) + third_v * b_coord[:, 2].reshape((-1, 1))

    return new_n


def apply_mapping(bone_new, mapping_file):
    import pickle as pkl
    mapping = pkl.load(open(mapping_file, 'rb'))

    b_coords = mapping['barycentric']
    new_faces = mapping['projected_faces']

    v = B2P(b_coords, new_faces, bone_new)
    f = mapping['plate_faces']
    n = B2P_normal(b_coords, new_faces, bone_new)

    v = v + mapping['eight'][:,np.newaxis] * n
    return v, f, n


def plate_extract(bone_mesh, 
                  tickness = cg.plate_extraction_offset,
                  plate_smoothing_iteration = cg.plate_smoothing_iteration,
                  plate_normal_smooting_iteration = cg.plate_normal_smooting_iteration, 
                  mapping_file = None):
    """
    @param bone_mesh: mesh on the bone to extract the plate from
    @param tickness: offset to add between the bone surface and the plate (in mm)
    @return: the extracted plate Mesh
    """

    bone_new = copy.copy(bone_mesh)
    mapping_file = cg.bone2plate_mapping
    v, f, n = apply_mapping(bone_new, mapping_file)

    # smooth normals got from the bone
    lap = geo.laplacian(Mesh(v=v, f=f))

    #smooth plate
    w = 0.5
    for i in range(plate_smoothing_iteration):
        v = v - w * lap * v

    v = geo.move_along_smoothed_normal(v, f, n, displacement=tickness, smoothing_iteration=plate_normal_smooting_iteration)

    extracted_plate = Mesh(v=v, f=f)
    return extracted_plate
