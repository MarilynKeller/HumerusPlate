"""
Copyright©2023 Max-Planck-Gesellschaft zur Förderung
der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
for Intelligent Systems. All rights reserved.

See LICENCE.txt for licensing and contact information.
"""

from psbody.mesh.lines import Lines
import numpy as np


def createLines(verts):
    '''Create a mesh.line.Line object from a list of 2 vertices [[x,y,z], [xx,yy,zz]]'''

    assert(len(verts)==2)
    import numpy as np
    lines_verts = np.zeros((len(verts), 3))
    edges = np.zeros((len(verts) // 2, 2), dtype=int)
    for i, v in enumerate(verts):
        lines_verts[i] = v

    for i in range(len(verts) // 2):
        edges[i][0] = 2 * i
        edges[i][1] = 2 * i + 1

    return Lines(v=lines_verts, e=edges)


def plane_frames_2_mesh_lines(plane_dict, line_length=15):
    """ Return a list of psbody.Mesh.Lines that draw local frames of the tangeant plane for each of the point group
    Input:
     dict  { key: nx3 matrix} """
    errors_dict = {}
    lines = []
    for frame_key, frame_params in plane_dict.items():

        center = plane_dict[frame_key].center
        [i,j,n] = plane_dict[frame_key].frame

        k=line_length

        line_i = createLines([center, center+k*i])  # pairs of consecutive vertices (start, end)
        line_i.set_edge_colors([1,0,0])

        line_j = createLines([center, center+k*j])  # pairs of consecutive vertices (start, end)
        line_j.set_edge_colors([0,1,0])

        line_n = createLines([center, center+2*k*n])  # pairs of consecutive vertices (start, end)
        line_n.set_edge_colors([0, 0, 1])

        lines+=[line_n,line_i,line_j]
    return lines


def get_annotation(annotated_mesh, colors):
    """
    From a colored mesh, get the vertices masks for each color in the list @colors
    @param annotated_mesh: Mesh object with vc
    @param colors:
    @return:
    """
    zone_masks = []
    for color in colors:

        # Find the indices that have that color
        nb = annotated_mesh.v.shape[0]

        indices = np.where(np.all(annotated_mesh.vc == color, axis=-1))[0]

        # Create a mask for those indices
        mask = np.zeros(nb, dtype=bool)
        mask[indices] = 1

        zone_masks.append(mask)

    return zone_masks
