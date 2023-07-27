import cv2
from psbody.mesh import Mesh
from humplate.templates.plate_extraction import plate_extract
from humplate.templates.plate_annotation import PlateAnnotation
from humplate.utils.geometry import rigid_alignment_analytic_weighted

def closed_form_plate_position(plate, bone):

    # closed form init
    weight_head = 1
    weight_tail = 0.05
    source_plate = plate

    vol_mesh = bone
    target_plate = plate_extract(vol_mesh)

    pa = PlateAnnotation()
    weight_vector = weight_head * pa.head_mask.astype(int) + weight_tail * pa.tail_mask.astype(int)
    unposed_verts, R, T = rigid_alignment_analytic_weighted(source_plate.v, target_plate.v, weight_vector)
    positioned_plate = Mesh(v=unposed_verts, f=target_plate.f, vc=[1, 1, 1])
    # bdw.show_meshes([positioned_plate, target_plate, bone], title="closed_form_plate_position", colors=True)

    trans_init = T[0]
    rot_vect = cv2.Rodrigues(R)[0]

    return positioned_plate, trans_init, rot_vect

