"""Apply a rigid transfo to plate -p to make it fit a bone i or all the bones"""

import copy
import numpy as np
import chumpy as ch
import random
import pickle as pkl
from psbody.mesh import Mesh, MeshViewers
import os

from sbody.objectives import sample_from_mesh
from sbody.ch.mesh_distance import PtsToMesh, sample_from_mesh

import humplate.config as cg
import humplate.utils.plate_tools as pt
from humplate.utils.geometry import get_submesh, Rodrigues, symmetrize_mesh, distance_computation, symmetrize_mesh
from humplate.templates.plate_annotation import PlateAnnotation
from humplate.templates.bone_annotation import BoneAnnotation
from humplate.templates.plate_extraction import plate_extract
from humplate.fit.fit_function import plate_snap_cost, plate_collide_cost, plot_snap_collide_cost
from humplate.fit.fit_criteria import FitData
import humplate.data_paths as bd


def fit_plate_to_bone(plate, bone, visual_check, title, snapshot_file, 
                      rot_vect=np.array([0, 0, 0.0]),
                      trans_init=np.zeros(3),
                      symmetrize=False, display=False,
                      random_trans_range=0):


    verbose = display
    close_form_init = cg.close_form_init
    do_first_opt = cg.do_first_opt
    do_second_opt = cg.do_second_opt
    do_third_opt = cg.do_third_opt
    opt_method = cg.opt_method

    if do_first_opt:
        print("WARNING: Do first init is forced.")
    if do_second_opt == False:
        print("WARNING: Optimization 2 is skipped.")
    if do_third_opt == False:
        print("WARNING: Optimization 3 is skipped.")

    bone = copy.deepcopy(bone)

    if symmetrize:
        bone = symmetrize_mesh(bone)

    if close_form_init and (trans_init is None):
        print("Doing close form init")
        plate_init, trans_init, rot_vect = pt.closed_form_plate_position(plate, bone)

    if random_trans_range != 0:
        dd = random_trans_range
        random.seed(0)
        dy = random.uniform(-dd, dd)
        dz = random.uniform(-dd, dd)
        trans_init = trans_init + np.array([0, dy, dz])

    # Declare plate transformation variables
    trans = ch.array(trans_init)
    rot_angles = ch.array(rot_vect)
    rot = Rodrigues(rt=rot_angles)

    verts = ch.dot(plate.v, rot.T) + trans
    # positioned_mesh = Mesh(v=verts.r, f=plate.f)
    # bdw.show_meshes([positioned_mesh, bone], title="Initial plate placement in fct optimisation")

    ch.max(verts)


    pa = PlateAnnotation()

    #snap cost
    bone_mesh_sampler = sample_from_mesh(bone, sample_type='uniformly-from-vertices', num_samples=len(bone.v))
    plate2bone = PtsToMesh(
        sample_verts=verts, reference_verts=bone.v, reference_faces=bone.f, reference_template_or_sampler=bone_mesh_sampler,
        signed=True, normalize=False)
    snap_cost = plate_snap_cost(plate2bone, cg.m_dist)
    pene_cost = plate_collide_cost(plate2bone)

    mesh_sampler = sample_from_mesh(plate, sample_type='uniformly-from-vertices', num_samples=len(plate.v))

    # zone snap
    ba = BoneAnnotation()

    custom_plate = plate_extract(bone)

    (bone_tail_verts, bone_tail_faces, _, _) = get_submesh(bone.v, bone.f,
                                                           verts_retained=ba.tail_mask)
    (bone_head_verts, bone_head_faces, _, _) = get_submesh(bone.v, bone.f,
                                                           verts_retained=ba.head_mask)
    bone_tail = Mesh(v=bone_tail_verts, f=bone_tail_faces)
    bone_head = Mesh(v=bone_head_verts, f=bone_head_faces)
    # Check parts visually
    # bone_tail.show()
    # bone_head.show()
    # import ipdb; ipdb.set_trace()

    bone_tail_mesh_sampler = sample_from_mesh(bone_tail, sample_type='uniformly-from-vertices',
                                              num_samples=len(bone_tail.v))
    bone_head_mesh_sampler = sample_from_mesh(bone_head, sample_type='uniformly-from-vertices',
                                              num_samples=len(bone_head.v))

    plate2bonetail = PtsToMesh(
        sample_verts=verts[pa.tail_mask], reference_verts=bone_tail.v, reference_faces=bone_tail.f,
        reference_template_or_sampler=bone_tail_mesh_sampler,
        signed=True, normalize=False)

    snap_tail = plate_snap_cost(plate2bonetail, cg.m_dist)
    penetrate_tail = plate_collide_cost(plate2bonetail)

    plate2bonehead = PtsToMesh(
        sample_verts=verts[pa.head_mask], reference_verts=bone_head.v, reference_faces=bone_head.f,
        reference_template_or_sampler=bone_head_mesh_sampler,
        signed=True, normalize=False)

    snap_head = plate_snap_cost(plate2bonehead, cg.h_dist)

    penetrate_head = plate_collide_cost(plate2bonehead)

    position_head = ch.sqrt(ch.power(verts[pa.head_mask] - custom_plate.v[pa.head_mask], 2))
    position_tail = ch.sqrt(ch.power(verts[pa.tail_mask] - custom_plate.v[pa.tail_mask], 2))


    # Check parts visually
    # plate.vc = np.ones_like(plate.v)
    # plate.set_vertex_colors(plate2bonehead.r, np.where(pa.head_mask))
    # bdw.show_meshes([bone_head, plate], colors=False)
    #
    # plate.set_vertex_colors(plate2bonetail.r, np.where(pa.tail_mask))
    # bdw.show_meshes([bone_tail, plate], colors=False)


    def on_step(_):
        global c

        if display:
            mesh_list = []
            plate_mesh = Mesh(v=verts.r, f=plate.f)

            plate_mesh.set_vertex_colors_from_weights(pene_cost.r)
            plate_mesh.set_vertex_colors(penetrate_head.r,vertex_indices=pa.head_mask)
            plate_mesh.set_vertex_colors(penetrate_tail.r,vertex_indices=pa.tail_mask)
            # plate_mesh.vc[pa.head_mask] = penetrate_head.r
            mv[0][0].dynamic_meshes = [plate_mesh]

            plate_mesh.set_vertex_colors_from_weights(snap_cost.r)
            plate_mesh.set_vertex_colors(snap_head.r/cg.h_dist,vertex_indices=pa.head_mask)
            plate_mesh.set_vertex_colors(snap_tail.r/cg.t_dist,vertex_indices=pa.tail_mask)
            mv[0][1].dynamic_meshes = [plate_mesh]

            plot_snap_collide_cost(plate2bone)

            mesh_list = [plate_mesh]
            mv[0][1].dynamic_meshes = mesh_list
            # print('rot : {}   trans : {}'.format(rot_angles.r, trans.r))

    # positioned_mesh = Mesh(v=verts.r, f=plate.f)
    # bdw.show_meshes([positioned_mesh, bone])
    
    if display:
        mv = MeshViewers(shape=(1, 2), keepalive=False, titlebar=title)
        # bone.set_face_colors(np.asarray([0., 1.0, 0.]))
        
        bone.vc = Mesh(filename=cg.bone_template_annotated).vc

        mv[0][0].dynamic_meshes = [Mesh(v=verts.r, f=plate.f)]
        mv[0][1].static_meshes = [bone]
        mv[0][1].dynamic_meshes = [Mesh(v=verts.r, f=plate.f)]
        on_step(None)


    # --------------------------
    # 1. Plate rigid translation and rotation
    # --------------------------

    maxiter = 100

    # Step 1
    obj = {}
    obj['position_head'] = 5 * position_head / position_head.shape[0]
    obj['position_tail'] = position_tail / position_tail.shape[0]

    print('Rigid transfo 1')
    if np.all(rot_vect==0) and do_first_opt:
        if visual_check:
            print("Right click to continue")
            mv[0][0].get_mouseclick()
        if np.count_nonzero(position_head)>0:
            ch.minimize(obj, x0=[trans, rot_angles], method=opt_method, callback=on_step,
                    options={'disp': verbose, 'maxiter': maxiter})
    else:
        print("Initial positioning of the plate provided, skipping rigid optimisation one.")

    # Step 2
    obj = {}
    obj['snap_head'] = 1 * snap_head / snap_head.shape[0]
    obj['snap_tail'] = 1* snap_tail/snap_tail.shape[0]
    obj['pene'] = 0.01* pene_cost / pene_cost.shape[0]

    if do_second_opt:
        print('Rigid transfo 2')
        if visual_check:
            print("Right click to continue")
            mv[0][0].get_mouseclick()
        if np.count_nonzero(position_head)>0:
            ch.minimize(obj, x0=[trans, rot_angles], method=opt_method, callback=on_step,
                    options={'disp': verbose, 'maxiter': maxiter})
    else:
        print("WARNING: skip second opt.")

    # Step 3
    obj = {}
    obj['position_head'] = 1*position_head / position_head.shape[0]
    obj['snap_head'] = 100*snap_head/snap_head.shape[0]
    obj['snap_tail'] = snap_tail/snap_tail.shape[0]
    obj['pene'] = 0.10*pene_cost
    obj['snap'] = snap_cost/snap_cost.shape[0]

    if do_third_opt:
        print('Rigid transfo 3')
        if visual_check: 
            print("Right click to continue")
            mv[0][0].get_mouseclick()
        ch.minimize(obj, x0=[trans, rot_angles], method=opt_method, callback=on_step,
                    options={'disp': verbose, 'maxiter': maxiter})
        print(trans.r)

    thresh_mask = cg.h_dist * pa.head_mask + cg.m_dist * pa.midd_mask + cg.t_dist * pa.tail_mask
    final_fit_fct = ch.clip(plate2bone - thresh_mask, 0, None )

    # Step 4
    obj = {}
    obj['fit_cost'] = 1*final_fit_fct
    obj['pene'] = 10*pene_cost

    print('Rigid transfo 4')
    if visual_check: 
        print("Right click to continue")
        mv[0][0].get_mouseclick()
    ch.minimize(obj, x0=[trans, rot_angles], method=cg.opt_final, callback=on_step,
                options={'disp': verbose, 'maxiter': maxiter})
    print(trans.r)


    plate_transformed = Mesh(v=verts, f=plate.f)
    plate2bone = distance_computation(verts, bone)

    return plate_transformed, rot, trans, obj


def run_fit_plate2bone(plate_id, vol_id, plate_mesh, vol_mesh, swarm_index=None, display=False, visual_check=False, force_recompute=True, rot_vect=np.array([0, 0, 0.0]), trans_init=np.zeros(3)):
    """
    Fit a plate to a bone rigidly
    @param plate_id: id of the plate to fit (the id is used to store the results)
    @param vol_id: id of the bone to fit the plate to
    @param plate_mesh: mesh of the plate to fit
    @param vol_mesh: mesh of the bone to fit the plate to
    @param display: display the fitting process in meshviewer
    @param visual_check: pause display after each optimisation step
    @param force_recompute: recompute the fits for the bones that already have a plate-distance file saved
    @param rot_vect: initial rotation of the plate
    @param trans_init: initial translation of the plate
    @return: rot_angles, T, fit_rate, cost_tot or tuple of None
"""

    vol_mesh = copy.deepcopy(vol_mesh)

    # Check if fit file exists, otherwise align to compute the fit
    if swarm_index is not None:
        fit_file = bd.get_plate_bone_fit(vol_id, plate_id, swarm_index)
    else:
        fit_file = bd.get_plate_bone_fit(vol_id, plate_id)

    title = "plate {} fit on {}".format(plate_id, vol_id)
    snapshot_file = bd.get_plate_on_bone_snapshot(plate_id, vol_id)

    # Run rigid alignment and get the distances
    if force_recompute or (not os.path.exists(fit_file)):
        plate_transformed, R, T, cost_dict = fit_plate_to_bone(plate_mesh, vol_mesh, visual_check, title, snapshot_file,
                                                      symmetrize=False,
                                                      display=display,
                                                      trans_init=trans_init,
                                                      rot_vect=rot_vect)

        print("Final costs")
        for key, value in cost_dict.items():
            cost = np.sum(value.r ** 2)
            print("{}: {}".format(key, cost))


        obj_concat = ch.concatenate([f.ravel() for f in list(cost_dict.values())])
        cost_tot = np.sum(obj_concat.r ** 2)
        print("Total cost: {}".format(cost_tot))

        # Compute fit data
        fit_data = FitData(plate_transformed, vol_mesh, cost_tot, R, T)
        print(fit_data)

        # Show plate on bone
        if display:
            p = plate_transformed
            b = vol_mesh

            # from plate.plate_tools import plate_dist2scalar
            # color_scalar = plate_dist2scalar(plate2bone_dists=plate2bone.r)
            p.set_vertex_colors_from_weights(fit_data.dists / 5.0, scale_to_range_1=False)  # max 5mm
            mv2 = MeshViewers(shape=(1, 1), keepalive=False, titlebar=title)

            mv2[0][0].dynamic_meshes = [p, b]
            mv2[0][0].set_background_color(np.ones((3)))
            mv2[0][0].save_snapshot(blocking=True, path=snapshot_file.replace(".png", "_head.png"))
            # rotate bone to view other side
            mv_rotation = [-3.14 / 2, 0, 0]
            b.rotate_vertices(mv_rotation)
            p.rotate_vertices(mv_rotation)
            mv2[0][0].dynamic_meshes = [p, b]
            mv2[0][0].save_snapshot(blocking=True, path=snapshot_file.replace(".png", "_tail.png"))
            if visual_check: mv2[0][0].get_mouseclick()

        # save the fit data
        pkl.dump(fit_data, open(fit_file, "wb"))
        print("plate to bone fit info saved at {}".format(fit_file))

        return fit_data

    else:
        print("Fit plate '{}' to bone '{}' already computed, will be ignored.".format(
            plate_id, vol_id))

        with open(fit_file, "rb") as f:
            fit_data = pkl.load(f)
        return fit_data

