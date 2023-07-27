"""
Copyright©2023 Max-Planck-Gesellschaft zur Förderung
der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
for Intelligent Systems. All rights reserved.

Author: Marilyn Keller and Sergi Pujades
See LICENCE.txt for licensing and contact information.
"""

import os 
import numpy as np
import chumpy as ch
from psbody.mesh import Mesh, MeshViewers
from pickle import dump

from sbody.objectives import sample_from_mesh
from sbody.ch.mesh_distance import ScanToMesh, MeshToScan
from sbody.robustifiers import GMOf
from humplate.templates.bone_annotation import BoneAnnotation
from humplate.utils.topology import get_vertices_per_edge
from humplate.utils.geometry import laplacian, rigid_alignment_analytic, symmetrize_mesh
from humplate.utils.bone_tools import load_pca
import humplate.utils.bone_tools as geo


class RegParams:
    def __init__(self):
        self.trans = np.array([0.,0.,0.])
        self.rot = np.array([0.,0.,0.])
        self.num_samples_max = 25e3
        self.laplace_coupling = True
        self.ignore_watertight = False
        self.symmetrize = False
        self.template_filename = ""
        self.max_iter = 50
        self.free_verts_max_iter = 50
        self.missing_part = False # Set to True if parts are missing in the scan. Then after the pca registration step,
        # only the template points closer to the scan than missing_part_dist, will be optimised in the free verts optimisation
        self.missing_part_dist = 3 #mm
        self.laplace_cpl_weight = 0.05
        self.weight_beta_norm = 2
        self.gmo_sigma_beta = 5  # snap if <5mm
        self.gmo_sigma_ff = 5  # snap if <5mm
        self.is_scan_clean = False # I set to True, a second round of free verts registration will be performed after the first one, with released laplacian coupling

        # Visualisation
        self.display = False
        self.visual_check = False
        self.title = ""

        # output paths
        self.force_recompute = True
        self.align_filename = ""
        self.align_data_filename = ""
        self.unposed_filename = ""
        self.unposed_data_filename = ""

        self.align_pca_only_filename = ""
        self.align_pca_only_data_filename = ""
        self.unposed_pca_only_filename = ""
        self.unposed_pca_only_data_filename = ""


def align_bonepca2scan(pca_filename, scan_filename, num_betas, reg_params):

    scan = Mesh(filename=scan_filename)

    if reg_params.symmetrize:
        scan = symmetrize_mesh(scan)

    do_mesh2scan = False

    t, r, betas, verts, faces = load_pca(pca_filename, num_betas)
    ff = ch.zeros((verts.shape[0], 3))
    verts = verts + ff


    template = Mesh(v=verts.r, f=faces)
    ba = BoneAnnotation()
    ba.annotate_bone(template)
    bone_vc = template.vc

    t[:] = reg_params.trans
    r[:] = reg_params.rot

    num_samples = len(scan.v)
    if num_samples > reg_params.num_samples_max:
        num_samples = reg_params.num_samples_max

    scan_sampler = sample_from_mesh(scan, sample_type='uniformly-from-vertices', num_samples=num_samples)
    mesh_sampler = sample_from_mesh(template, sample_type='uniformly-from-vertices', num_samples=len(template.v))

    s2m = ScanToMesh(scan, verts, faces,
                       scan_sampler=scan_sampler)


    if reg_params.display:
        mv = MeshViewers(shape=(1, 2), keepalive=False, titlebar=reg_params.title)
        scan.set_face_colors(np.asarray([0., 1.0, 0.]))
        # mv[0][0].static_meshes = [scan]
        mv[0][1].static_meshes = [scan]
        mv[0][1].dynamic_meshes = [Mesh(v=verts.r, f=faces)]

    global c

    def on_step(_):
        global c

        if reg_params.display:
            mv[0][0].dynamic_meshes = [Mesh(v=verts.r, f=faces, vc=bone_vc)]
            mv[0][1].dynamic_meshes = [Mesh(v=verts.r, f=faces, vc=bone_vc)]

    #-------------------------------------------------------------
    # 2. Tanslation only
    # -------------------------------------------------------------
    obj = {
        's2m': s2m,
    }

    print('Trans only')
    if reg_params.visual_check : mv[0][0].get_mouseclick()
    ch.minimize(obj, x0=[t], method='dogleg', callback=on_step, options={'maxiter': reg_params.max_iter})
    print(t.r)

    #-------------------------------------------------------------
    # 2. 'Trans, rot '
    # -------------------------------------------------------------
    print('Trans and rot')
    if reg_params.visual_check : mv[0][0].get_mouseclick()
    ch.minimize(obj, x0=[t, r], method='dogleg', callback=on_step, options={'maxiter': reg_params.max_iter})
    print(t.r, r.r)

    #-------------------------------------------------------------
    # 3. 'Trans, rot and betas'
    # -------------------------------------------------------------

    s2m = ScanToMesh(scan, verts, faces,
                       scan_sampler=scan_sampler,
                       rho = lambda x: GMOf(x, reg_params.gmo_sigma_beta))

    beta_norm = ch.linalg.norm(betas)*1.0/num_betas


    weights = dict()
    weights['s2m'] = 1.
    weights['beta_norm'] = reg_params.weight_beta_norm
    obj = {
        's2m': s2m,
        'beta_norm' : weights['beta_norm'] * beta_norm
    }

    print('Trans, rot and betas')
    if reg_params.visual_check : mv[0][0].get_mouseclick()
    ch.minimize(obj, x0=[t, r, betas],
                method='dogleg', callback=on_step, options={'maxiter': reg_params.max_iter})

    print(t.r, r.r, betas.r)

    # Save results
    aligned_bone = Mesh(v=verts.r, f=faces)
    os.makedirs(os.path.dirname(reg_params.align_filename), exist_ok=True)
    if reg_params.align_pca_only_filename:
        aligned_bone.write_ply(reg_params.align_pca_only_filename)
    result = dict()
    result['v'] = verts.r
    result['rot'] = r.r
    result['trans'] = t.r
    result['betas'] = betas.r
    dump(result, open(reg_params.align_data_filename, 'wb'))

    #-------------------------------------------------------------
    # 4.  free vertices
    # -------------------------------------------------------------

    vpe = get_vertices_per_edge(verts.r, faces)
    edges_for = lambda x: x[vpe[:, 0]] - x[vpe[:, 1]]

    free_verts = verts

    s2m_f = ScanToMesh(scan, free_verts, faces,
                       scan_sampler=scan_sampler,
                       rho = lambda x: GMOf(x, reg_params.gmo_sigma_ff))

    if do_mesh2scan:
        m2s_f = MeshToScan(scan, free_verts, faces,
                           mesh_template_or_sampler=mesh_sampler)
    else:
        m2s_f = ch.zeros(1)


    weights = dict()
    weights['s2m'] = 1.


    if do_mesh2scan:
        weights['m2s'] = 1.
        weights['cpl'] = 0.01
    else:
        weights['m2s'] = 0.
        weights['cpl_lap'] = reg_params.laplace_cpl_weight

    if reg_params.laplace_coupling == True:
        lap_op = np.asarray(laplacian(Mesh(verts.r, template.f)).todense())
        objs = {'s2m': weights['s2m'] * s2m_f,
                'm2s': weights['m2s'] * m2s_f,
                'cpl_lap': weights['cpl_lap'] * ch.dot(lap_op, ff)}
    else:
        objs = {'s2m': weights['s2m'] * s2m_f,
                'm2s': weights['m2s'] * m2s_f,
                'cpl': weights['cpl'] * (edges_for(verts.r) - edges_for(free_verts))
                }

    def on_step_free(_):
        global c

        if reg_params.display:
            mv[0][0].dynamic_meshes = [Mesh(v=free_verts.r, f=faces, vc=bone_vc)]
            mv[0][1].dynamic_meshes = [Mesh(v=free_verts.r, f=faces, vc=bone_vc)]
            # print(rot_angles, trans) snapshot


    print('Free vertices')
    if reg_params.visual_check : mv[0][0].get_mouseclick()
    if reg_params.missing_part:
        print("WARNING: Option for scan with missing parts is True")
        mesh2scan_dists = geo.distance_computation(free_verts.r, scan).r
        not_missing_mask = (mesh2scan_dists < reg_params.missing_part_dist).astype(np.bool)
        opt_var = [ff[not_missing_mask, :]]
    else:
        opt_var = [ff]
    ch.minimize(objs, x0=opt_var, method='dogleg', callback=on_step_free, options={'maxiter': reg_params.free_verts_max_iter})

   
    # Save results
    result = dict()
    result['v'] = free_verts.r
    result['rot'] = r.r
    result['trans'] = t.r
    result['betas'] = betas.r
    if reg_params.align_data_filename:
        dump(result, open(reg_params.align_data_filename, 'wb'))
    aligned_bone = Mesh(v=free_verts.r, f=faces)
    aligned_bone.write_ply(reg_params.align_filename)
    print("Registered posed bone ply saved as {}".format(reg_params.align_filename))

    s2m = ScanToMesh(scan, free_verts, faces,
               scan_sampler=scan_sampler,
               normalize=False)
    geo.print_dist_stats(s2m)

    return aligned_bone, template


def run_alignment(input):
    [pca_filename, scan_filename, num_betas, reg_params] = input

    if os.path.exists(reg_params.align_filename) and not reg_params.force_recompute:
        print('Skipping registration ({})'.format(reg_params.align_filename))
        align = Mesh(filename=reg_params.align_filename)
        template = Mesh(filename=reg_params.template_filename)
        # return
    else:
        print('Processing {}'.format(reg_params.title))
        align, template = align_bonepca2scan(pca_filename, scan_filename, num_betas, reg_params)

        if align is None:
            return

    # Now "unpose" the alignment (procrustes like) towards the template
    if os.path.exists(reg_params.unposed_filename) and not reg_params.force_recompute:
        print('Skipping unposing ({})'.format(reg_params.unposed_filename))
    else:
        print('Unposing {}'.format(reg_params.title))
        unposed_verts, R, T = rigid_alignment_analytic(template.v, align.v,)

        unposed = Mesh(v=unposed_verts, f=template.f)
        unposed.write_ply(reg_params.unposed_filename)
        print("Unposed bone ply saved as {}".format(reg_params.unposed_filename))

        result = dict()
        result['R'] = R
        result['T'] = T

        dump(result, open(reg_params.unposed_data_filename, 'wb'))



