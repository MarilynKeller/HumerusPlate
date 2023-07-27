"""
Copyright©2023 Max-Planck-Gesellschaft zur Förderung
der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
for Intelligent Systems. All rights reserved.

Author: Marilyn Keller
See LICENCE.txt for licensing and contact information.
"""

import argparse
import os
import numpy as np
import humplate.config as cg
import humplate.data_paths as bd
from psbody.mesh import Mesh, MeshViewers

from humplate.fit.fit_plate_rigid_swarm import load_swarm_fit_result, plot_swarm_fit_results, run_swarm_fit_plate2bone

plate_choices = ['s0', 's1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 'demo']

def load_plate(name='s1'):
    assert name in plate_choices
    
    if name == 'demo':
        plate_mesh = Mesh(filename = cg.demo_plate)
    else:
        plate_mesh = Mesh(filename = f"data/plate_set/{name}.ply")
    return plate_mesh


if __name__=="__main__":
    
    parser = argparse.ArgumentParser(description='Fit a plate to a bone')
    parser.add_argument('-p', '--plate', type=str, default='s1', help='Name of the plate to position on the bone', choices=plate_choices)
    parser.add_argument('-b', '--bone', type=str, default=cg.bone_sample_reg, help='Path to a target registered bone')
    parser.add_argument('-D', '--display', action='store_true', help='Display the ongoing plate fitting optimization')
    parser.add_argument('-F', '--force', action='store_true', help='Force recomputing the fit')
    parser.add_argument('--plot_swarm_results', action='store_true', help='Plot the fit results of each particle')
    
    args = parser.parse_args()

    display = args.display
    parallelize = cg.parallelize
    nb_sample = cg.nb_sample
    force_recompute = True
    visual_check = False

    assert (parallelize in ["cluster", "local", "no"])
    fit_thresh = cg.fit_thresh

    # Select plate bone pair
    plate_id = args.plate
    vol_id = os.path.basename(args.bone).split("_")[0]
    vol_mesh = Mesh(filename=cg.bone_sample_reg)
    plate_mesh = load_plate(name=args.plate)

    fit_result_path = bd.get_plate_bone_fit(vol_id, plate_id)
    if not os.path.exists(fit_result_path) or args.force:
        # run parallel swarm fit
        fit_data = run_swarm_fit_plate2bone(plate_id, vol_id, plate_mesh, vol_mesh, nb_sample=nb_sample, display=display,
                                     visual_check=visual_check, force_recompute=True, parallelize=parallelize)

    # Load the best found position
    best_fit_data, particle_fit_rate = load_swarm_fit_result(vol_id, plate_id)
    
    # print results
    print(f"Optimal plate rotation and translation for {plate_id} on {vol_id} stored at {fit_result_path}")
    print("best_fit_rate = {}".format(best_fit_data))
    print("particle_fit_rate = {}".format(particle_fit_rate))

    if args.plot_swarm_results:
        plot_swarm_fit_results(vol_id, plate_id)

    # Transform the plate with the best fit transformation found
    plate_mesh_posed = Mesh(plate_mesh.v, plate_mesh.f)
    R, T = best_fit_data.R, best_fit_data.T
    plate_mesh_posed.v = np.dot(plate_mesh.v, R.T) + T
    
    # Visualize the transformed plate on the bone
    plate_mesh.vc = cg.purple
    plate_mesh_posed.vc = cg.purple
    vol_mesh.vc = cg.green
    mv = MeshViewers((1,2),window_width=800, window_height=600, titlebar='Left:Unposed plate - Right:Posed plate on bone')
    mv[0][0].set_background_color(1.*np.ones(3))
    mv[0][0].set_static_meshes([plate_mesh, vol_mesh])
    mv[0][1].set_static_meshes([plate_mesh_posed, vol_mesh])
    print('Press a key to exit')
    mv[0][0].get_keypress()
    
    