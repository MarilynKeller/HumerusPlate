"""
Copyright©2023 Max-Planck-Gesellschaft zur Förderung
der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
for Intelligent Systems. All rights reserved.

Author: Marilyn Keller
See LICENCE.txt for licensing and contact information.

Command line tool to register a template bone to a bone scan using a bone pca model
"""

import humplate.data_paths as bd
import numpy as np
import humplate.utils.bone_tools as geo
from humplate.registration import registration_pca as br
import humplate.config as cg
import os 

############################################### Align registration and Gradishar

def register_bone(scan_filename, is_right_side, vol_id, num_betas, init_rot,  display, visual_check):

    # Set parameters for the registration
    rp = br.RegParams()
    rp.rot = np.array(init_rot)
    rp.symmetrize = is_right_side
    rp.template_filename = cg.humerus_template
    rp.max_iter = 50
    rp.free_verts_max_iter = 25
    rp.missing_part = False
    rp.is_scan_clean = False
    rp.laplace_cpl_weight = 0.02
    rp.weight_beta_norm = 2
    rp.gmo_sigma_beta = 50
    rp.gmo_sigma_ff = 5
    rp.num_samples_max = 5e3

    # Visualization
    rp.display = display
    rp.visual_check = visual_check
    rp.title = scan_filename

    # output paths
    rp.align_filename = bd.get_align_filename(vol_id)
    rp.align_data_filename = bd.get_align_data_filename(vol_id)
    rp.unposed_filename = bd.get_unposed_filename(vol_id)
    rp.unposed_data_filename = bd.get_unposed_data_filename(vol_id)

    pca_filename = cg.humerus_pca
    print('Using pca: {}'.format(pca_filename))
    print('Using laplace coupling: {}'.format(rp.laplace_coupling))

    br.run_alignment([pca_filename, scan_filename, num_betas, rp])

    print(f'Registration saved in {rp.align_filename}')
    
if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description='Command line tool to register a template bone to a bone scan using a bone pca')
    parser.add_argument("--scan_path", help="Path to the bone scan to register", type=str, required=True)
    
    parser.add_argument( "--beta_nb", help="Number of components to use for the pca registration", type=int, default=25)
    parser.add_argument( "--init_rot", help="3d vector giving initial rotation to the template to align it with the bone. "
                                            "Example : --init_rot 0 1.7 0", type=float, nargs='+', default=[0.,0,0])
    parser.add_argument("-s","--symmetrize", help="Set to true if the bone scan is the right side bone", action="store_true")
    parser.add_argument("-D", "--display", help="Display registration", action="store_true")
    parser.add_argument("-V", "--visual_check", help="Pause between step", action="store_true")

    args = parser.parse_args()

    assert(len(args.init_rot)==3) #should be 3d rotation vector

    scan_id = os.path.basename(args.scan_path).split("_")[0] # Name of the bone, for saving the registration outcome
    register_bone(args.scan_path, args.symmetrize, scan_id, args.beta_nb, args.init_rot, args.display, args.visual_check)

    if args.symmetrize:
        geo.save_right_side_ply(path=bd.get_align_filename(scan_id))
