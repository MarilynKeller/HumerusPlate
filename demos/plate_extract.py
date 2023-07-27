"""
Copyright©2023 Max-Planck-Gesellschaft zur Förderung
der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
for Intelligent Systems. All rights reserved.

Author: Marilyn Keller and Sergi Pujades
See LICENCE.txt for licensing and contact information.
"""

import os 
import argparse
import numpy as np
from psbody.mesh import Mesh, MeshViewers
import humplate.config as cg
from humplate.templates.plate_extraction import plate_extract

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Extract a custom plate from a registered bone')
    parser.add_argument("-b", "--bone_path", help="Path to the registered bone mesh", type=str, default=cg.bone_sample_reg)
    
    args = parser.parse_args()

    # Humerus template registered to a subject's scan
    bone_reg = Mesh(filename=args.bone_path)    
    bone_id = os.path.basename(args.bone_path).split("_")[0]
    
    # Check that the input mesh is loaded and has the proper topology
    assert bone_reg is not None, f"Could not load the input mesh {args.bone_path}.ply"
    bone_template = Mesh(filename=cg.humerus_template)
    if bone_reg.v.shape[0] != bone_template.v.shape[0]:
        raise ValueError(f"The input mesh {bone_id} has a different topology than the template ({bone_reg.v.shape[0]} vs {bone_template.v.shape[0]} vertices).\n Please register your bone mesh by running demos/register_bone.py --scan_path {args.bone_path} -D.")

    extracted_plate = plate_extract(bone_reg)
    
    # Display the plate 
    mv = MeshViewers((1, 2), keepalive=True, titlebar="bone scan - registered bone with extracted custom plate - bone scan with extracted custom plate")
    
    extracted_plate.set_vertex_colors(cg.blue)
    bone_reg.set_vertex_colors(cg.green)
    
    mv[0][0].set_background_color(1.*np.ones(3))
    mv[0][0].set_static_meshes([bone_reg])
    mv[0][1].set_static_meshes([bone_reg, extracted_plate])
    
    # Save the plate 
    dest_file = f'output/plate_extraction/{bone_id}_extracted_plate.ply'
    os.makedirs(os.path.dirname(dest_file), exist_ok=True)
    extracted_plate.write_ply(dest_file)
    print(f"Generated custom plate saved as {dest_file}")