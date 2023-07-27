"""
Copyright©2023 Max-Planck-Gesellschaft zur Förderung
der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
for Intelligent Systems. All rights reserved.

Author: Marilyn Keller
See LICENCE.txt for licensing and contact information.
"""

import humplate.config as cg
from os.path import join
import os


""" Create the path @directory if it does not exists and return @directory"""
def rmakedir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory

# Bone registration

def get_registrations_folder():
    return rmakedir(join(cg.output_folder, 'registrations'))

def get_align_filename(scan_id):
    return join(get_registrations_folder(), scan_id + '.ply')

def get_align_data_filename(scan_id):
    return get_align_filename(scan_id).replace('.ply', '.pkl')

def get_unposed_filename(scan_id):
    return get_align_filename(scan_id).replace('.ply', get_unposed_suffix())

def get_unposed_data_filename(scan_id):
    return get_unposed_filename(scan_id).replace('.ply', '.pkl')

def get_unposed_suffix():
    return '_reg.ply'


# Plate fit


def get_plate_bone_fit_swarm_folder(vol_id, plate_id):
    return join(cg.output_folder, "fits", vol_id, plate_id, "swarm")

def get_plate_bone_fit_swarm_input(vol_id, plate_id):
    return join(cg.output_folder, "fits", vol_id, plate_id, "swarm_inputs.pkl")

def get_plate_bone_fit(vol_id, plate_id, swarm_index=None):
    if swarm_index is None:
        return join(cg.output_folder, "fits", vol_id, plate_id, plate_id + ".pkl")
    else:
        return join(get_plate_bone_fit_swarm_folder(vol_id, plate_id), str(swarm_index) +".pkl")

def get_plate_on_bone_snapshot(plate_id, vol_id):
    folder = join(cg.output_folder, "fit", "snapshots", "plate_on_bone_view")
    path = join(folder, "{}_on_{}.png".format(plate_id, vol_id))
    return path

def get_plate_posed_ply(vol_id, plate_id):
    return join(cg.output_folder, "fits", vol_id, plate_id + ".ply")
    




#