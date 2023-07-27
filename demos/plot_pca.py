"""
Copyright©2023 Max-Planck-Gesellschaft zur Förderung
der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
for Intelligent Systems. All rights reserved.

Author: Sergi Pujades and Marilyn Keller
See LICENCE.txt for licensing and contact information.
"""

import os
from os.path import join
from os import makedirs
from itertools import product
from time import sleep
from math import pi
import numpy as np

from psbody.mesh import Mesh, MeshViewer
from psbody.mesh.sphere import Sphere
from psbody.mesh.colors import name_to_rgb

import humplate.config as cg
from humplate.utils.bone_tools import get_plane_frames_from_humerus
from humplate.utils.draw import plane_frames_2_mesh_lines
from humplate.utils.geometry import rotate
import humplate.templates.bone_annotation as ba

import argparse
import logging

logging.basicConfig(level=logging.INFO)


def _create_cage(vs, radius=1e-2):
    ''' Given an array of 3d vertices, it creates a *cage* containing all given vertices inside.
    The output are a set of spheres with the "corners" of the bounding box of the points.
    
    :param vs: an array of 3d points
    :returns: an array of eight spheres, creating a *cage*
    '''

    return([Sphere(np.asarray(corner), radius=radius).to_mesh()
            for corner in product(*zip(vs.min(axis=0), vs.max(axis=0)))])


def _create_cage_from_meshes(meshes, radius=1e-2):
    ''' Given an iterable with meshes, their vertices are concatenated
    and a cage is created using :func:`create_cage`.

    :param meshes: an iterable with meshes
    :type meshes: Mesh
    :returns: an array of eight spheres, creating a *cage*

    .. seealso:: :func:`create_cage`
    '''
    
    cage_v = np.zeros([0, 3])
    for m in meshes:
        cage_v = np.vstack((cage_v, m.v))

    return _create_cage(cage_v, radius=radius)


def visualize_pca_space(B, M, f, output_dir, annotation_filename='', heatmap=False, anatomical_annotation_mesh=False):
    '''
    Move across principal components and visualize
    :param B: Basis vectors
    :param M: Mean vector
    :param f: Faces of the template mesh
    :param output_dir: Folder containing the snapshots
    :return:
    '''
    print('Saving pca space visualization at ' + output_dir)

    if os.path.exists(output_dir):
        from shutil import rmtree
        rmtree(output_dir)
        # raise IOError('Output directory' + output_dir + ' already exists!')
    makedirs(output_dir)

    mv = MeshViewer()
    mv.set_background_color(np.asarray([1.0, 1.0, 1.0]))

    # camera orientations
    if not heatmap:
        camera_orientations = {
            'top': [pi / 2, 0, 0.], 
            'side': [pi, 0., pi], 
            'front': [0., pi / 2, 0.],
        }
    else :
        camera_orientations = {
            'side': [pi/2, 0, 0.], 
            'head': [pi/5, 0, 0], 
            'front': [0., pi / 2, 0.],
        }


    step = 0.2
    limits = [-2., 2.]
    colors = ['bisque', 'lavender', 'honeydew', 'gray']


    for ci, orientation in enumerate(camera_orientations):
        if not os.path.exists(os.path.join(output_dir, orientation)):
            makedirs(os.path.join(output_dir, orientation))

        mean_shape = Mesh(v=rotate(M, camera_orientations[orientation]))
        img_num = 0

        # iterate over pca components
        for b in range(3):  # B.shape[0]):
            mv.set_background_color(name_to_rgb[colors[b % len(colors)]])

            # collect all the meshes to generate a cage (to avoid zoom effects in the viewer)
            meshes = []
            lines = []
            for k in (list(np.arange(limits[0], limits[1], step)) + list(np.arange(limits[1], limits[0], -step))) :
                m = Mesh(v=M + k * B[:, :, b], f=f, vc=[0.7, 0.7, 0.7])

                if anatomical_annotation_mesh:
                    m.vc=anatomical_annotation_mesh.vc
                if annotation_filename and args.heatmap:
                    annotated_points = ba.apply_annotations(m, annotation_filename)
                    _, rms_dico = ba.compute_local_frame_lines(annotated_points)
                    if heatmap :
                        ba.annotate_heatmap(m, annotated_points, rms_dico)
                    plane_dict = get_plane_frames_from_humerus(annotated_points, m)
                    mesh_lines = plane_frames_2_mesh_lines(plane_dict)

                    for l in mesh_lines:
                        l.v = rotate(l.v, camera_orientations[orientation])
                    lines.append(mesh_lines)
                m.v = rotate(m.v, camera_orientations[orientation])
                meshes.append(m)


            mv.static_meshes = _create_cage_from_meshes(meshes, radius=0.0005)

            # iterate over steps of std dev
            for i, m in enumerate(meshes):
                if not heatmap:
                    mv.dynamic_meshes = [m] + [mean_shape]
                else:
                    mv.dynamic_meshes = [m]
                if len(lines)==len(meshes):
                    mv.set_dynamic_lines(lines[i])
                sleep(0.02)

                mv.save_snapshot(blocking=True, path=join(output_dir, orientation, '{n:03d}.png'.format(n=img_num)))
                img_num += 1

        # Generate a video from the saved frames
        cmd_orientation = 'ffmpeg  -i {1}/%03d.png -vcodec h264 -pix_fmt yuv420p -r 30 -an -b:v 5000k {2}.mp4'.format(20, join(output_dir, orientation), join(output_dir, orientation))
        print(cmd_orientation)

        cmd = ['ffmpeg', '-framerate', '20', '-i', '{0}/%03d.png'.format(join(output_dir, orientation)), '-vcodec',
               'h264', '-pix_fmt', 'yuv420p', '-r', '30', '-an', '-b:v', '5000k', '{0}.mp4'.format(join(output_dir, orientation))]

        print(cmd)
        from subprocess import call
        call(cmd)



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--bone_id", help="the bone ID to work with", default='humerus', type=str)
    parser.add_argument("-H", "--heatmap", action='store_true', help='Fit a plate to the fixation area and show the distance to this plane')
    parser.add_argument("-A", "--anat", action='store_true', help='Paint the meshes with the anatomical zones anotations')

    args = parser.parse_args()

    output_filename = cg.humerus_template
    pca_filename = cg.humerus_pca  

    # Load the PCA
    from pickle import load
    pca_dict = load(open(pca_filename, 'rb'), encoding='latin1')
    B = pca_dict['B']
    M = pca_dict['M']

    faces = Mesh(filename=output_filename).f

    video_output_dir = os.path.join(cg.output_folder, 'pca_video')
    makedirs(video_output_dir, exist_ok=True)

    if args.heatmap:
        video_output_dir = os.path.join(video_output_dir, 'planes_heatmap')
        anatomical_annotation_mesh = False
    elif args.anat:
        anatomical_annotation_mesh = Mesh(filename=cg.anatomical_annotation_mesh)
        video_output_dir = os.path.join(video_output_dir, 'anatomic')
    else :
        video_output_dir = os.path.join(video_output_dir, 'pca_space')
        anatomical_annotation_mesh = False

    makedirs(video_output_dir, exist_ok=True)

    visualize_pca_space(B, M, faces, output_dir=video_output_dir, annotation_filename=cg.template_annotation,
                        heatmap=args.heatmap, anatomical_annotation_mesh=anatomical_annotation_mesh)
    
    print('Videos saved at ' + video_output_dir)