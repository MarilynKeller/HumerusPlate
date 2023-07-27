
"""
Copyright©2023 Max-Planck-Gesellschaft zur Förderung
der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
for Intelligent Systems. All rights reserved.

Author: Marilyn Keller
See LICENCE.txt for licensing and contact information.
"""

"""Show the humerus and plate templates with the fixation zones"""

from psbody.mesh import Mesh, MeshViewers
from humplate.templates.plate_annotation import PlateAnnotation
from humplate.templates.bone_annotation import BoneAnnotation
import numpy as np
    
if __name__=='__main__':

    mvs = MeshViewers((1,2))
    mvs[0][0].set_background_color(np.array([1,1,1]))

    pa = PlateAnnotation()
    ba = BoneAnnotation()

    mvs[0][0].set_titlebar("annotated parts")
    pa.annotated_mean_mesh()
    # mvs[0][0].set_static_meshes([pa.annotated_mean_mesh(), ba.annotated_mean_mesh()])
    mvs[0][0].set_static_meshes([ba.annotated_mean_mesh()])
    
    mvs[0][1].set_static_meshes([pa.annotated_mean_mesh(), Mesh(v=ba.annotated_mean_mesh().v)])
    a = mvs[0][0].get_mouseclick()
