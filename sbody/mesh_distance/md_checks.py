import numpy as n
import body.alignment.mesh_distance
import scipy.sparse
import os.path

from psbody.mesh import Mesh
from body import body_folder


# setup: scan, alignment, sample_spec
scan = Mesh()
scan.load_from_file(os.path.join(body_folder, 'stanford/RawScans/Adina1.ply'))
scan.f = scan.f.astype(n.int64)

alignment = Mesh()
alignment.load_from_file(os.path.join(body_folder, 'coalignment_dhirshberg/la_fixed/15/alignments/train/a/StanfordAdina1.ply'))
alignment.f = alignment.f.astype(n.int64)

sample_type = 'uniformly-at-random' if hasattr(scan, 'f') else 'uniformly-from-vertices'
sample_spec=body.alignment.objectives.sample_from_mesh(scan, sample_type=sample_type, num_samples=1e5, seed=0)

S = sample_spec['point2sample'].shape[0]/3
js = n.arange(3*S)
Dr_samplev = scipy.sparse.coo_matrix((js+1, (n.floor_divide(js,3), js)), (S,3*S)).tocsc()
sample_spec['dsample_pattern'] = dict(indices=Dr_samplev.indices, indptr=Dr_samplev.indptr, sortdata=Dr_samplev.data-1)


def compute(mesh_distance_module, ref=True, sample=True):
    s2m=mesh_distance_module.SampleMeshDistanceSquared(scan, sample_spec, alignment)
    s2m._setup_for_derivative_computation()
    r=s2m.r.flatten()
    dref = s2m.dr_reference_mesh if ref else None
    dsample = s2m.dr_sample_mesh if sample else None
    return r, dref, dsample


def compare((r,dr,ds),(ro,dro,dso)):
    return n.array([n.abs(r-ro).sum(), n.abs(dr-dro).sum(), n.abs(ds-dso).sum()])


def profile(code):
    import cProfile as profile
    import pstats
    profile.runctx(code, globals(), locals(), 's2m.prof')
    p = pstats.Stats('s2m.prof')
    p.strip_dirs().sort_stats(1).print_stats()

if __name__ == '__main__':
    #print 'sum of abs differences between old and new r, Dr_reference_mesh, and Dr_sample_points'
    #print compare(compute(body.alignment.mesh_distance), compute(body.alignment.mesh_distance_lazy))
    #import body.scape.train_theano.util as u
    #abc=u.timethunk(lambda:compute(body.alignment.mesh_distance, ref=True, sample=True))
    #print 'profiles'
    #print 's2m fast'; profile('compute(body.alignment.mesh_distance, ref=True, sample=False)')
    #print 's2m slow'; profile('compute(body.alignment.mesh_distance_lazy, ref=True, sample=False)')
    print 'm2s fast'; profile('compute(body.alignment.mesh_distance, ref=False, sample=True)')
    #print 'm2s slow'; profile('compute(body.alignment.mesh_distance_lazy, ref=False, sample=True)')


