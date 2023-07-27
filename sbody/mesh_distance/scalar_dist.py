# explores computing f=||r||^2 and Df without first computing r and Dr
# this saves a lot of allocation and bookkeeping -- tends to be much faster than working with r and Dr

import numpy as n
import body
import body.alignment.mesh_distance
from psbody.mesh import Mesh

# setup: scan, alignment, sample_spec
scan = Mesh()
scan.load_from_file('/is/ps/shared/data/body/stanford/RawScans/Adina1.ply')
scan.f=scan.f.astype(n.int64)

alignment = Mesh()
alignment.load_from_file('/is/ps/shared/data/body/coalignment_dhirshberg/la_fixed/15/alignments/train/a/StanfordAdina1.ply')
alignment.f = alignment.f.astype(n.int64)

sample_type = 'uniformly-at-random' if hasattr(scan, 'f') else 'uniformly-from-vertices'
sample_spec=body.alignment.objectives.sample_from_mesh(scan, sample_type=sample_type, num_samples=1e6, seed=0)

v=alignment.v
samples=sample_spec['point2sample'].dot(scan.v.flatten()).reshape(-1,3)
sigma=10#5e-3

import body.scape.train_theano.util as u
import sample2meshdist, mesh_distance

def newscalar(v, samples):
    s2m=body.alignment.mesh_distance.SampleMeshDistanceSquared(scan, sample_spec, alignment)
    return sample2meshdist.squared_distance_scalar(s2m.nearest_tri, s2m.nearest_part, s2m.reference_mesh.f, v, samples)

def gmscalar(v, samples, sigma):
    s2m=body.alignment.mesh_distance.SampleMeshDistanceSquared(scan, sample_spec, alignment)
    return sample2meshdist.gm_distance_scalar(sigma, s2m.nearest_tri, s2m.nearest_part, s2m.reference_mesh.f, v, samples)
    
def scalar_test():
    s2m=body.alignment.mesh_distance.SampleMeshDistanceSquared(scan, sample_spec, alignment)
    r, Dr_ref, Dr_sample = sample2meshdist.distance(s2m.nearest_tri, s2m.nearest_part, s2m.reference_mesh.f, v, samples)
    vf,vgr,vgs = n.dot(r,r), 2*Dr_ref.T.dot(r), 2*Dr_sample.T.dot(r)
    f,gr,gs = newscalar(v, samples)
    print vf-f
    print n.max(n.abs(vgr-gr))
    print n.max(n.abs(vgs-gs))
    print u.check_derivative(lambda v: newscalar(v, samples), v, 0) 
    print u.check_derivative(lambda samples: newscalar(v, samples), samples, 1) 

def gm_test():
    s2m=body.alignment.mesh_distance.SampleMeshDistanceSquared(scan, sample_spec, alignment)
    distsq, _, _ = sample2meshdist.squared_distance(s2m.nearest_tri, s2m.nearest_part, s2m.reference_mesh.f, v, samples)
    rho_v = sigma**2 * distsq / (sigma**2 + distsq)
    gmdist,_,_ = gmscalar(v,samples,sigma)
    print gmdist - n.sum(rho_v)
    print u.check_derivative(lambda v: gmscalar(v, samples, sigma), v, 0) 
    print u.check_derivative(lambda samples: gmscalar(v, samples, sigma), samples, 1) 

def profile(code):
    import cProfile as profile
    import pstats
    profile.runctx(code, globals(), locals(), 's2m.prof')
    p = pstats.Stats('s2m.prof')
    p.strip_dirs().sort_stats(1).print_stats()

if __name__ == '__main__':
    scalar_test()
    gm_test()
    profile('newscalar(v,samples)')
    profile('gmscalar(v,samples,sigma)')
    

