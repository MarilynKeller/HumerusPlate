from psbody.mesh import Mesh, MeshViewers
import humplate.data_paths as bd
import humplate.config as cg
import humplate.data_paths as bd
import tqdm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pickle as pkl
import os
from copy import deepcopy

import humplate.config as cg
from humplate.fit.fit_plate_rigid import run_fit_plate2bone
from humplate.utils.plate_tools import closed_form_plate_position

    
def run_fit_plate2bone_swarm_with_args(dico):
    swarm_index     = dico["swarm_index"]
    plate_id        = dico["plate_id"]
    vol_id          = dico["vol_id"]
    plate_mesh      = dico["plate_mesh"]
    vol_mesh        = dico["vol_mesh"]
    display         = dico["display"]
    visual_check    = dico["visual_check"]
    force_recompute = dico["force_recompute"]
    rot_vect        = dico["rot_vect"]
    trans_init      = dico["trans_init"]

    run_fit_plate2bone(plate_id=plate_id,
                        vol_id=vol_id,
                        plate_mesh=plate_mesh,
                        vol_mesh=vol_mesh,
                        swarm_index = swarm_index,
                        display=display,
                        visual_check=visual_check,
                        force_recompute=force_recompute,
                        rot_vect=rot_vect,
                        trans_init=trans_init)



def run_swarm_fit_plate2bone(plate_id, vol_id, plate_mesh, vol_mesh, nb_sample = 3,
                             display=False, visual_check=False, parallelize="cluster",
                             force_recompute=True, rot_vect=np.array([0, 0, 0.0]), trans_init=np.zeros(3), pool=None):
    """
    Fit a plate to a bone rigidly
    @param plate_id: id of the plate to fit (the id is used to store the results)
    @param vol_id: id of the bone to fit the plate to
    @param plate_mesh: mesh of the plate to fit
    @param vol_mesh: mesh of the bone to fit the plate to
    @param experiment: plate shape to consider for the ideal plate placement
    @param display: display the fitting process in meshviewer
    @param visual_check: pause display after each optimisation step
    @param force_recompute: recompute the fits for the bones that already have a plate-distance file saved
    @param : A multiprocessing pool, so that we don't need to recreate one each time this function is called.
        import multiprocessing
        pool = multiprocessing.Pool(processes=cg.process_counts)
    @return: Nb of particles

"""

    swarm_fit_folder = bd.get_plate_bone_fit_swarm_folder(vol_id, plate_id)

    if force_recompute:
        print("WARNING: Removing all swarm fits for recomputation fot plate {} on bone {} ...".format(plate_id, vol_id))
        os.system('rm -rf {}'.format(swarm_fit_folder))
        fit_file = bd.get_plate_bone_fit(vol_id, plate_id)
        os.system('rm -f {}'.format(fit_file))
    if not os.path.exists(swarm_fit_folder):
        os.makedirs(swarm_fit_folder)

    assert(parallelize in ["cluster", "local", "no"])
    vol_mesh = deepcopy(vol_mesh)
    
    # Initial placement
    _, trans_0, rot_vect = closed_form_plate_position(plate_mesh, vol_mesh)

    # Sample init transform space
    nb_sample = nb_sample
    dd = cg.particle_range #mm
    trans_space = []
    if cg.random_sample:
        import random
        random.seed(0)
        for y in range(nb_sample):
            for z in range(nb_sample):
                dy = random.uniform(-dd, dd)
                dz = random.uniform(-dd, dd)
                trans_space.append([trans_0[0], trans_0[1]+dy, trans_0[2]+dz])
    else :
        import itertools
        x_range = np.linspace(-dd, dd, nb_sample)
        y_range = np.linspace(-dd, dd, nb_sample)
        dd_space = list(itertools.product(x_range, y_range))
        trans_space.append([trans_0[0], trans_0[1], trans_0[2]])
        for (dy, dz) in dd_space:
            trans_space.append([trans_0[0], trans_0[1] + dy, trans_0[2] + dz])
    # print trans_space

    # Run fit for each particle in parallel
    dico_list = []
    for i, trans in enumerate(trans_space):

        dico = {}
        dico["swarm_index"] = i
        dico["plate_id"] = plate_id
        dico["vol_id"] = vol_id
        dico["plate_mesh"] = plate_mesh
        dico["vol_mesh"] = vol_mesh
        dico["display"] = display
        dico["visual_check"] = visual_check
        dico["force_recompute"] = force_recompute
        dico["rot_vect"] = rot_vect
        dico["trans_init"] = trans

        dico_list.append(dico)

    # Save input data
    swarm_input_file = bd.get_plate_bone_fit_swarm_input(vol_id, plate_id)
    os.makedirs(os.path.dirname(swarm_input_file), exist_ok=True)
    pkl.dump(dico_list, open(swarm_input_file,"wb"))

    print("Running swarm fit for plate {} on bone {} ...".format(plate_id, vol_id))
    if parallelize=="local" :
        import multiprocessing
        if pool is None:
            pool = multiprocessing.Pool(processes=cg.process_counts)
        pool.map(run_fit_plate2bone_swarm_with_args, dico_list)
    else :
        for dico in dico_list:
            run_fit_plate2bone_swarm_with_args(dico)

    best_fit_data, particle_fit_rate = load_swarm_fit_result(vol_id, plate_id)

    print("best_fit_rate = {}".format(best_fit_data))
    print("particle_fit_rate = {}".format(particle_fit_rate))

    fit_file = bd.get_plate_bone_fit(vol_id, plate_id)
    best_fit_data.paticle_fit_rate = particle_fit_rate
    pkl.dump(best_fit_data, open(fit_file, "wb"))
    print("plate to bone fit info saved at {}".format(fit_file))

    return best_fit_data


def load_swarm_fit_result(vol_id, plate_id, debug_disp=False):
    swarm_fit_folder = bd.get_plate_bone_fit_swarm_folder(vol_id, plate_id)
    swarm_fit_files = os.listdir(swarm_fit_folder)

    # swarm_input_file = pd.get_plate_bone_fit_swarm_input(vol_id, plate_id, experiment)
    # swarm_input_list = pkl.load(open(swarm_input_file, "rb"))

    best_fit_data = None
    best_fit_score = -1
    fit_rates = []
    is_fitting_nb = 0

    if debug_disp: print ('Load fit results ...')
    for i, fit_file in enumerate(swarm_fit_files):

        # load swarm fit data
        # fit_file = pd.get_plate_bone_fit(vol_id, plate_id, experiment, swarm_index=i)
        with open(os.path.join(swarm_fit_folder, fit_file), 'rb') as f:
            fit_data = pkl.load(f)
        fit_score = fit_data.get_fit_score()
        if debug_disp: print("   {}: {}".format(i, fit_score))

        #update fit data
        if fit_score > best_fit_score:
            best_fit_score = fit_score
            best_fit_data = fit_data

        fit_rates.append(fit_data)
        is_fitting_nb += fit_data.do_fit(cg.fit_thresh)

    particle_fit_rate = is_fitting_nb*1.0/len(swarm_fit_files)
    if debug_disp: print("result: fit rate {}   fit score:{}".format(particle_fit_rate, best_fit_data.get_fit_score()))

    assert(best_fit_data is not None)
    return best_fit_data, particle_fit_rate

def plot_swarm_fit_results(vol_id, plate_id):
    swarm_fit_folder = bd.get_plate_bone_fit_swarm_folder(vol_id, plate_id)
    os.makedirs(swarm_fit_folder, exist_ok=True)
    swarm_fit_files = os.listdir(swarm_fit_folder)
    swarm_input_file = bd.get_plate_bone_fit_swarm_input(vol_id, plate_id)
    with open(swarm_input_file,"rb") as f:
        swarm_input = pkl.load(f)

    # swarm_input_file = pd.get_plate_bone_fit_swarm_input(vol_id, plate_id, experiment)
    # swarm_input_list = pkl.load(open(swarm_input_file, "rb"))

    best_fit_rate = -1
    best_fit_data = None
    init_pos = [dico["trans_init"] for dico in swarm_input]
    plate_pos = []
    costs = []
    fit_rates = []
    is_fitting_nb = 0

    print ('Load fit results ...')
    for i, fit_file in tqdm.tqdm(enumerate(swarm_fit_files)):
        fit_file = bd.get_plate_bone_fit(vol_id, plate_id, swarm_index=i)

        with open(fit_file, 'rb') as f:
            fit_data = pkl.load(f)
        plate_pos.append(fit_data.T.r)
        costs.append(fit_data.cost)
        fit_rates.append(fit_data.get_fit_score())

    _draw_swarm_fit_results(plate_init_pos_list = init_pos, plate_final_pos_list=plate_pos, cost_list=costs, fit_rate_list=fit_rates)

def _draw_swarm_fit_results(plate_init_pos_list, plate_final_pos_list, cost_list, fit_rate_list):
    

    xlabel_init = "Initial plate translation along y"
    ylabel_init = "Initial plate translation along z"
    xlabel_final = "Final plate translation along y"
    ylabel_final = "Final plate translation along z"
    x0 = np.array([t[1] for t in plate_init_pos_list])
    y0 = np.array([t[2] for t in plate_init_pos_list])
    xf = np.array([t[1] for t in plate_final_pos_list])
    yf = np.array([t[2] for t in plate_final_pos_list])

    fit_rates = np.array(fit_rate_list)
    fit_mask = (fit_rates >= cg.fit_thresh)

    fig, axs = plt.subplots(1, 3, figsize=(40, 10))

    ax = axs[0]
    sc = ax.scatter(x0, y0, c=fit_rate_list, s=200)  # row=0, col=0
    ax.set_title("Fit rate")
    ax.set_xlabel(xlabel_init)
    ax.set_ylabel(ylabel_init)
    fig.colorbar(sc, ax=ax)
    
    #print the ones that fit in orange
    ax.scatter(x0[fit_mask], y0[fit_mask], c='orange', s=200)

    ax = axs[1]
    sc = ax.scatter(xf, yf, c=fit_rate_list, s=200)  # row=0, col=0
    ax.set_title("Fit rate")
    ax.set_xlabel(xlabel_final)
    ax.set_ylabel(ylabel_final)
    fig.colorbar(sc, ax=ax)
    
    #print the ones that fit in orange
    ax.scatter(xf[fit_mask], yf[fit_mask], c='orange', s=200)

    ax = axs[2]
    eps = 1e-4
    final_costs_plot = np.array(cost_list) + eps  # np.clip(final_costs, 0, cost_thresh)
    sc = ax.scatter(xf, yf, c=final_costs_plot, s=200,
                       norm=matplotlib.colors.LogNorm())  # row=1, col=0 , norm=matplotlib.colors.LogNorm()
    ax.set_title("Cost")
    ax.set_xlabel(xlabel_final)
    ax.set_ylabel(ylabel_final)
    fig.colorbar(sc, ax=ax)

    plt.show()
    pass

