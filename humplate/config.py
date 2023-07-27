
#################################
# Template and annotations
#################################
humerus_template = 'data/templates/humerus_t3_mean.ply' 
humerus_pca = 'data/templates/humerus_pca_init_smoothshells.pkl'
anatomical_annotation_mesh = 'data/templates/humerus_final_v1_anatomical_zones.ply'

template_annotation = 'data/templates/annotations.pkl' # Defines a painting of the bone template different parts
bone2plate_mapping = 'data/templates/B2P.pkl'
bone_template_annotated = 'data/templates/humerus_t2_mean_annotated.ply' # Bone with the fixation zones annotated 

plate_template_contoured = 'data/templates/plate_template_contoured.ply' # Plate template created from the humerus template surface
plate_template_contoured_annotated = 'data/templates/plate_template_contoured_annotated.ply' # Plate template created from the humerus template surface


#################################
# Example data to test with
#################################
# bone_sample_reg = 'data/H003_reg.ply'
bone_sample_scan = 'data/samples/0677_scan.ply'
bone_sample_reg = 'data/samples/0677_reg.ply'


#################################
# Plate extraction
#################################
plate_extraction_offset = 0.5 #mm
plate_smoothing_iteration = 1
plate_normal_smooting_iteration = 1

bone_id = "humerus"
template_mode = "smooth_shells"
experiment = "shape_v4_swarm" #"shape_test2"


#################################
# Plate fitting to a bone
#################################

# Swarm fit
parallelize = "no" # Should be in ["cluster", "local", "no"]
particle_range = 1 #mm translation +- this value
nb_sample = 1 # For the swarm fit
process_counts = 1 #Nb of processed when parallelizing
random_sample = False

opt_method = 'dogleg' # dogleg 'Nelder-Mead'
opt_final = 'dogleg'
close_form_init = True
push_out_init = False

do_first_opt = False
do_second_opt = True
do_third_opt = True


#################################
# Fit evaluation
#################################

#This Rate of plate points must be close enough to the bone
fit_thresh = 0.99

# Torelated intersection distance for the plate in the bone
intersection_dist = 0.2

#fit criteria constants
h_dist = 2 #mm
m_dist = 7 #mm
# m_dist_min = 3 #mm
t_dist = 2 #mm


#################################
# Other
#################################
output_folder = 'output'

blue = [0.4, 0.3, 0.8]
purple = [0.6, 0.3, 0.8]
green = [0.3, 0.8, 0.4]