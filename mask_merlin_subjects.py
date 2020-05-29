# script to process a list of the merlin subjects and masks, apply them all
from nilearn.image import load_img
from nilearn.datasets import load_mni152_template
from nilearn.image import resample_to_img
from nilearn.masking import apply_mask
import numpy as np
from pathlib import Path
from os import listdir
from os import walk

# fn to load a subject's data
def load_sub_data(sub):
    directory_string = "/om2/user/jsmentch/neuroscout/SherlockMerlin/preproc/fmriprep/" + sub + "/func/"
    nifti_file = "_task-MerlinMovie_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"
    #load subject's functional data
    nii_file = directory_string + sub + nifti_file
    data_MNI152NLin2009cAsym = load_img(nii_file) # load func img
    data = resample_to_img(data_MNI152NLin2009cAsym, template)
    return data

# fn to apply a mask to a nii file
def apply_roi_mask(data, mask, template):
    #load masks of auditory region
    roi_dir = "/om2/user/jsmentch/data/rois/svnh_ROIS_anatlabels_surf_mni/mni152_te11-te10-te12-pt-pp/"
    roi_mask = load_img(roi_dir + mask)
    #Mask
    data_masked = apply_mask(data, roi_mask) #apply brain mask to data
    return data_masked

template = load_mni152_template() # load mni152 template

subject_list = list(walk("/om2/user/jsmentch/neuroscout/out/wl8RX/fitlins"))[0][1][:-1]
roi_list = listdir("/om2/user/jsmentch/data/rois/svnh_ROIS_anatlabels_surf_mni/mni152_te11-te10-te12-pt-pp")[:-3] #exclude lh, rh, lh_rh

for sub in subject_list:
    data = load_sub_data(sub)
    for mask in roi_list:
        data_masked = apply_roi_mask(data, mask, template)
        out_dir = "/om2/user/jsmentch/data/" + sub + "/"
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        save_name = out_dir + sub + "_" + mask.split('.')[0] + "_" + mask.split('.')[1]
        np.save(save_name, data_masked)
        
