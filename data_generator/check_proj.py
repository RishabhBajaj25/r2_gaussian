import os
import os.path as osp
import numpy as np
import sys
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")  # Force it to use the Qt5 GUI backend

sys.path.append("./")
from r2_gaussian.utils.plot_utils import show_one_volume

# proj_path = "data/real_dataset/seashell/train"
# proj_path = "/home/rishabh/projects/r2_gaussian/data/real_dataset/pine/proj_train"
proj_path = "/media/rishabh/SSD_1/Data/UTokyo/cone_ntrain_25_angle_360/CR_20250618_170040_/proj_test"
proj_list = sorted(os.listdir(proj_path))

projs = np.stack(
    [np.load(osp.join(proj_path, proj_id)) for proj_id in proj_list], axis=-1
)

show_one_volume(projs)
