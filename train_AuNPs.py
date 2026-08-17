from argparse import Namespace

import numpy as np
from pathlib import Path

import train
from options.option import BaseOptions


if __name__ == '__main__':
    args = BaseOptions().parser.parse_args([])

    # base_dir = Path('/groups/scicompsoft/home/raridenm/projects/cryoet/au_np/SampleDatasets/EMPIAR_10045/')
    base_dir = Path('/Users/raridenm/projects/cryoet/data/tmp/data/20241206_preprocessed/data')

    CONFIG = {
        "act": "relu",
        "base_path": str(base_dir),
        "batch_size": 8,
        "block_size": 72,
        "checkpoints": None,
        "class_weights": [1.0, 1.0, 2.0],
        "coord_format": ".coords",
        "coord_path": f"{base_dir}/coords",
        # Six positional values, always:
        #   [train_start, train_end, val_start, val_end, test_start, test_end]
        # Ranges are np.arange(start, end), so end is exclusive. A shorter list
        # raises IndexError when the train dataset is built.
        # Full set (~370 GB, needs gpu_a100 -n 12): [0, 16, 15, 16, 15, 16]
        # Just tomogram 0 (~43 GB): train on 0, validate on 0.
        "data_split": [0, 14, 14, 15, 15, 16],
        "de_dup_fmt": "fmt4",
        "de_duplication": False,
        "dset_name": "3rd_train",
        "f_maps": [24, 48, 72, 108],
        "final_double": False,
        "gpu_id": [0],
        "in_channels": 1,
        "input_cat": False,
        "input_cat_items": "None",
        "label_diameter": 3,
        "label_name": "gaussian3",
        "label_path": f"{base_dir}/gaussian3",
        "label_type": "gaussian",
        "learning_rate": 0.001,
        "loss_func_seg": "Dice",
        "lw_kernel": 3,
        "max_epoch": 60,
        "meanPool_kernel": 5,
        "meanPool_NMS": True,
        "mini_dist": 10,
        "network": "ResUNet",
        "norm": "bn",
        "norm_type": "standardization",
        "num_classes": 3,
        "ocp_diameter": "12, 20",
        "ocp_name": "data_ocp",
        "ocp_path": f"{base_dir}/data_ocp",
        "optim": "AdamW",
        "others": "",
        # train_func() does `args.pad_size = args.pad_size[0]`, so this is a list
        "pad_size": [12],
        "paf_sigmoid": False,
        "pif_sigmoid": False,
        "prf1_alpha": 3,
        "random_num": 0,
        "scheduler": "OneCycleLR",
        "seg_tau": 0.95,
        "Sel_Referance": False,
        "sel_train_num": None,
        "skip_4v94": False,
        "skip_vesicles": False,
        "test_mode": "val",
        "test_use_pad": True,
        "threshold": 0.4,
        "tomo_format": ".mrc",
        "tomo_path": f"{base_dir}/data_std",
        "train_mode": "train",
        "use_aspp": False,
        "use_att": False,
        "use_bg": True,
        "use_bg_part": False,
        "use_CL": False,
        "use_CL_DA": False,
        "use_cluster": False,
        "use_coord": True,
        "use_ice_part": False,
        "use_IP": True,
        "use_lw": False,
        "use_mask": False,
        "use_paf": False,
        "use_se_loss": False,
        "use_sigmoid": False,
        "use_softmax": True,
        "use_softpool": False,
        "use_tanh": False,
        "use_uncert": False,
        "val_batch_size": 8,
        "val_block_size": 72,
        "weight_decay": 0.01,
    }
    args = Namespace(**CONFIG)
    args.cfg = CONFIG

    # Training
    train.train_func(args, stdout=None)