import argparse
import errno
import glob
import json
import os
import os.path as osp
import signal
import sys
import time
from subprocess import PIPE, Popen
import numpy as np

def create_default_train_dict(name_str,desc_str,indir_str,outdir_root):
    # Default args dict
    # name_str = "{}_rad_{:.3f}_dataset_lightweight_{:.3f}"
    # desc_str = "Using {} to verify effect of patch radius sizes. Currently testing radius of size {:.3f} with dataset lightweight AABB {:.3f}."
    # indir_str = "./dataset_wSoftLabels_AABB_lightweight_{:.3f}/"

    train_opt = argparse.Namespace()
    d = vars(train_opt)
    d["CUR_MODEL"] = ""
    d["name"] = name_str
    d["desc"] = desc_str
    d["indir"] = indir_str
    d["job_outdir"] = outdir_root
    d["outdir"] = "{}/models".format(outdir_root)
    d["logdir"] = "{}/logs".format(outdir_root)
    d["trainset"] = "trainingset.txt"
    d["testset"] = "validationset.txt"
    d["saveinterval"] = 10
    d["refine"] = ""
    # training paramters
    d["patch_radius"] = [-1]#[cur_radius]
    d["patch_center"] = "point"
    d["patch_point_count_std"] = 0
    d["patches_per_shape"] = 500000
    d["workers"] = 8
    d["cache_capacity"] = 10
    d["seed"] = 3627473
    d["training_order"] = "random"
    d["identical_epochs"] = False
    d["lr"] = 0.0001
    d["momentum"] = 0.9
    d["use_pca"] = False
    d["use_point_stn"] = True
    d["use_feat_stn"] = True
    d["sym_op"] = "max"
    d["point_tuple"] = 1
    d["points_per_patch"] = -1#cur_npts_per_patch

    d["max_depth"] = 5000.
    d["nSupportSamples"] = 15000

    d["saveinterval"] = 1
    # training paramters
    d["nepoch"] = 50
    d["batchSize"] = 32

    d["refine"] = ""

    d["features"] = ["softLabel"]
    d["task"] = "GT_outliers"

    d["semantic_output"] = False

    d["TRAINING"] = True

    d["data_augmentation"] = True

    d["patch_radius"] = [2500]
    d["points_per_patch"] = 1000

    d["patch_geometry"] = "SPHERE"
    d["rmx_dir"] = "{}/RMX_SCAN"
    d["rmx_str"] = "SCAN_{}_{}"

    return d

def create_default_test_dict(name_str,indir_str,outdir_root):
    train_opt = argparse.Namespace()
    d = vars(train_opt)
    d["indir"] = indir_str
    d["outdir"] = "{}/".format(outdir_root)
    d["dataset"] = "testingset.txt"
    d["modeldir"] = "{}/models".format(outdir_root)
    d["model"] = name_str
    d["CUR_MODEL"] = ""
    d["modelpostfix"] = "_model.pth"
    d["parmpostfix"] = "_params.pth"
    d["sparse_patches"] = False
    d["sampling"] = "sequential_shapes_random_patches"

    d["batchSize"] = 0
    d["patches_per_shape"] = 10000
    d["workers"] = 16
    d["cache_capacity"] = 5
    d["seed"] = 3627473

    return d


def readFromJson(path):
    with open(path) as json_file:
        data = json.load(json_file)
        return data

def writeToJson(path, data):
    with open(path, 'w') as json_file:
        json.dump(data, json_file)

def mkdir_p(path):
#Cree un dossier
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and osp.isdir(path):
            pass
        else:
            raise


def cmdline(command):
    process = Popen(
        args=command,
        stdout=PIPE,
        shell=True
    )
    result = str(process.communicate()[0].decode("utf-8"))
    return result


def signal_handler(sig, frame):
    print()
    print()
    print()
    print()
    print('Exiting...')
    sys.exit(0)
