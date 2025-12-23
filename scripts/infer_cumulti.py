# infer.py is a script to infer on a dataset using a trained model.
#!/usr/bin/env python3

import argparse
import datetime
import os
import shutil
import subprocess
from shutil import copyfile
import yaml
import sys

# Internal modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from lidar2osm.models.user import User

def load_config(config_path):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config

if __name__ == "__main__":
    # Default configuration path
    default_config_path = "lidar2osm/config/inference.yaml"
    
    # Load default settings from YAML
    config = load_config(default_config_path)

    # Setup command line arguments
    splits = ["train", "valid", "test"]
    parser = argparse.ArgumentParser("./infer_cumulti.py")
    parser.add_argument(
        "--dataset_path",
        "-dataset_path",
        type=str,
        default=config.get('dataset_path', None),
        required=False,
        help="Dataset to train with. Default: (from config.yaml)"
    )
    parser.add_argument(
        "--dataset_name",
        "-d_name",
        type=str,
        default=config.get('dataset_name', None),
        required=False,
        help="Dataset's name. Default: (from config.yaml)"
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default=config.get('inference', {}).get('model_path', None),
        required=False,
        help="Directory to get the trained model. Default: (from config.yaml)"
    )
    parser.add_argument(
        "--split",
        "-s",
        type=str,
        choices=splits,
        default=config.get('inference', {}).get('split', 'valid'),
        required=False,
        help=f"Split to evaluate on. One of {splits}. Defaults to %(default)s"
    )
    parser.add_argument(
        "--data_config",
        "-data_config",
        type=str,
        default=config.get('data_config', None),
        required=False,
        help="Yaml file with configuration params for inference. Default: None"
    )
    parser.add_argument(
        "--environments",
        "-environments",
        type=list,
        default=config.get('environments', None),
        required=False,
        help="Environments to infer on. Default: None"
    )
    print(f"\n\nenvironments: {config.get('environments', None)}\n\n")
    parser.add_argument(
        "--robots",
        "-robots",
        type=list,
        default=config.get('robots', None),
        required=False,
        help="Robots to infer on. Default: None"
    )
    FLAGS, unparsed = parser.parse_known_args()

    # print summary of what we will do
    print("----------")
    print("INTERFACE:")
    print("dataset_name", FLAGS.dataset_name)
    print("dataset_path", FLAGS.dataset_path)
    print("data_config", FLAGS.data_config)
    print("model", FLAGS.model)
    print("infering", FLAGS.split)
    print("environments", FLAGS.environments)
    print("robots", FLAGS.robots)
    print("----------\n")

    # open arch config file
    model_name = os.path.basename(FLAGS.model)
    try:
        print("Opening arch config file for %s" % FLAGS.model)
        # ARCH = yaml.safe_load(open(f"{FLAGS.model}", "r"))
        ARCH = yaml.safe_load(open(FLAGS.model + "/arch_cfg.yaml", 'r'))
    except Exception as e:
        print(e)
        print("Error opening arch yaml file.")
        quit()

    # open data config file
    try:
        print("Opening data config file from %s" % FLAGS.model)
        # DATA = yaml.safe_load(open(FLAGS.model + "/data_cfg.yaml", 'r'))
        DATA = yaml.safe_load(open(f"{FLAGS.data_config}", "r"))
    except Exception as e:
        print(e)
        print("Error opening data yaml file.")
        quit()

    # create log folder for each sequence
    try:
        if FLAGS.dataset_name == "CU-MULTI":
            for env in FLAGS.environments:
                for robot in FLAGS.robots:
                    inference_dir = os.path.join(FLAGS.dataset_path, env, robot, "lidar_labels")
                    print(f"inference_dir: {inference_dir}")
                    # if os.path.isdir(inference_dir):
                    os.makedirs(inference_dir, exist_ok=True)
    except Exception as e:
        print(e)
        print("Error creating log directory. Check permissions!")
        raise

    # does model folder exist?
    if os.path.isdir(FLAGS.model):
        print("model folder exists! Using model from %s" % (FLAGS.model))
    else:
        print("model folder doesnt exist! Can't infer...")
        quit()

    for env in FLAGS.environments:
        for robot in FLAGS.robots:
            # create user and infer dataset
            user = User(ARCH, DATA, FLAGS.dataset_name, FLAGS.dataset_path, FLAGS.model, FLAGS.split, env, robot)
            
            print("\n\nDONE CREATING USER\n\n")
            user.infer()
