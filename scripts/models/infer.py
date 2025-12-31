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
from lidar2osm import CONFIG_DIR

def load_yaml(config_path):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config

if __name__ == "__main__":
    # Default configuration path
    config = load_yaml(CONFIG_DIR / "inference.yaml")

    # Setup command line arguments
    splits = ["train", "valid", "test"]
    parser = argparse.ArgumentParser("./infer.py")
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
    # New arguments for multiple environments/robots (merged from infer_cumulti.py)
    parser.add_argument(
        "--environments",
        "-envs",
        nargs='+',
        default=config.get('environments', None),
        required=False,
        help="List of environments to infer on (e.g. main_campus kittredge_loop)"
    )
    parser.add_argument(
        "--robots",
        "-robots",
        nargs='+',
        default=config.get('robots', None),
        required=False,
        help="List of robots to infer on (e.g. robot1 robot2)"
    )
    parser.add_argument(
        "--infer_all",
        action="store_true",
        help="If set, infer on all combinations of provided environments and robots (CU-MULTI only)"
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
    print("infer_all", FLAGS.infer_all)
    print("----------\n")

    # open arch config file
    model_name = os.path.basename(FLAGS.model)
    try:
        print("Opening arch config file for %s" % FLAGS.model)
        ARCH = yaml.safe_load(open(FLAGS.model + "/arch_cfg.yaml", 'r'))
    except Exception as e:
        print(e)
        print("Error opening arch yaml file.")
        quit()

    # open data config file
    try:
        print("Opening data config file from %s" % FLAGS.model)
        DATA = yaml.safe_load(open(f"{FLAGS.data_config}", "r"))
    except Exception as e:
        print(e)
        print("Error opening data yaml file.")
        quit()

    # Check model folder
    if not os.path.isdir(FLAGS.model):
        print("model folder doesnt exist! Can't infer...")
        quit()
    print("model folder exists! Using model from %s" % (FLAGS.model))

    # Logic for CU-MULTI: Iterate over envs/robots if requested
    if FLAGS.dataset_name == "CU-MULTI":
        tasks = []
        if FLAGS.infer_all and FLAGS.environments and FLAGS.robots:
            # Generate tasks from combinations provided in args
            for env in FLAGS.environments:
                for robot in FLAGS.robots:
                    tasks.append((env, robot))
        else:
            # Fallback to DATA config (or just whatever is in config files)
            # This handles the case where we just want to run what's in the data config
            env = DATA["environment"]
            robots = DATA.get("test_robots", [])
            if isinstance(robots, str): robots = [robots]
            
            # If robots not in DATA, check FLAGS.robots
            if not robots and FLAGS.robots:
                robots = FLAGS.robots
            
            for robot in robots:
                tasks.append((env, robot))
        
        if not tasks:
            print("No tasks (environment/robot pairs) found to infer on!")
            quit()

        for env, robot in tasks:
            print(f"\n>>> Running inference for Environment: {env}, Robot: {robot}")
            
            # Update DATA for current task to ensure User picks the correct one
            DATA["environment"] = env
            DATA["test_robots"] = [robot] 
            
            # Create log folders
            try:
                inference_dir = os.path.join(FLAGS.dataset_path, env, robot, f"{robot}_{env}_lidar_labels_confidence")
                conf_dir = os.path.join(inference_dir, "confidence_scores")
                multiclass_conf_dir = os.path.join(inference_dir, "multiclass_confidence_scores")
                print(f"inference_dir: {inference_dir}")
                
                os.makedirs(inference_dir, exist_ok=True)
                os.makedirs(conf_dir, exist_ok=True)
                os.makedirs(multiclass_conf_dir, exist_ok=True)
            except Exception as e:
                print(e)
                print("Error creating log directory. Check permissions!")
                raise

            # Create user and infer
            user = User(ARCH, DATA, FLAGS.dataset_name, FLAGS.dataset_path, FLAGS.model, FLAGS.split)
            user.infer()

    elif FLAGS.dataset_name == "KITTI-360":
        # Create log folders for KITTI
        try:
            for seq in DATA["split"]["test"]:
                seq_str = f"2013_05_28_drive_{seq:04d}_sync"
                inference_dir = os.path.join(FLAGS.dataset_path, "data_3d_semantics", seq_str, "inferred")
                print(f"inference_dir: {inference_dir}")
                os.makedirs(inference_dir, exist_ok=True)
        except Exception as e:
            print(e)
            print("Error creating log directory. Check permissions!")
            raise

        # Single user instance for KITTI (handles sequences internally)
        user = User(ARCH, DATA, FLAGS.dataset_name, FLAGS.dataset_path, FLAGS.model, FLAGS.split)
        user.infer()
    
    else:
        print(f"Unknown dataset name: {FLAGS.dataset_name}")
        # Try generic inference
        user = User(ARCH, DATA, FLAGS.dataset_name, FLAGS.dataset_path, FLAGS.model, FLAGS.split)
        user.infer()
