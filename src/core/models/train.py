# train.py is the main script to train the model. It loads the configuration from the config.yaml file and starts the training process.
#!/usr/bin/env python3

# External
import argparse
import os
import shutil
from shutil import copyfile
from pathlib import Path
import yaml
import sys

# Internal
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from lidar2osm.config_loader import find_repo_root, get_config_dir, load_yaml, resolve_path


def parse_yaml(base_config, model_config):
    # Setup command line arguments based on progressive growing or single model
    parser = argparse.ArgumentParser("./train.py")
    parser.add_argument(
        "--dataset",
        "-d",
        type=str,
        default=base_config['dataset_path'],
        required=False,
        help="Dataset to train with. Default: (from config.yaml)"
    )
    parser.add_argument(
        "--arch_cfg",
        "-ac",
        type=str,
        default=model_config['arch_config'],
        required=False,
        help="Architecture yaml cfg file. See /config/arch for sample. Default: (from config.yaml)"
    )
    parser.add_argument(
        "--data_cfg",
        "-dc",
        type=str,
        default=base_config['data_config'],
        required=False,
        help="Classification yaml cfg file. See /config/labels for sample. Default: (from config.yaml)"
    )

    # Where log dir will be made
    parser.add_argument(
        "--log",
        "-l",
        type=str,
        default=base_config['training']['model_path'],
        help="Directory to put the log data. Default: (from config.yaml)"
    )

    parser.add_argument(
        "--name",
        "-n",
        type=str,
        default=model_config.get('model_name', ''),
        help="If you want to give an additional descriptive name"
    )
    parser.add_argument(
        "--pretrained",
        "-p",
        type=str,
        default=model_config.get('pretrained_model_path', None),
        required=False,
        help="Directory to get the pretrained model. If not passed, do from scratch! Default: (from config.yaml)"
    )

    FLAGS, unparsed = parser.parse_known_args()

    return FLAGS


def load_config_and_train(FLAGS):
    if FLAGS.name:
        FLAGS.log = os.path.join(FLAGS.log, FLAGS.name)

    # Print summary after final configuration
    print("\n----------")
    print("Train Configuration:\n")
    print("use_progressive_growing: ", use_progressive)
    print("dataset: ", FLAGS.dataset)
    print("arch_cfg: ", FLAGS.arch_cfg)
    print("data_cfg: ", FLAGS.data_cfg)
    print("log: ", FLAGS.log)
    print("pretrained: ", FLAGS.pretrained)
    print("----------\n")

    # open arch config file
    try:
        print("Opening arch config file %s" % FLAGS.arch_cfg)
        ARCH = yaml.safe_load(open(FLAGS.arch_cfg, "r"))
    except Exception as e:
        print(e)
        print("Error opening arch yaml file.")
        quit()

    # open data config file
    try:
        print("Opening data config file %s" % FLAGS.data_cfg)
        DATA = yaml.safe_load(open(FLAGS.data_cfg, "r"))
    except Exception as e:
        print(e)
        print("Error opening data yaml file.")
        quit()

    # create log folder
    try:
        if os.path.isdir(FLAGS.log):
            shutil.rmtree(FLAGS.log)
        os.makedirs(FLAGS.log)
    except Exception as e:
        print(e)
        print("Error creating log directory. Check permissions!")
        quit()

    # does model folder exist?
    if FLAGS.pretrained is not None:
        if os.path.isdir(FLAGS.pretrained):
            print("model folder exists! Using model from %s" % (FLAGS.pretrained))
        else:
            print("model folder doesnt exist! Start with random weights...")
    else:
        print("No pretrained directory found.")

    # copy all files to log folder (to remember what we did, and make inference
    # easier). Also, standardize name to be able to open it later
    try:
        print("Copying files to %s for further reference." % FLAGS.log)
        copyfile(FLAGS.arch_cfg, FLAGS.log + "/arch_cfg.yaml")
        copyfile(FLAGS.data_cfg, FLAGS.log + "/data_cfg.yaml")
    except Exception as e:
        print(e)
        print("Error copying files, check permissions. Exiting...")
        quit()

    return ARCH, DATA


if __name__ == "__main__":
    # Allow overriding the base YAML used to populate defaults.
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config (default: repo-root config.yaml).",
    )
    bootstrap_args, _ = bootstrap.parse_known_args()

    # Seed function (unchanged)
    def seed_torch(seed=1024):
        import random
        import numpy as np
        import torch

        random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        print("We use the seed: {}".format(seed))

    repo_root = find_repo_root(Path(__file__))
    cfg_dir = get_config_dir(repo_root)
    base_cfg_path = repo_root / "config.yaml"
    if bootstrap_args.config:
        base_cfg_path = Path(bootstrap_args.config)
    base_config = load_yaml(base_cfg_path)
    dataset_name = base_config['dataset_name']

    # Check for progressive growing first
    # This is basically inverse distilation
    use_progressive = base_config['training']['use_progressive_growing']
    
    if use_progressive:
        model_configs = base_config['training']['model_configs']
    else:
        model_configs = base_config['training']['model_configs']['model_name']

    # Train the model progressively if configured or train a single model
    if use_progressive:
        for model_cfg in model_configs:
            current_model_config = model_configs[model_cfg]
            print(f"Training model: {model_cfg}")
            print(f"pretrained_model_path: {current_model_config.get('pretrained_model_path', None)}")
            FLAGS = parse_yaml(base_config, current_model_config)
            # Resolve any repo-relative config paths from the base/model YAML.
            FLAGS.arch_cfg = str(resolve_path(FLAGS.arch_cfg, base_dir=cfg_dir.parent))
            FLAGS.data_cfg = str(resolve_path(FLAGS.data_cfg, base_dir=cfg_dir.parent))
            ARCH, DATA = load_config_and_train(FLAGS)
            seed_torch()
            # Lazy import so `--help` works even if heavy deps (e.g. torch) aren't installed.
            from lidar2osm.models.trainer_wb import Trainer
            trainer = Trainer(ARCH, DATA, dataset_name, FLAGS.dataset, FLAGS.log, FLAGS.pretrained)
            trainer.train()
    # else:
    #     load_config_and_train(base_config, training_config, model_config)
