<p align="center">
  <img src="./assets/banner_light.png" alt="banner" height="400">
  <img src="./assets/semantic_clipper.png" alt="banner" height="400">
</p>

# Lidar2OSM: Distributed Urban Mapping with Bandwidth-Efficient Geospatial Descriptors 📡🗺️🤖

This repository supports a project investigating the feasibility of converting raw lidar scans into OpenStreetMap (OSM)-style representations for distributed multi-robot operations. The aim is to empower autonomous robot teams to efficiently generate and accurately share low-dimensional maps of their environments, particularly in bandwidth-limited urban areas with constrained GPS signals.

The sheer complexity of LiDAR data, with its high-dimensional semantic richness, could introduce noise or irrelevant details that don't actually help with multi-robot map matching but instead make the process more difficult. This difficulty becomes exacerbated when considering transient objects and varying observational perspectives. Counterintuitively, utilizing OSM-style data, though simpler, might highlight the key features that are most useful for peer-to-peer map associations without overwhelming an underlying algorithm. The maximum clique technique is particularly effective at exploiting these core relationships in the filtered data.

<div align="center">
  <img src="assets/lidar2osm_example.gif" width="400" alt="Example of Lidar2OSM goal">
</div>

### Included Projects

This repository includes the `semantic_clipper` project, which is based on work originally developed by the MIT Aerospace Controls Lab. The code is used under the MIT License, and the original license is included in the `semantic_clipper/` directory.

## CU-MULTI Dataset

There are a limited number of datasets available that use a mobile ground-based robot in urban scenarios with both accurate GPS data, lidar, and IMU. It is for this reason we demonstrate our findings on two major on-road datasets, KITTI-360 and NuScenes. However, we fill this gap with the CU-MULTI Dataset, a multi-robot dataset collected in an off-road urban environment consisting of two large environments on the University of Colorado Boulder's Main Campus.

## Running Lidar2OSM (Python)

This repo uses a standard `src/` Python package layout. Install the package in editable mode once, then you can run scripts/CLI **without** setting `PYTHONPATH`.

### Install (recommended)

```bash
conda env create -f environment.yaml
conda activate lidar2osm_env
python -m pip install -e .
```

### One-config workflow

Edit the single repo-root config:
- `config.yaml`

### Run the pipelines

```bash
# Data relabeling pipeline (runs the existing scripts under src/core/data/)
lidar2osm --data_pipeline --config config.yaml

# Training
lidar2osm --training_pipeline --config config.yaml

# Inference
lidar2osm --inference_pipeline --config config.yaml
```

You can also use module invocation:

```bash
python -m lidar2osm --data_pipeline --config config.yaml
```

### Run individual scripts (still supported)

After `pip install -e .`, you can run the underlying scripts directly:

```bash
# Data pipeline scripts
python src/core/data/create_global_sem_map_octree.py --dataset_path /mnt/semkitti/cu-multi-data/ --num_scans 10 --global_voxel 0.1
python src/core/data/extend_building_offsets.py --dataset_path /mnt/semkitti/cu-multi-data/
python src/core/data/relabel_maps.py --dataset_path /mnt/semkitti/cu-multi-data/
python src/core/data/relabel_dense_semantics.py --dataset_path /mnt/semkitti/cu-multi-data/
python src/core/data/relabel_scans.py --dataset_path "/mnt/semkitti/cu-multi-data/"

# Model Testing
python src/core/models/train.py --config config.yaml
python src/core/models/infer.py --config config.yaml
```