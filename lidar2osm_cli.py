#!/usr/bin/env python3

"""
Non-conflicting repo-root driver for Lidar2OSM.

Why not `lidar2osm.py`?
Because the repo is commonly run with `PYTHONPATH=$(pwd)`, and a top-level
`lidar2osm.py` would shadow the `lidar2osm` package, breaking imports like
`from lidar2osm.models...`.

Use:
  - `python lidar2osm_cli.py --data_pipeline --config config.yaml`
  - `python -m lidar2osm --data_pipeline --config config.yaml`
  - `lidar2osm --data_pipeline --config config.yaml` (after `pip install -e .`)
"""

from lidar2osm.cli import main


if __name__ == "__main__":
    raise SystemExit(main())


