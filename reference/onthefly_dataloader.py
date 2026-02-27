"""
On-the-fly chip dataset for xView3.

Reads raw GeoTIFFs and extracts 800x800 crops during training,
eliminating the need for a separate preprocessing/chipping step.
Uses an LRU scene cache to amortize I/O across chips from the same scene.
"""

import math
import os
from collections import OrderedDict, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import torch
from rasterio.enums import Resampling
from torch.utils.data import Sampler

from constants import BACKGROUND, FISHING, NONFISHING, NONVESSEL


class SceneGroupedSampler(Sampler):
    """
    Samples chips grouped by scene so the on-the-fly scene cache stays hot.

    Each epoch: shuffle scene order, then shuffle chips within each scene.
    This ensures all chips from a scene are loaded consecutively, avoiding
    repeated 1+ GB GeoTIFF reads due to cache misses.
    """

    def __init__(self, chip_indices, shuffle=True, seed=0):
        self.chip_indices = chip_indices  # list of (scene_id, chip_index)
        self.shuffle = shuffle
        self.seed = seed
        self._epoch = 0

        # Group chip positions by scene
        self._scene_to_positions = defaultdict(list)
        for pos, (scene_id, _) in enumerate(chip_indices):
            self._scene_to_positions[scene_id].append(pos)
        self._scenes = list(self._scene_to_positions.keys())

    def set_epoch(self, epoch):
        self._epoch = epoch

    def __len__(self):
        return len(self.chip_indices)

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self._epoch)

        if self.shuffle:
            scene_order = rng.permutation(len(self._scenes))
        else:
            scene_order = np.arange(len(self._scenes))

        indices = []
        for si in scene_order:
            scene_id = self._scenes[si]
            positions = self._scene_to_positions[scene_id]
            if self.shuffle:
                positions = rng.permutation(positions).tolist()
            indices.extend(positions)

        return iter(indices)


class SceneGroupedDistributedSampler(Sampler):
    """
    Distributed version of SceneGroupedSampler for multi-GPU/multi-node training.

    Splits scenes across ranks so each rank owns a disjoint set of scenes,
    then iterates chips grouped by scene within each rank's subset.
    This keeps each rank's scene cache hot while ensuring no two ranks
    redundantly load the same scene.

    Use this in place of DistributedSampler when dataset_mode=onthefly
    with distributed training.
    """

    def __init__(self, chip_indices, num_replicas, rank, shuffle=True, seed=0):
        self.chip_indices = chip_indices
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.seed = seed
        self._epoch = 0

        # Group chip positions by scene
        scene_to_positions = defaultdict(list)
        for pos, (scene_id, _) in enumerate(chip_indices):
            scene_to_positions[scene_id].append(pos)
        self._all_scenes = list(scene_to_positions.keys())
        self._scene_to_positions = dict(scene_to_positions)

        # Assign scenes to this rank (round-robin)
        self._my_scenes = self._all_scenes[rank::num_replicas]

    def set_epoch(self, epoch):
        self._epoch = epoch

    def __len__(self):
        return sum(len(self._scene_to_positions[s]) for s in self._my_scenes)

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self._epoch)

        if self.shuffle:
            scene_order = rng.permutation(len(self._my_scenes)).tolist()
        else:
            scene_order = list(range(len(self._my_scenes)))

        indices = []
        for si in scene_order:
            scene_id = self._my_scenes[si]
            positions = self._scene_to_positions[scene_id]
            if self.shuffle:
                positions = rng.permutation(positions).tolist()
            indices.extend(positions)

        return iter(indices)


def _compute_grid_coords(height, width, chip_size):
    """
    Compute chip grid coordinates matching the existing row-major ordering
    from dataloader.py:get_grid_coords.

    The image is conceptually padded to be a multiple of chip_size,
    then tiled with evenly-spaced origins.
    """
    padded_h = math.ceil(height / chip_size) * chip_size
    padded_w = math.ceil(width / chip_size) * chip_size

    n_rows = padded_h // chip_size
    n_cols = padded_w // chip_size

    grid_coords_y = np.linspace(0, padded_h - chip_size, n_rows)
    grid_coords_x = np.linspace(0, padded_w - chip_size, n_cols)

    # Row-major: iterate y first, then x within each row
    grid_coords = [(int(x), int(y)) for y in grid_coords_y for x in grid_coords_x]
    return grid_coords, padded_h, padded_w


class OnTheFlyXView3Dataset:
    """
    Dataset that reads raw GeoTIFFs on-the-fly and extracts 800x800 chips.

    Same interface as OptimizedXView3Dataset so train.py can swap between them.
    """

    # Channel name -> GeoTIFF filename mapping (matches dataloader.py:91-98)
    CHANNEL_FILES = {
        "vh": "VH_dB.tif",
        "vv": "VV_dB.tif",
        "bathymetry": "bathymetry.tif",
        "wind_speed": "owiWindSpeed.tif",
        "wind_direction": "owiWindDirection.tif",
        "wind_quality": "owiWindQuality.tif",
        "mask": "owiMask.tif",
    }

    def __init__(
        self,
        root,
        transforms,
        split,
        detect_file=None,
        scene_list=None,
        channels=None,
        chip_size=800,
        bbox_size=5,
        background_frac=None,
        background_min=3,
        ais_only=True,
        min_max_norm=True,
        scene_cache_size=3,
        # Accepted but unused — keeps signature compatible with OptimizedXView3Dataset
        chips_path=".",
        overwrite_preproc=False,
        num_workers=1,
    ):
        if channels is None:
            channels = ["vh", "vv", "wind_direction"]

        self.root = root
        self.split = split
        self.transforms = transforms
        self.channels = channels
        self.chip_size = chip_size
        self.bbox_size = bbox_size
        self.background_frac = background_frac
        self.background_min = background_min
        self.ais_only = ais_only
        self.min_max_norm = min_max_norm
        self.scene_cache_size = scene_cache_size
        self.label_map = self._get_label_map()

        # Discover scenes
        if scene_list:
            self.scenes = scene_list
        else:
            self.scenes = sorted([
                d for d in os.listdir(root)
                if os.path.isdir(os.path.join(root, d))
            ])

        # For each scene, read the reference channel shape (metadata only)
        # and compute grid coords
        self._scene_shapes = {}   # scene_id -> (height, width)
        self._grid_coords = {}    # scene_id -> [(x, y), ...]
        self._padded_shapes = {}  # scene_id -> (padded_h, padded_w)

        for scene_id in self.scenes:
            ref_path = os.path.join(root, scene_id, self.CHANNEL_FILES[channels[0]])
            with rasterio.open(ref_path) as src:
                h, w = src.height, src.width
            self._scene_shapes[scene_id] = (h, w)
            coords, ph, pw = _compute_grid_coords(h, w, chip_size)
            self._grid_coords[scene_id] = coords
            self._padded_shapes[scene_id] = (ph, pw)

        # Parse detections
        if detect_file:
            det = pd.read_csv(detect_file, low_memory=False)
            vessel_class = []
            for _, row in det.iterrows():
                if row.is_vessel and row.is_fishing:
                    vessel_class.append(FISHING)
                elif row.is_vessel and not row.is_fishing:
                    vessel_class.append(NONFISHING)
                elif not row.is_vessel:
                    vessel_class.append(NONVESSEL)
            det["vessel_class"] = vessel_class
            if ais_only:
                det = det.dropna(subset=["vessel_class"])
            self.detections = det
        else:
            self.detections = None

        # Build detection_index: (scene_id, chip_index) -> [dict, ...]
        self.detection_index = defaultdict(list)
        if self.detections is not None:
            for scene_id in self.scenes:
                scene_det = self.detections[self.detections["scene_id"] == scene_id]
                if len(scene_det) == 0:
                    continue

                grid_coords = self._grid_coords[scene_id]
                scene_rows = np.array(scene_det["detect_scene_row"])
                scene_cols = np.array(scene_det["detect_scene_column"])

                # Convert scene pixel coords to chip-local coords + chip index
                chip_rows = scene_rows % self.chip_size
                chip_cols = scene_cols % self.chip_size
                chip_ind_row = scene_rows // self.chip_size * self.chip_size
                chip_ind_col = scene_cols // self.chip_size * self.chip_size

                for i, (_, det_row) in enumerate(scene_det.iterrows()):
                    try:
                        ci = grid_coords.index((int(chip_ind_col[i]), int(chip_ind_row[i])))
                    except ValueError:
                        # Detection falls outside grid — skip
                        continue
                    self.detection_index[(scene_id, ci)].append({
                        "rows": int(chip_rows[i]),
                        "columns": int(chip_cols[i]),
                        "vessel_class": det_row["vessel_class"],
                        "vessel_length_m": det_row.get("vessel_length_m", 0.0),
                    })

        # Build chip_indices list
        if self.detections is not None:
            # Only chips that have detections (+ background chips added below)
            self.chip_indices = list(set(self.detection_index.keys()))
        else:
            # All chips for inference
            self.chip_indices = []
            for scene_id in self.scenes:
                n_chips = len(self._grid_coords[scene_id])
                self.chip_indices += [(scene_id, i) for i in range(n_chips)]

        # Add background chips
        if self.background_frac and self.detections is not None:
            self._add_background_chips()

        # Drop the full detections DataFrame — detection_index has everything we need.
        # This is critical for Windows multiprocessing (spawn) which pickles the dataset.
        self._has_detections = self.detections is not None
        self.detections = None

        # Convert to regular dict so pickle doesn't carry defaultdict factory
        self.detection_index = dict(self.detection_index)

        # Per-worker scene cache (initialized lazily in workers)
        self._scene_cache = OrderedDict()

        print(f"[OnTheFly] {len(self.scenes)} scenes, {len(self.chip_indices)} chips")
        print("Initialization complete")

    def __getstate__(self):
        """Clear scene cache before pickling (for DataLoader workers)."""
        state = self.__dict__.copy()
        state["_scene_cache"] = OrderedDict()
        return state

    def _add_background_chips(self):
        """Add background chips for negative sampling."""
        np.random.seed(seed=0)
        for scene_id in self.scenes:
            n_chips = len(self._grid_coords[scene_id])
            det_chips = set(
                ci for (sid, ci) in self.detection_index.keys()
                if sid == scene_id
            )
            bg_chips = [i for i in range(n_chips) if i not in det_chips]
            if not bg_chips:
                continue
            n_bg = int(self.background_frac * max(len(det_chips), self.background_min))
            n_bg = min(n_bg, len(bg_chips))
            chosen = np.random.choice(bg_chips, size=n_bg, replace=False)
            for ci in chosen:
                self.chip_indices.append((scene_id, int(ci)))
                # Mark as background in detection_index
                self.detection_index[(scene_id, int(ci))].append({
                    "rows": -1,
                    "columns": -1,
                    "vessel_class": BACKGROUND,
                    "vessel_length_m": -1,
                })

    @staticmethod
    def _get_label_map():
        return {
            "background": BACKGROUND,
            "non_fishing": NONFISHING,
            "fishing": FISHING,
            "personnel": NONVESSEL,
            "other": NONVESSEL,
        }

    def _load_scene(self, scene_id):
        """
        Load all channels for a scene, with LRU caching.
        Returns dict of channel_name -> np.ndarray (padded to chip-multiple).
        """
        if scene_id in self._scene_cache:
            # Move to end (most recently used)
            self._scene_cache.move_to_end(scene_id)
            return self._scene_cache[scene_id]

        h, w = self._scene_shapes[scene_id]
        ph, pw = self._padded_shapes[scene_id]
        scene_dir = os.path.join(self.root, scene_id)

        arrays = {}
        for ch in self.channels:
            tif_path = os.path.join(scene_dir, self.CHANNEL_FILES[ch])
            with rasterio.open(tif_path) as src:
                if ch == self.channels[0]:
                    data = src.read(1)
                else:
                    # Resample to match reference channel shape if needed
                    if src.height != h or src.width != w:
                        data = src.read(
                            out_shape=(h, w),
                            resampling=Resampling.bilinear,
                        ).squeeze()
                    else:
                        data = src.read(1)

            # Pad to chip-multiple size (pad right/bottom with zeros)
            if data.shape[0] < ph or data.shape[1] < pw:
                padded = np.zeros((ph, pw), dtype=data.dtype)
                padded[:data.shape[0], :data.shape[1]] = data
                data = padded

            arrays[ch] = data

        # Evict oldest if cache is full
        if len(self._scene_cache) >= self.scene_cache_size:
            self._scene_cache.popitem(last=False)

        self._scene_cache[scene_id] = arrays
        return arrays

    def _extract_chip(self, scene_arrays, scene_id, chip_index):
        """Extract an 800x800 chip from cached scene arrays."""
        x0, y0 = self._grid_coords[scene_id][chip_index]
        cs = self.chip_size
        return {
            ch: arr[y0:y0 + cs, x0:x0 + cs].copy()
            for ch, arr in scene_arrays.items()
        }

    def __len__(self):
        return len(self.chip_indices)

    def __getitem__(self, idx):
        scene_id, chip_index = self.chip_indices[idx]

        # Load scene (cached) and extract chip
        scene_arrays = self._load_scene(scene_id)
        data = self._extract_chip(scene_arrays, scene_id, chip_index)

        # Channel conditioning (matches OptimizedXView3Dataset.__getitem__)
        for fl in self.channels:
            if fl == "wind_direction":
                data[fl][data[fl] < 0] = np.random.randint(0, 360, size=1)
                data[fl][data[fl] > 360] = np.random.randint(0, 360, size=1)
                data[fl] = data[fl] - 180
            elif fl == "wind_speed":
                data[fl][data[fl] < 0] = 0
                data[fl][data[fl] > 100] = 100
            elif fl in ["vh", "vv"]:
                data[fl][data[fl] < -50] = -50

            if self.min_max_norm:
                data_min, data_max = np.min(data[fl]), np.max(data[fl])
                denom = data_max - data_min
                if denom == 0:
                    data[fl] = np.zeros_like(data[fl], dtype=np.float32)
                else:
                    data[fl] = ((data[fl] - data_min) / denom).astype(np.float32)

        # Stack channels
        img = torch.tensor(np.array([data[fl] for fl in self.channels]))

        # Build target dict
        detects = self.detection_index.get((scene_id, chip_index), [])
        num_objs = len(detects)

        if num_objs == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            class_labels = torch.zeros((0,), dtype=torch.int64)
            length_labels = torch.zeros((0,), dtype=torch.float32)
            area = torch.zeros((0,), dtype=torch.float32)
        elif num_objs == 1 and detects[0]["vessel_class"] == BACKGROUND:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            class_labels = torch.zeros((1,), dtype=torch.int64)
            length_labels = torch.zeros((1,), dtype=torch.float32)
            area = torch.zeros((1,), dtype=torch.float32)
        else:
            boxes = []
            class_labels = []
            length_labels = []
            for det in detects:
                xmin = det["columns"] - self.bbox_size
                xmax = det["columns"] + self.bbox_size
                ymin = det["rows"] - self.bbox_size
                ymax = det["rows"] + self.bbox_size
                boxes.append([xmin, ymin, xmax, ymax])
                class_labels.append(det["vessel_class"])
                length_labels.append(det["vessel_length_m"])
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            class_labels = torch.as_tensor(class_labels, dtype=torch.int64)
            length_labels = torch.as_tensor(length_labels, dtype=torch.float32)
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        target = {
            "boxes": boxes,
            "labels": class_labels,
            "length_labels": length_labels,
            "scene_id": scene_id,
            "chip_id": torch.tensor(chip_index),
            "image_id": torch.tensor(idx),
            "area": area,
            "iscrowd": torch.zeros((num_objs,), dtype=torch.int64),
        }

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img.float(), target
