import glob
import json
import os
import time
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import rasterio
import ray
import torch
from rasterio.enums import Resampling

from constants import BACKGROUND, FISHING, NONFISHING, NONVESSEL
from utils import chip_sar_img, pad


class OptimizedXView3Dataset:
    """
    Optimized version of XView3Dataset with better performance
    """
    
    def __init__(
        self,
        root,
        transforms,
        split,
        detect_file=None,
        scene_list=None,
        chips_path=".",
        channels=["vh", "vv", "wind_direction"],
        chip_size=800,
        overwrite_preproc=False,
        bbox_size=5,
        background_frac=None,
        background_min=3,
        ais_only=True,
        num_workers=1,
        min_max_norm=True,
    ):
        self.root = root
        self.split = split
        self.bbox_size = bbox_size
        self.background_frac = background_frac
        self.background_min = background_min
        self.chips_path = chips_path
        self.transforms = transforms
        self.channels = channels
        self.chip_size = chip_size
        self.overwrite_preproc = overwrite_preproc
        self.ais_only = ais_only
        self.num_workers = num_workers
        self.label_map = self.get_label_map()
        self.min_max_norm = min_max_norm
        self.coords = {}
        
        # Getting image list
        if not scene_list:
            self.scenes = [
                a.strip("\n").strip("/").split("/")[-1][:67] for a in os.listdir(root)
            ]
        else:
            self.scenes = scene_list

        # Get all detections
        if detect_file:
            self.detections = pd.read_csv(detect_file, low_memory=False)
            vessel_class = []
            for ii, row in self.detections.iterrows():
                if row.is_vessel and row.is_fishing:
                    vessel_class.append(FISHING)
                elif row.is_vessel and not row.is_fishing:
                    vessel_class.append(NONFISHING)
                elif not row.is_vessel:
                    vessel_class.append(NONVESSEL)
            self.detections["vessel_class"] = vessel_class
            if self.ais_only:
                self.detections = self.detections.dropna(subset=["vessel_class"])
        else:
            self.detections = None

        # Get chip-level detection coordinates
        self.pixel_detections = self.chip_and_get_pixel_detections()

        # Add background chips for negative sampling
        if self.background_frac and (self.detections is not None):
            print("Adding background chips...")
            self.add_background_chips()

        # Write annotations to file
        if self.overwrite_preproc or not os.path.exists(
            f"{self.chips_path}/{self.split}_chip_annotations.csv"
        ):
            print("Writing chip annotations to file...")
            self.pixel_detections.to_csv(
                f"{self.chips_path}/{self.split}_chip_annotations.csv", index=False
            )
            
        # Get chip indices for each scene
        if self.detections is not None:
            self.chip_indices = list(
                set(
                    zip(
                        self.pixel_detections.scene_id, self.pixel_detections.chip_index
                    )
                )
            )
        else:
            self.chip_indices = []
            for scene_id in self.scenes:
                chip_num = self.get_chip_number(scene_id)
                self.chip_indices += [(scene_id, a) for a in range(chip_num)]

        # OPTIMIZATION: Pre-index the detections for faster lookup
        self._create_detection_index()
        
        print(f"Number of Unique Chips: {len(self.chip_indices)}")
        print("Initialization complete")

    def _create_detection_index(self):
        """Create a fast lookup index for detections"""
        if self.detections is not None:
            # Create a dictionary for O(1) lookup instead of DataFrame filtering
            self.detection_index = defaultdict(list)
            for _, row in self.pixel_detections.iterrows():
                key = (row['scene_id'], row['chip_index'])
                self.detection_index[key].append(row)
        else:
            self.detection_index = None

    @staticmethod
    def get_label_map():
        """Get label mapping"""
        background_labels = ["background"]
        non_fishing_labels = ["non_fishing"]
        fishing_labels = ["fishing"]
        personnel_labels = ["personnel"]
        other_labels = ["other"]

        label_map = {a: NONFISHING for a in non_fishing_labels}
        label_map.update({a: FISHING for a in fishing_labels})
        label_map.update({a: NONVESSEL for a in personnel_labels})
        label_map.update({a: NONVESSEL for a in other_labels})
        label_map.update({a: BACKGROUND for a in background_labels})

        return label_map

    def __len__(self):
        return len(self.chip_indices)

    def __getitem__(self, idx):
        # Load and condition image chip data
        scene_id, chip_index = self.chip_indices[idx]
        
        # OPTIMIZATION: Pre-allocate data dictionary
        data = {}
        
        # OPTIMIZATION: Load all channels at once with error handling
        try:
            for fl in self.channels:
                pth = f"{self.chips_path}/{scene_id}/{fl}/{int(chip_index)}_{fl}.npy"
                data[fl] = np.load(pth)
                
                # Apply channel-specific processing
                if fl == "wind_direction":
                    data[fl][data[fl] < 0] = np.random.randint(0, 360, size=1)
                    data[fl][data[fl] > 360] = np.random.randint(0, 360, size=1)
                    data[fl] = data[fl] - 180
                elif fl == "wind_speed":
                    data[fl][data[fl] < 0] = 0
                    data[fl][data[fl] > 100] = 100
                elif fl in ["vh", "vv"]:
                    data[fl][data[fl] < -50] = -50
                    
                # OPTIMIZATION: Vectorized normalization
                if self.min_max_norm:
                    data_min, data_max = np.min(data[fl]), np.max(data[fl])
                    denom = data_max - data_min
                    if denom == 0:
                        data[fl][:] = 0.0
                    else:
                        data[fl][:] = (data[fl] - data_min) / denom
        except Exception as e:
            print(f"Error loading data for {scene_id}, chip {chip_index}: {e}")
            # Return dummy data on error
            dummy_shape = (800, 800)
            data = {fl: np.zeros(dummy_shape) for fl in self.channels}

        # Stacking channels to create multi-band image chip
        img = torch.tensor(np.array([data[fl] for fl in self.channels]))

        # OPTIMIZATION: Use pre-indexed detections for O(1) lookup
        if self.detection_index is not None:
            key = (scene_id, chip_index)
            detects = self.detection_index[key]
            
            num_objs = len(detects)
            if num_objs == 0:
                # No detections
                boxes = torch.zeros((0, 4), dtype=torch.float32)
                class_labels = torch.zeros((0,), dtype=torch.int64)
                length_labels = torch.zeros((0,), dtype=torch.float32)
                area = torch.zeros((0,), dtype=torch.float32)
            elif (num_objs == 1) and (detects[0].vessel_class == BACKGROUND):
                # Background chip
                boxes = torch.zeros((0, 4), dtype=torch.float32)
                class_labels = torch.zeros((1,), dtype=torch.int64)
                length_labels = torch.zeros((1,), dtype=torch.float32)
                area = torch.zeros((1,), dtype=torch.float32)
            else:
                # Regular detections
                boxes = []
                class_labels = []
                length_labels = []
                
                for detect in detects:
                    xmin = detect.columns - self.bbox_size
                    xmax = detect.columns + self.bbox_size
                    ymin = detect.rows - self.bbox_size
                    ymax = detect.rows + self.bbox_size
                    boxes.append([xmin, ymin, xmax, ymax])
                    class_labels.append(detect.vessel_class)
                    length_labels.append(detect.vessel_length_m)

                boxes = torch.as_tensor(boxes, dtype=torch.float32)
                class_labels = torch.as_tensor(class_labels, dtype=torch.int64)
                length_labels = torch.as_tensor(length_labels, dtype=torch.float32)
                area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        else:
            # Return dummy values for inference
            boxes = torch.tensor([])
            class_labels = torch.tensor(-1)
            length_labels = torch.tensor(-1)
            area = torch.tensor(-1)
            num_objs = 0

        # Create target dictionary
        target = {
            "boxes": boxes,
            "labels": class_labels,
            "length_labels": length_labels,
            "scene_id": scene_id,
            "chip_id": torch.tensor(chip_index),
            "image_id": torch.tensor(idx),
            "area": area,
            "iscrowd": torch.zeros((num_objs,), dtype=torch.int64)
        }

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img.float(), target

    def get_chip_number(self, scene_id):
        """Get number of chips using first channel"""
        return len(glob.glob(f"{self.chips_path}/{scene_id}/{self.channels[0]}/*.npy"))

    def add_background_chips(self):
        """Add background chips with no detections"""
        # Implementation same as original
        for scene_id in self.scenes:
            num_chips = self.get_chip_number(scene_id)
            scene_detect_chips = (
                self.pixel_detections[self.pixel_detections["scene_id"] == scene_id][
                    "chip_index"
                ]
                .astype(int)
                .tolist()
            )
            scene_background_chips = [
                a for a in range(num_chips) if a not in list(set(scene_detect_chips))
            ]
            num_background = int(
                self.background_frac * max(len(scene_detect_chips), self.background_min)
            )
            np.random.seed(seed=0)
            chip_nums = np.random.choice(
                scene_background_chips, size=num_background, replace=False
            )
            rows = []
            cols = [
                "index", "detect_lat", "detect_lon", "vessel_length_m", "source",
                "detect_scene_row", "detect_scene_column", "is_vessel", "is_fishing",
                "distance_from_shore_km", "scene_id", "confidence", "top", "left",
                "bottom", "right", "detect_id", "vessel_class", "scene_rows",
                "scene_cols", "rows", "columns", "chip_index",
            ]
            for ii in range(num_background):
                row = [
                    -1, -1, -1, -1, "background", -1, -1, -1, -1, -1, scene_id,
                    -1, -1, -1, -1, -1, -1, BACKGROUND, -1, -1, -1, -1, chip_nums[ii],
                ]
                rows.append(row)
            df_background = pd.DataFrame(rows, columns=cols)
            self.pixel_detections = pd.concat((self.pixel_detections, df_background))

    def chip_and_get_pixel_detections(self):
        """Preprocess all scenes to chip xView3 dataset images"""
        # Implementation same as original - this is called during initialization
        start = time.time()
        if self.num_workers > 1:
            ray.init()
            remote_process_scene = ray.remote(process_scene)
            jobs = []
            for jj, scene_id in enumerate(self.scenes):
                jobs.append(
                    remote_process_scene.remote(
                        scene_id, self.detections, self.channels, self.chip_size,
                        self.chips_path, self.overwrite_preproc, self.root, self.split, jj,
                    )
                )
            chip_detects = ray.get(jobs)
            pixel_detections = pd.concat(chip_detects)
        else:
            chip_detects = []
            for jj, scene_id in enumerate(self.scenes):
                print(f"Processing scene {jj} of {len(self.scenes)}...")
                chip_detects.append(
                    process_scene(
                        scene_id, self.detections, self.channels, self.chip_size,
                        self.chips_path, self.overwrite_preproc, self.root, self.split, jj,
                    )
                )
            pixel_detections = pd.concat(chip_detects).reset_index()

        el = time.time() - start
        print(f"Elapsed Time: {np.round(el/60, 2)} Minutes")
        return pixel_detections


# Import the original process_scene function
from dataloader import process_scene 