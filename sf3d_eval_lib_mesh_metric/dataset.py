from __future__ import annotations

import os

import numpy as np
import torch
from torch.utils.data import Dataset

from .cameras import EVAL_CAMERA_PARAMS, INPUT_VIEW_IDS
from .config import EvalConfig
from .image_io import depth_to_hw_resized, load_depth_file, load_rgba_cv2, resize_chw, rgba_to_rgb_mask
from .sf3d_input import make_sf3d_input_image


class SF3DEvalDataset(Dataset):
    def __init__(self, cfg: EvalConfig):
        self.cfg = cfg
        self.data_path = cfg.data_path
        self.depth_path = cfg.depth_path
        self.eval_path = cfg.eval_path
        self.gt_mesh_path = cfg.gt_mesh_path
        self.output_size = cfg.output_size
        self.depth_render_size = cfg.depth_render_size

        required_paths = [self.data_path, self.depth_path, self.eval_path]
        if self.gt_mesh_path is not None:
            required_paths.append(self.gt_mesh_path)

        for required_path in required_paths:
            if not os.path.isdir(required_path):
                raise FileNotFoundError(f"Directory not found: {required_path}")

        depth_items = [
            (archive, obj, os.path.join(self.depth_path, archive, obj))
            for archive in sorted(os.listdir(self.depth_path))
            if os.path.isdir(os.path.join(self.depth_path, archive))
            for obj in sorted(os.listdir(os.path.join(self.depth_path, archive)))
            if os.path.isdir(os.path.join(self.depth_path, archive, obj, "depth"))
        ]

        n = len(depth_items)
        end = max(0, int(cfg.val_size * n))
        depth_items = depth_items[-end:] if end > 0 else []

        if cfg.object_start is not None or cfg.object_end is not None:
            depth_items = depth_items[cfg.object_start : cfg.object_end]

        if cfg.max_objects is not None:
            depth_items = depth_items[: cfg.max_objects]

        self.items = depth_items
        self.eval_index = self._build_eval_index()
        self.gt_mesh_index = self._build_gt_mesh_index()
        print(f"[dataset] objects = {len(self.items)}")

    def _build_eval_index(self) -> dict[str, str]:
        index: dict[str, str] = {}
        for archive in sorted(os.listdir(self.eval_path)):
            archive_path = os.path.join(self.eval_path, archive)
            if not os.path.isdir(archive_path):
                continue
            for obj in sorted(os.listdir(archive_path)):
                p = os.path.join(archive_path, obj)
                if os.path.isdir(p):
                    index[obj] = p
        return index

    def _build_gt_mesh_index(self) -> dict[str, str]:
        """
        Walk gt_mesh_path recursively and index every .glb file by object name.

        Supported layouts (all resolved to the same flat {item_name -> path} dict):
          gt_mesh_path/archive_xxx/object_id.glb          -> key: object_id
          gt_mesh_path/chunk_xxx/object_id.glb            -> key: object_id
          gt_mesh_path/archive_xxx/object_id/mesh.glb     -> key: object_id  (dir name)
          gt_mesh_path/archive_xxx/object_id/object_id.glb -> key: object_id

        The first match wins; deeper / later paths do NOT overwrite earlier ones so
        that a deterministic walk order (sorted) gives stable results.
        """
        index: dict[str, str] = {}
        if self.gt_mesh_path is None:
            return index

        for root, dirs, files in os.walk(self.gt_mesh_path):
            dirs.sort()  # deterministic traversal order
            for fname in sorted(files):
                if not fname.lower().endswith(".glb"):
                    continue
                full_path = os.path.join(root, fname)
                stem = os.path.splitext(fname)[0]
                # If the file is named "mesh.glb" use the parent directory name as the key,
                # otherwise use the file stem directly.
                item_name = os.path.basename(root) if stem == "mesh" else stem
                if item_name not in index:
                    index[item_name] = full_path

        return index

    def __len__(self) -> int:
        return len(self.items)

    def _find_eval_item_path(self, item_name: str) -> str:
        if item_name in self.eval_index:
            return self.eval_index[item_name]
        raise FileNotFoundError(f"Cannot find {item_name} in EVAL_PATH={self.eval_path}")

    def _find_gt_mesh_path(self, archive_name: str, item_name: str) -> str | None:
        if self.gt_mesh_path is None:
            return None

        if item_name in self.gt_mesh_index:
            return self.gt_mesh_index[item_name]

        raise FileNotFoundError(
            f"Cannot find GT mesh for item={item_name} anywhere under {self.gt_mesh_path}. "
            f"Indexed {len(self.gt_mesh_index)} objects. "
            "Check that the folder contains .glb files and that the object name matches."
        )

    def __getitem__(self, idx: int) -> dict:
        archive_name, item_name, _ = self.items[idx]
        item_path = os.path.join(self.data_path, archive_name, item_name)
        eval_item_path = self._find_eval_item_path(item_name)

        depth_dir = os.path.join(eval_item_path, "depth")

        input_view_id = INPUT_VIEW_IDS[self.cfg.sf3d_input_view_index]
        input_view_name = f"{input_view_id:03d}"
        input_rgb_path = os.path.join(item_path, "rgb", f"{input_view_name}.png")
        input_rgba = load_rgba_cv2(input_rgb_path)
        sf3d_input = make_sf3d_input_image(input_rgba, self.cfg)

        input_rgb_preview, input_mask_preview = rgba_to_rgb_mask(input_rgba, white_bg=True)
        input_rgb_preview = resize_chw(input_rgb_preview.transpose(2, 0, 1), self.output_size, mode="bilinear").transpose(1, 2, 0)
        input_mask_preview = resize_chw(input_mask_preview.transpose(2, 0, 1), self.output_size, mode="nearest").transpose(1, 2, 0)

        target_rgbs, target_masks = [], []
        for view_idx, (elev, azim) in enumerate(EVAL_CAMERA_PARAMS):
            view_name = f"{view_idx:03d}"
            rgba = load_rgba_cv2(os.path.join(eval_item_path, "rgb", f"{view_name}.png"))
            rgb, mask = rgba_to_rgb_mask(rgba, white_bg=True)
            rgb = resize_chw(rgb.transpose(2, 0, 1), self.output_size, mode="bilinear").transpose(1, 2, 0)
            mask = resize_chw(mask.transpose(2, 0, 1), self.output_size, mode="nearest").transpose(1, 2, 0)
            target_rgbs.append(rgb)
            target_masks.append(mask)

        # Depth evaluation follows the same 16 eval views as RGB evaluation.
        depth_masks, gt_depths = [], []
        for view_idx, (elev, azim) in enumerate(EVAL_CAMERA_PARAMS):
            view_name = f"{view_idx:03d}"

            # Use eval RGB/mask so GT depth mask and rendered pred depth are aligned
            # with the same eval camera/view indexing.
            rgba = load_rgba_cv2(os.path.join(eval_item_path, "rgb", f"{view_name}.png"))
            _, mask = rgba_to_rgb_mask(rgba, white_bg=True)
            mask = resize_chw(mask.transpose(2, 0, 1), self.depth_render_size, mode="nearest").transpose(1, 2, 0)

            # Expected depth names now match eval view names: 000.npy/.npz ... 015.npy/.npz.
            depth = load_depth_file(depth_dir, view_name)
            depth = depth_to_hw_resized(depth, self.depth_render_size)

            depth_masks.append(mask)
            gt_depths.append(depth[..., None])

        return {
            "archive_name": archive_name,
            "item_name": item_name,
            "object_id": f"{archive_name}/{item_name}",
            "gt_mesh_path": self._find_gt_mesh_path(archive_name, item_name),
            "input_view_name": input_view_name,
            "input_rgb_preview": torch.from_numpy(input_rgb_preview.astype(np.float32)),
            "input_mask_preview": torch.from_numpy(input_mask_preview.astype(np.float32)),
            "sf3d_input": sf3d_input,
            "target_rgbs": torch.from_numpy(np.stack(target_rgbs).astype(np.float32)),
            "target_masks": torch.from_numpy(np.stack(target_masks).astype(np.float32)),
            "target_camera_params": torch.tensor(EVAL_CAMERA_PARAMS, dtype=torch.float32),
            "depth_masks": torch.from_numpy(np.stack(depth_masks).astype(np.float32)),
            "gt_depths": torch.from_numpy(np.stack(gt_depths).astype(np.float32)),
            "depth_camera_params": torch.tensor(EVAL_CAMERA_PARAMS, dtype=torch.float32),
        }