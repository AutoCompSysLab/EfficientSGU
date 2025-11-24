'''
The script is used to model Grounded SAM detections in 3D, it assumes the tag2text classes are avaialable. It also assumes the dataset has Clip features saved for each object/mask.
'''
from __future__ import annotations

# Standard library imports
import os
import copy
import uuid
from pathlib import Path
import pickle
import gzip
import faiss
# Third-party imports
import cv2
import numpy as np
import scipy.ndimage as ndi
import torch
from PIL import Image
from tqdm import trange
from open3d.io import read_pinhole_camera_parameters
import hydra
from omegaconf import DictConfig
import open_clip
from ultralytics import YOLO, SAM
import supervision as sv
from collections import Counter
from dataclasses import dataclass, field
from typing import Sequence, Dict, Any, List, Optional
import re
from math import inf

# Local application/library specific imports
from EfficientGraph.utils.optional_rerun_wrapper import (
    OptionalReRun, 
    orr_log_annotated_image, 
    orr_log_camera, 
    orr_log_depth_image, 
    orr_log_edges, 
    orr_log_objs_pcd_and_bbox, 
    orr_log_rgb_image, 
    orr_log_vlm_image
)
from EfficientGraph.utils.optional_wandb_wrapper import OptionalWandB
from EfficientGraph.utils.geometry import rotation_matrix_to_quaternion
from EfficientGraph.utils.logging_metrics import DenoisingTracker, MappingTracker
from EfficientGraph.utils.vlm import consolidate_captions, get_obj_rel_from_image_gpt4v, get_openai_client
from EfficientGraph.utils.ious import mask_subtract_contained, compute_3d_iou, compute_3d_iou_accurate_batch, compute_iou_batch
from EfficientGraph.utils.general_utils import (
    ObjectClasses, 
    find_existing_image_path, 
    get_det_out_path, 
    get_exp_out_path, 
    get_vlm_annotated_image_path, 
    handle_rerun_saving, 
    load_saved_detections, 
    load_saved_hydra_json_config, 
    make_vlm_edges_and_captions, 
    measure_time, 
    save_detection_results,
    save_edge_json,
    save_text_feats_npy,
    save_clip_feats_npy, 
    save_hydra_config,
    save_obj_json, 
    save_objects_for_frame, 
    save_pointcloud, 
    should_exit_early, 
    vis_render_image
)
from EfficientGraph.dataset.datasets_common import get_dataset
from EfficientGraph.utils.vis import (
    OnlineObjectRenderer, 
    save_video_from_frames, 
    vis_result_fast_on_depth, 
    vis_result_for_vlm, 
    vis_result_fast, 
    save_video_detections
)
from EfficientGraph.slam.slam_classes import MapEdgeMapping, MapObjectList, DetectionList, to_tensor

from EfficientGraph.slam.utils import (
    filter_gobs,
    filter_objects,
    get_bounding_box,
    init_process_pcd,
    make_detection_list_from_pcd_and_gobs,
    denoise_objects,
    merge_objects, 
    detections_to_obj_pcd_and_bbox,
    prepare_objects_save_vis,
    process_cfg,
    process_edges,
    process_pcd,
    processing_needed,
    resize_gobs,
    compute_overlap_matrix_general
)
from EfficientGraph.slam.mapping import (
    compute_spatial_similarities,
    compute_visual_similarities,
    compute_text_similarities,
    aggregate_similarities,
    match_detections_to_objects_adaptive,
    match_detections_to_objects,
    merge_obj_matches
)
from EfficientGraph.utils.model_utils import compute_clip_features_batched
from EfficientGraph.utils.general_utils import get_vis_out_path, cfg_to_dict, check_run_detections

import json

import inspect

from collections import defaultdict


def _to_builtin(x):
    """일반 필드용: None 값은 필드 자체를 빼서 가볍게 저장"""
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (list, tuple)):
        return [_to_builtin(v) for v in x]
    if isinstance(x, dict):
        return {k: _to_builtin(v) for k, v in x.items() if v is not None}
    return x

def _to_builtin_keep_none(x):
    """parent용: None도 유지해서 JSON에 null로 기록"""
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (list, tuple)):
        return [_to_builtin_keep_none(v) for v in x]
    if isinstance(x, dict):
        return {k: _to_builtin_keep_none(v) for k, v in x.items()}
    return x

def _parent_minimal_always(p):
    """parent를 항상 {'rel': ..., 'tag': ...}로 만들되, 없으면 None → JSON에 null"""
    rel = p.get("rel") if isinstance(p, dict) else None
    tag = p.get("tag") if isinstance(p, dict) else None
    return {"rel": rel, "tag": tag}

def save_sg_json_parent_min(path: Path | str, sg_objects: list) -> Path:
    """
    업로드한 SG JSON 형태로 저장하되,
    parent는 항상 {'rel','tag'} 두 키만 기록(없으면 null).
    나머지 필드는 기존 값 유지.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    out = {}
    for i, o in enumerate(sg_objects):
        key = o.get("key", f"object_{i+1}")  # 없으면 보정
        parent_min = _parent_minimal_always(o.get("parent"))

        # 일반 필드(없으면 제거)
        rec = {
            "id":             o.get("id", i+1),
            "object_tag":     o.get("class_name", ""),
            "object_caption": o.get("object_caption", []),
            "bbox_extent":    o.get("extent"),
            "bbox_center":    o.get("center"),
            "bbox_volume":    o.get("volume"),
            "is_fixed":       bool(o.get("is_fixed", False)),
            "anchor":         o.get("anchor"),  # 앵커는 원본 유지(없으면 빠짐)
            # parent는 아래에서 따로 넣음(Null 유지)
        }
        rec = _to_builtin({k: v for k, v in rec.items() if v is not None})
        rec["parent"] = _to_builtin_keep_none(parent_min)  # ← rel/tag가 None이어도 유지

        out[key] = rec

    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    return path

def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)

def _poly_iou(p1, p2) -> float:
    if p1 is None or p2 is None:
        return 0.0
    try:
        inter = p1.intersection(p2).area
        uni   = p1.union(p2).area
        return float(inter / uni) if uni > 1e-9 else 0.0
    except Exception:
        return 0.0

# SG 쓰기 편하게 load하는 법
def _trail_int(key: str, default: int = 10**9) -> int:
    """'object_12' → 12. 숫자 없으면 default."""
    m = re.search(r'(\d+)$', key)
    return int(m.group(1)) if m else default

def load_sg_objects(base_dir: Path, filename: str = "obj_json_r_mapping_stride10.json") -> List[Dict[str, Any]]:
    """
    - 'object_1', 'object_2', ... 키를 trailing number 기준으로 정렬해 순서대로 로드
    - parent는 {'rel', 'tag'}만 남기고 단순화
    - 바로 obj["is_fixed"], obj["class_name"], obj["parent"]["rel"] 식으로 사용 가능
    """
    path = base_dir / filename
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("SG json은 dict 형태(키: 'object_<n>')여야 합니다.")

    objects: List[Dict[str, Any]] = []
    for key in sorted(raw.keys(), key=_trail_int):
        r = raw[key]
        p = r.get("parent")
        parent_min = None
        if isinstance(p, dict):
            # 필요 최소만 유지
            rel = p.get("rel")
            tag = p.get("tag")
            parent_min = {"rel": rel, "tag": tag} if (rel is not None or tag is not None) else None

        obj = {
            "key": key,                                            # "object_6" 등
            "id": int(r["id"]) if "id" in r and r["id"] is not None else None,
            "class_name": str(r.get("object_tag", "")).lower(),    # 통일 키
            "is_fixed": bool(r.get("is_fixed", False)),
            "center": r.get("bbox_center"),
            "extent": r.get("bbox_extent"),
            "volume": r.get("bbox_volume"),
            "parent": parent_min,                                  # {'rel','tag'} 또는 None
            "anchor": r.get("anchor"),
            
            # 필요하면 이후에 외부 저장소(lower_y, polys)와 i순서로 매칭
        }
        objects.append(obj)

    return objects

# footprint load 받을 때
try:
    from shapely.geometry import Polygon, MultiPolygon
    _HAS_SHAPELY = True
except Exception:
    _HAS_SHAPELY = False

def load_lower_and_polys(npz_path: Path, as_shapely: bool = True):
    """
    polys_r_mapping_stride10.npz 에서
      - lower_y: (N,) float32
      - polys:   길이 N 리스트 (각 원소 = shapely Polygon 또는 None)
    만 반환.
    내부적으로 verts/offsets는 사용하되 바깥으로는 노출하지 않음.
    """
    data = np.load(npz_path)

    # 필수 키
    offsets  = data["offsets"].astype(np.int64)     # (N+1,)
    lower_y  = data["lower_y"].astype(np.float32)   # (N,)
    has_poly = data["has_poly"].astype(bool)        # (N,)

    # 정량화 복원
    if "verts_q" in data:
        scale = float(np.array(data["scale"]).reshape(-1)[0])
        verts = data["verts_q"].astype(np.float32) / scale   # (M, 2) (x,z)
    else:
        verts = data["verts"].astype(np.float32)             # (M, 2)

    N = lower_y.shape[0]
    assert offsets.shape[0] == N + 1
    assert has_poly.shape[0] == N

    polys = [None] * N
    for i in range(N):
        if not has_poly[i]:
            polys[i] = None
            continue

        s, e = int(offsets[i]), int(offsets[i+1])
        ring = verts[s:e]   # (K,2) with (x,z)
        if ring.shape[0] < 3:
            polys[i] = None
            continue

        if as_shapely and _HAS_SHAPELY:
            poly = Polygon(ring)  # 자동으로 닫힘
            if (not poly.is_valid) or poly.is_empty or (poly.area <= 0):
                poly = poly.buffer(0)
                if isinstance(poly, MultiPolygon):
                    poly = max(poly.geoms, key=lambda p: p.area)
            polys[i] = poly if hasattr(poly, "area") and poly.area > 0 else None
        else:
            # shapely 미사용 모드: ring 좌표만 넘겨줌
            polys[i] = ring

    # 메모리 정리(원하면)
    del verts

    return lower_y, polys



##
def _abs_frame_idx(frame_idx: int, cfg) -> int:
    """데이터셋의 샘플 인덱스(frame_idx) → 절대 프레임 인덱스(abs_idx)"""
    stride = int(getattr(cfg, "stride", 1))
    start  = int(getattr(cfg, "start", 0))
    return start + frame_idx * stride

def before_dir_for_idx(frame_idx: int, cfg) -> Path:
    abs_idx = _abs_frame_idx(frame_idx, cfg)
    return Path(cfg.before_results_root) / f"frame{abs_idx:06d}"

def load_before_by_dataset_idx(frame_idx: int, cfg) -> dict:
    """stride 반영해서 before_results에서 해당 프레임 로드"""
    frame_dir = before_dir_for_idx(frame_idx, cfg)
    return load_before_frame_as_results(frame_dir)

import logging, sys
logger = logging.getLogger("sg.filter")
if not logger.handlers:
    logger.setLevel(logging.DEBUG)
    h1 = logging.StreamHandler(sys.stdout)
    h2 = logging.FileHandler("sg_filter.log", encoding="utf-8")
    fmt = logging.Formatter("[%(asctime)s][%(name)s][%(levelname)s] fr=%(frame)d :: %(message)s",
                            datefmt="%H:%M:%S")
    h1.setFormatter(fmt); h2.setFormatter(fmt)
    logger.addHandler(h1); logger.addHandler(h2)

def log_filter_hit(frame_idx, **kv):
    extra = {"frame": frame_idx}
    msg = "FILTER_ENTER " + " ".join(f"{k}={v}" for k,v in kv.items())
    logger.debug(msg, extra=extra)



# def _agg_sim_compat(spatial_sim, visual_sim, cfg, text_sim=None):
#     """
#     aggregate_similarities 시그니처를 런타임에 확인하고
#     지원되는 키워드만 안전하게 전달한다.
#     """
#     params = inspect.signature(aggregate_similarities).parameters
#     kwargs = {
#         "match_method": cfg["match_method"],
#         "phys_bias": cfg["phys_bias"],
#         "spatial_sim": spatial_sim,
#         "visual_sim": visual_sim,
#     }
#     if "text_sim" in params and text_sim is not None:
#         kwargs["text_sim"] = text_sim
#     if "alpha" in params:
#         kwargs["alpha"] = cfg.get("text_alpha", 0.5)
#     if "normalize" in params:
#         kwargs["normalize"] = True
#     return aggregate_similarities(**kwargs)


# ###


def subset_gobs(g: Dict[str, Any], idxs: Sequence[int]) -> Dict[str, Any]:
    """
    g에서 객체 단위 필드만 idxs로 서브셋팅해서 반환.
    - 길이가 num_objects와 같은 리스트/ndarray는 선택 인덱싱
    - 길이가 다르거나 스칼라/메타 데이터는 그대로 복사
    """
    # out = {}
    # if "class_id" not in g:
    #     raise KeyError("gobs dict must have 'class_id' key")
    # num = len(g["class_id"])
    # idxs = list(idxs)

    # for k, v in g.items():
    #     # 객체 단위로 정렬된 필드(길이가 num과 같으면 그렇게 간주)
    #     if isinstance(v, list) and len(v) == num:
    #         out[k] = [v[i] for i in idxs]
    #     elif isinstance(v, np.ndarray) and (v.shape[0] == num):
    #         out[k] = v[idxs, ...]
    #     else:
    #         # 스칼라/메타(예: 이미지 경로, 프레임 번호 등)는 그대로 둠
    #         out[k] = v
    # return out
    if "class_id" not in g:
        raise KeyError("gobs dict must have 'class_id' key")

    num = int(len(g["class_id"])) if isinstance(g["class_id"], (list, np.ndarray)) else 0

    # idxs 정규화
    try:
        import torch
        if isinstance(idxs, torch.Tensor):
            idxs = idxs.detach().cpu().numpy()
    except Exception:
        pass

    idxs = np.array(idxs, dtype=np.int64, copy=False).reshape(-1) if np.size(idxs) else np.array([], dtype=np.int64)

    # --- 필요하면 1-based → 0-based 보정 (의심될 때만 켜서 확인)
    # if idxs.size and idxs.min() >= 1 and idxs.max() == num:
    #     idxs = idxs - 1

    # 범위 밖 제거 + 중복 제거
    idxs = idxs[(idxs >= 0) & (idxs < num)]
    if idxs.size:
        idxs = np.unique(idxs)

    out = {}
    for k, v in g.items():
        if isinstance(v, list) and len(v) == num:
            out[k] = [v[i] for i in idxs] if idxs.size else []
        elif hasattr(v, "shape") and getattr(v, "shape", None) and v.shape[0] == num:
            out[k] = v[idxs, ...] if idxs.size else v[:0, ...]
        else:
            out[k] = v
    return out



# --- 유틸: 마스크 안전 카운터 ---
def _num_dets(g):
    if not isinstance(g, dict):
        return 0
    m = g.get('mask', None)
    if m is None:
        return 0
    try:
        return len(m)  # np.ndarray(N, H, W) 또는 list
    except Exception:
        return 0

# # --- CLIP feat npy 안전 로더들 ---
# def load_clip_feats_npy(out_path, mmap=True, to_torch=True, device=None, dtype=np.float32):
#     """
#     - mmap=True: 큰 배열도 가볍게(읽을 때만 건드림)
#     - to_torch=True: torch.Tensor로 변환(정규화는 호출부에서)
#     - dtype: float32로 캐스팅(half 저장된 경우에도 연산 안정화)
#     """
#     arr = np.load(out_path, mmap_mode='r' if mmap else None)  # shape: (N, D)
#     if dtype is not None:
#         # 메모리맵이면 직접 캐스팅 시 복사 발생(필요)
#         arr = np.array(arr, copy=True, dtype=dtype)
#     if to_torch:
#         t = torch.from_numpy(arr)
#         if device is not None:
#             t = t.to(device)
#         return t
#     return arr

# def load_clip_feat_row_npy(out_path, idx, to_torch=True, device=None, dtype=np.float32):
#     """
#     (N, D) 중 idx번째 한 행만 가져오기 (대용량에서도 빠르게).
#     """
#     arr = np.load(out_path, mmap_mode='r')
#     row = np.array(arr[idx], copy=True)
#     if dtype is not None and row.dtype != dtype:
#         row = row.astype(dtype, copy=False)
#     if to_torch:
#         t = torch.from_numpy(row)
#         if device is not None:
#             t = t.to(device)
#         return t
#     return row

import copy
    
def match_before_after_3d(ago_gobs, after_gobs, pose_before, pose_after, depth_before, depth_after, ago_rgb, after_rgb, intrinsics, pre_color_path, color_path, obj_classes, frame_idx ,obj_pcd_max_points ,overlap_thresh=1.2):
    """
    3D 공간에서 두 프레임의 객체를 매칭합니다.
    RGB, Depth, Pose 정보를 활용하여 객체들을 3D로 변환하고, 겹침이 일정 임계값 이상이면 매칭합니다.
    """

    # 아래 가정하는 이유 벽면만 보는 경우 아무것도 검출 안될 수 있어서

    # len_a = len(ago_gobs['mask'])
    # len_b = len(after_gobs['mask'])
    
    # if len_a == 0 and len_b == 0:
    #     return [], [] 
    # elif len_a == 0 and len_b != 0:
    #     temp_appeared = [i for i in range(len(ago_gobs))]
    #     return [], temp_appeared
    # elif len_a != 0 and len_b == 0:
    #     temp_disappeared = [j for j in range(len(ago_gobs))]
    #     return temp_disappeared, []

# === PATCH 2: match_before_after_3d의 len 처리 (함수 맨 앞 부분 교체) ===
    len_a = int(len(ago_gobs.get('mask', [])))
    len_b = int(len(after_gobs.get('mask', [])))

    if len_a == 0 and len_b == 0:
        return [], []
    elif len_a == 0 and len_b != 0:
        # 이전엔 없고 지금은 있음 → 전부 appeared (after 기준 인덱스)
        return [], list(range(len_b))
    elif len_a != 0 and len_b == 0:
        # 이전엔 있었고 지금은 없음 → 전부 disappeared (before 기준 인덱스)
        return list(range(len_a)), []
    
    # obj to pcd
    # temp_ago_gobs = copy.deepcopy(ago_gobs)
    # temp_after_gobs = copy.deepcopy(after_gobs)
    

    # after_gobs = mask_subtract_contained(after_gobs['xyxy'], after_gobs['mask'])
    # ago_gobs = mask_subtract_contained(ago_gobs['xyxy'], ago_gobs['mask'])

    after_obj_pcds_and_bboxes = measure_time(detections_to_obj_pcd_and_bbox)(
        depth_array=depth_after,
        masks=after_gobs['mask'],
        cam_K=intrinsics.cpu().numpy()[:3, :3],  # Camera intrinsics
        image_rgb= after_rgb,
        trans_pose= pose_after,
        min_points_threshold= 5,
        spatial_sim_type='axis_aligned',
        obj_pcd_max_points=obj_pcd_max_points,
        device='cuda',
    )

    for after_obj in after_obj_pcds_and_bboxes:
        if after_obj:
            after_obj["pcd"] = init_process_pcd(
                pcd=after_obj["pcd"],
                downsample_voxel_size= 0.01,
                dbscan_remove_noise= True,
                dbscan_eps=0.1,
                dbscan_min_points= 10,
            )
            after_obj["bbox"] = get_bounding_box(
                spatial_sim_type='overlap', 
                pcd=after_obj["pcd"],
            )
            
    
    ago_obj_pcds_and_bboxes = measure_time(detections_to_obj_pcd_and_bbox)(
        depth_array=depth_before,
        masks=ago_gobs['mask'],
        cam_K=intrinsics.cpu().numpy()[:3, :3],  # Camera intrinsics
        image_rgb= ago_rgb,
        trans_pose= pose_before,
        min_points_threshold= 5,
        spatial_sim_type='axis_aligned',
        obj_pcd_max_points=obj_pcd_max_points,
        device='cuda',
    )

    for ago_obj in ago_obj_pcds_and_bboxes:
        if ago_obj:
            ago_obj["pcd"] = init_process_pcd(
                pcd=ago_obj["pcd"],
                downsample_voxel_size= 0.01,
                dbscan_remove_noise= True,
                dbscan_eps=0.1,
                dbscan_min_points= 10,
            )
            ago_obj["bbox"] = get_bounding_box(
                spatial_sim_type='overlap', 
                pcd=ago_obj["pcd"],
            )        
                    
    # ago_detection_list = make_detection_list_from_pcd_and_gobs( # 이게 중요한듯
    #         ago_obj_pcds_and_bboxes, ago_gobs, pre_color_path, obj_classes, frame_idx
    #     )
    
    # after_detection_list = make_detection_list_from_pcd_and_gobs( # 이게 중요한듯
    #         after_obj_pcds_and_bboxes, after_gobs, color_path, obj_classes, frame_idx
    #     )
        
    # # 공간 유사도 구하기 

    # spatial_similar = compute_spatial_similarities(
    #     spatial_sim_type='overlap', 
    #     detection_list=ago_detection_list, 
    #     objects=after_detection_list,
    #     downsample_voxel_size= 0.01,
    # )

    # # 매칭된 객체 추적
    # matched_before = []
    # matched_after = []
    
    # for i, before_obj in enumerate(ago_detection_list):
    #     best_match_score = -float('inf')
    #     best_match_idx = -1
        
    #     for j, after_obj in enumerate(after_detection_list):
    #         if spatial_similar[i, j] > best_match_score:
    #             best_match_score = spatial_similar[i, j]
    #             best_match_idx = j
        
    #     # 유사도가 일정 기준 이상일 경우 매칭
    #     if best_match_score >= overlap_thresh:
    #         matched_before.append(i)
    #         matched_after.append(best_match_idx)
    
    # # 매칭되지 않은 객체는 사라진 객체로 처리
    # disappeared = [i for i in range(len(ago_detection_list)) if i not in matched_before]
    
    # # 매칭되지 않은 객체는 새로운 객체로 처리
    # appeared = [j for j in range(len(after_detection_list)) if j not in matched_after]
    
    # return disappeared, appeared
    

    # for ago_obj in ago_obj_pcds_and_bboxes:
    #     if ago_obj:
    #         ago_obj["pcd"] = init_process_pcd(
    #             pcd=ago_obj["pcd"],
    #             downsample_voxel_size= 0.01,
    #             dbscan_remove_noise= True,
    #             dbscan_eps=0.1,
    #             dbscan_min_points= 10,
    #         )
    #         ago_obj["bbox"] = get_bounding_box(
    #             spatial_sim_type='overlap', 
    #             pcd=ago_obj["pcd"],
    #         )        
                    
    # You asked me to complete the code starting from here to calculate the overlap
    # and identify appeared/disappeared objects.
    
    # 1. Filter out invalid objects (None) and keep track of original indices.
    ago_valid_objects = []
    ago_original_indices = []
    for i, obj in enumerate(ago_obj_pcds_and_bboxes):
        if obj and len(obj['pcd'].points) > 0:
            ago_valid_objects.append(obj)
            ago_original_indices.append(i)

    after_valid_objects = []
    after_original_indices = []
    for i, obj in enumerate(after_obj_pcds_and_bboxes):
        if obj and len(obj['pcd'].points) > 0:
            after_valid_objects.append(obj)
            after_original_indices.append(i)

    # 2. Handle edge cases where one or both frames have no valid objects.
    if not ago_valid_objects:
        # If there were no objects before, all new objects have appeared.
        return [], after_original_indices

    if not after_valid_objects:
        # If there are no objects now, all previous objects have disappeared.
        return ago_original_indices, []
        
    # 3. Convert the lists of valid objects into MapObjectList for the utility function.
    ago_map_list = MapObjectList(ago_valid_objects)
    after_map_list = MapObjectList(after_valid_objects)

    # 4. Compute the bidirectional overlap matrix.
    # `overlap_ago_to_after[i, j]` = ratio of 'after' object j's points covered by 'ago' object i.
    overlap_ago_to_after = compute_overlap_matrix_general(
        objects_a=ago_map_list,
        objects_b=after_map_list,
        downsample_voxel_size=0.025 # A value is required by the function
    )

    # `overlap_after_to_ago[j, i]` = ratio of 'ago' object i's points covered by 'after' object j.
    overlap_after_to_ago = compute_overlap_matrix_general(
        objects_a=after_map_list,
        objects_b=ago_map_list,
        downsample_voxel_size=0.025
    )
    
    # A match is considered robust if the overlap is high in at least one direction.
    # We take the element-wise maximum of the two overlap matrices.
    # The shape of both matrices is (num_ago_valid, num_after_valid).
    combined_overlap_matrix = np.maximum(overlap_ago_to_after, overlap_after_to_ago.T)

    # 5. Identify matched objects based on the threshold.
    # This allows for one-to-many or many-to-one matches as requested.
    
    # Find indices in the VALID list for any object that has a match.
    matched_ago_indices_in_valid_list = set(np.where(combined_overlap_matrix >= overlap_thresh)[0])
    matched_after_indices_in_valid_list = set(np.where(combined_overlap_matrix >= overlap_thresh)[1])

    # 6. Determine which objects are unmatched (disappeared or appeared).
    # An object disappeared if its index (in the valid list) is not in the matched set.
    disappeared_valid_indices = [
        i for i, _ in enumerate(ago_valid_objects) 
        if i not in matched_ago_indices_in_valid_list
    ]
    
    # An object appeared if its index (in the valid list) is not in the matched set.
    appeared_valid_indices = [
        j for j, _ in enumerate(after_valid_objects) 
        if j not in matched_after_indices_in_valid_list
    ]

    # 7. Map the indices from the valid lists back to the original gobs indices.
    disappeared = [ago_original_indices[i] for i in disappeared_valid_indices]
    appeared = [after_original_indices[j] for j in appeared_valid_indices]

    return disappeared, appeared


# def l2norm_torch(x, eps=1e-9):
#     return x / (x.norm(dim=-1, keepdim=True) + eps)

# def _to_bool_mask(m):
#     m = np.asarray(m)
#     if m.dtype != bool:
#         return m > 0
#     return m

# def match_before_after_masks_robust_gobs(
#     before_gobs, after_gobs,
#     tol_px=2, iou_thresh=0.5,
#     w_iou=0.7, w_pos=0.2, w_area=0.1
# ):
#     # 1) 마스크 꺼내기 (HxW bool 배열 리스트)
#     before_masks = [ _to_bool_mask(m) for m in before_gobs.get("mask", []) ]
#     after_masks  = [ _to_bool_mask(m) for m in after_gobs.get("mask", []) ]

#     # 2) 이미지 크기 추출 (마스크에서 얻는 게 가장 안전)
#     if len(before_masks) > 0:
#         H, W = before_masks[0].shape
#     elif len(after_masks) > 0:
#         H, W = after_masks[0].shape
#     else:
#         raise ValueError("No masks found in both before_gobs and after_gobs.")

#     # 3) 원 함수 호출
#     return match_before_after_masks_robust(
#         before_masks, after_masks, (H, W),       # ← image_shape 전달
#         tol_px=tol_px, iou_thresh=iou_thresh,
#         w_iou=w_iou, w_pos=w_pos, w_area=w_area
#     )

# def compute_iou(box1, box2):
#     # xyxy: [x1, y1, x2, y2]
#     x1 = max(box1[0], box2[0])
#     y1 = max(box1[1], box2[1])
#     x2 = min(box1[2], box2[2])
#     y2 = min(box1[3], box2[3])
#     inter = max(0, x2 - x1) * max(0, y2 - y1)
#     area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
#     area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
#     union = area1 + area2 - inter
#     return inter / union if union > 0 else 0

# def cosine_sim2(a, b, eps=1e-9):
#     return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + eps)

# def expand_box(box, tol=2):
#     """bbox를 살짝 확장해서 IoU 계산할 때 tolerance 반영"""
#     x1, y1, x2, y2 = box
#     return [x1 - tol, y1 - tol, x2 + tol, y2 + tol]

# from scipy.optimize import linear_sum_assignment
# from scipy.ndimage import binary_dilation

# def mask_iou(m1: np.ndarray, m2: np.ndarray) -> float:
#     inter = np.logical_and(m1, m2).sum()
#     union = np.logical_or(m1, m2).sum()
#     return float(inter) / union if union > 0 else 0.0

# def dilated_mask_iou(m1: np.ndarray, m2: np.ndarray, tol_px: int = 2) -> float:
#     if tol_px > 0:
#         m1 = binary_dilation(m1, iterations=tol_px)
#         m2 = binary_dilation(m2, iterations=tol_px)
#     return mask_iou(m1, m2)

# def centroid(mask):
#     ys, xs = np.nonzero(mask)
#     if len(xs) == 0: return None
#     return np.array([xs.mean(), ys.mean()], dtype=np.float32)

# def area(mask):
#     return float(mask.sum())

# def match_before_after_masks_robust(
#     before_masks, after_masks, image_shape,
#     tol_px=2, iou_thresh=0.5,
#     w_iou=0.7, w_pos=0.2, w_area=0.1
# ):
#     H, W = image_shape[:2]
#     diag = np.hypot(H, W)  # 위치 정규화

#     B, A = len(before_masks), len(after_masks)
#     if B == 0 and A == 0: return [], []

#     S = np.zeros((B, A), dtype=np.float32)
#     b_c = [centroid(m) for m in before_masks]
#     a_c = [centroid(m) for m in after_masks]
#     b_a = [area(m) for m in before_masks]
#     a_a = [area(m) for m in after_masks]

#     for i in range(B):
#         for j in range(A):
#             # 1) mask IoU (with dilation tolerance)
#             iou = dilated_mask_iou(before_masks[i], after_masks[j], tol_px)

#             # 2) 위치 유사도 (0~1, 가까울수록 1)
#             if b_c[i] is None or a_c[j] is None:
#                 pos_sim = 0.0
#             else:
#                 d = np.linalg.norm(b_c[i] - a_c[j])
#                 pos_sim = max(0.0, 1.0 - (d / (diag * 0.02)))  # 예: 대각선의 2% 내면 거의 1

#             # 3) 면적 유사도 (0~1, 비율 가까울수록 1)
#             if b_a[i] == 0 or a_a[j] == 0:
#                 area_sim = 0.0
#             else:
#                 r = min(b_a[i], a_a[j]) / max(b_a[i], a_a[j])
#                 area_sim = float(r)  # 같은 크기면 1, 2배 차이면 0.5

#             S[i, j] = w_iou*iou + w_pos*pos_sim + w_area*area_sim

#     row_ind, col_ind = linear_sum_assignment(1.0 - S)

#     matched_b, matched_a = set(), set()
#     for i, j in zip(row_ind, col_ind):
#         # 최종 수락 기준: 주로 IoU 기준을 한 번 더 확인(안전장치)
#         if dilated_mask_iou(before_masks[i], after_masks[j], tol_px) >= iou_thresh:
#             matched_b.add(i); matched_a.add(j)

#     disappeared = [i for i in range(B) if i not in matched_b]
#     appeared    = [j for j in range(A) if j not in matched_a]
#     return disappeared, appeared



# def map_det_to_canon_by_feat(det_idx: int, gobs: dict, cls_arr, 
#                              canon_vis_n_t: torch.Tensor,   # (N_canon, D)
#                              canon_class: np.ndarray,
#                              device: str = "cuda",
#                              sim_thresh: float = 0.25):
#     """
#     반환: 매칭된 전역 object row index(int) 또는 None
#     """
#     # 1) 클래스 필터
#     det_cls = int(gobs["class_id"][det_idx])
#     if det_cls < 0:
#         return None
#     cand_rows = np.where(canon_class == det_cls)[0]
#     if cand_rows.size == 0:
#         return None

#     # 2) query feature (gobs["image_feats"][det_idx]) -> torch, L2 정규화
#     feat = gobs.get("image_feats", None)
#     if feat is None or len(feat) == 0:
#         return None
#     v_np = np.asarray(feat[det_idx])
#     if v_np.ndim == 0 or v_np.size == 0:
#         return None
#     v_t = torch.as_tensor(v_np, dtype=torch.float32, device=device)
#     v_t = l2norm_torch(v_t)

#     # 3) 코사인 유사도: (|cand|, D) x (D,) -> (|cand|)
#     cand_rows_t = torch.as_tensor(cand_rows, dtype=torch.long, device=device)
#     cand_bank = canon_vis_n_t.index_select(dim=0, index=cand_rows_t)  # (C, D)
#     sims = torch.mv(cand_bank, v_t)  # (C,)

#     # 4) 선택
#     sim_max, idx_max = torch.max(sims, dim=0)
#     if sim_max.item() < sim_thresh:
#         return None
#     best_row = int(cand_rows[idx_max.item()])
#     return best_row


# 베이스라인이 만들던 results dict 키와 동일 (저장 직전 구성)  :contentReference[oaicite:1]{index=1}
_RESULTS_KEYS = [
    "xyxy", "confidence", "class_id", "mask",
    "classes", "image_crops", "image_feats", "text_feats",
    "detection_class_labels", "labels", "edges", "captions",
]


# def _load_npz(path: Path):
#     with np.load(path) as data:
#         # 보통 배열 하나만 저장됨
#         if len(data.files) == 1:
#             return data[data.files[0]]
#         return {k: data[k] for k in data.files}

def _load_npz(path: Path):
    with np.load(path, mmap_mode='r') as data:  # ← mmap
        if len(data.files) == 1:
            return data[data.files[0]]
        return {k: data[k] for k in data.files}

def _load_pkl_gz(path: Path):
    with gzip.open(path, "rb") as f:
        return pickle.load(f)

def load_before_frame_as_results(frame_dir: Path) -> dict:
    """
    frame_dir 예: /.../before_results/frame000123
    반환: 베이스라인의 `results` 딕셔너리(키/타입/shape 최대한 동일화)
    """
    if not frame_dir.exists():
        raise FileNotFoundError(f"[before_results] missing frame dir: {frame_dir}")

    out = {}

    # 1) npz로 저장된 키들
    for k, fname in {
        "xyxy": "xyxy.npz",
        "confidence": "confidence.npz",
        "class_id": "class_id.npz",
        "mask": "mask.npz",
        "image_feats": "image_feats.npz",
    }.items():
        f = frame_dir / fname
        if f.exists():
            out[k] = _load_npz(f)

    # 2) pkl.gz로 저장된 키들
    for k, fname in {
        "classes": "classes.pkl.gz",
        "image_crops": "image_crops.pkl.gz",
        "text_feats": "text_feats.pkl.gz",
        "detection_class_labels": "detection_class_labels.pkl.gz",
        "labels": "labels.pkl.gz",
        "edges": "edges.pkl.gz",
        "captions": "captions.pkl.gz",
    }.items():
        f = frame_dir / fname
        if f.exists():
            out[k] = _load_pkl_gz(f)

    # 3) 기본값 채우기(누락 키 대비)
    defaults = {
        "classes": [], "image_crops": [], "text_feats": [],
        "detection_class_labels": [], "labels": [], "edges": [], "captions": [],
    }
    for k, v in defaults.items():
        out.setdefault(k, v)

    # 4) dtype/shape 정돈 (downstream에서 바로 사용 가능하도록)
    if "class_id" in out and isinstance(out["class_id"], np.ndarray):
        out["class_id"] = out["class_id"].astype(int)

    # mask: (N,H,W,1) → (N,H,W)
    if "mask" in out and isinstance(out["mask"], np.ndarray):
        m = out["mask"]
        if m.ndim == 4 and m.shape[-1] == 1:
            out["mask"] = m[..., 0]

    # 5) 1차원 길이 정합(검출 수 N 기준으로 맞춤)
    N = int(len(out.get("class_id", []))) if isinstance(out.get("class_id"), np.ndarray) else 0

    def _trim_first_dim(arr, n):
        try:
            if hasattr(arr, "shape") and arr.shape and arr.shape[0] != n:
                return arr[:n]
        except Exception:
            pass
        return arr

    for k in ["xyxy", "confidence", "mask", "image_feats"]:
        if k in out and isinstance(out[k], np.ndarray):
            out[k] = _trim_first_dim(out[k], N)

    # 6) 최종 키 세트 보장
    for k in _RESULTS_KEYS:
        out.setdefault(k, [] if k not in ("xyxy","confidence","class_id","mask","image_feats") else
                          np.zeros((0,), dtype=float))

    return out

def make_frame_name(color_path_stem: str, idx: int) -> str:
    """dataset.color_paths[frame_idx].stem 이 숫자면 그걸 쓰고, 아니면 frame%06d"""
    return f"frame{int(color_path_stem):06d}" if color_path_stem.isdigit() else f"frame{idx:06d}"


## 여기까지

def save_on_edges_simple(exp_suffix, exp_out_path, edges):
    """
    edges : [{'subj': 'object_i', 'rel':'on', 'obj':'object_j'}, ...]
    """
    out_path = Path(exp_out_path) / f"on_edges_{exp_suffix}.json"
    out_path.parent.mkdir(exist_ok=True, parents=True)
    
    # list 그대로 떨굼
    with open(out_path, 'w') as f:
        json.dump(edges, f, indent=2)
    
    print(f"Saved on-edge JSON to {out_path}")

# Disable torch gradient computation
torch.set_grad_enabled(False)

# A logger for this file
@hydra.main(version_base=None, config_path="../hydra_configs/", config_name="rerun_realtime_mapping")
# @profile
def main(cfg : DictConfig):

    tracker = MappingTracker()
    
    # # ==== 최소화 세팅 강제 ====
    # try:
    #     cfg.use_rerun = False
    #     cfg.vis_render = False
    #     cfg.use_wandb = False
    #     cfg.save_detections = False
    #     cfg.save_objects_all_frames = False
    #     cfg.save_video = False
    # except Exception:
    #     pass

    # if cfg.use_rerun:
    #     prev_adjusted_pose = orr_log_camera(intrinsics, adjusted_pose, prev_adjusted_pose, cfg.image_width, cfg.image_height, frame_idx)
    #     orr_log_rgb_image(color_path)
    #     orr_log_annotated_image(color_path, det_exp_vis_path)
    #     orr_log_depth_image(depth_tensor)
    #     orr_log_vlm_image(vis_save_path_for_vlm)
    #     orr_log_vlm_image(vis_save_path_for_vlm_edges, label="w_edges")


    
    orr = OptionalReRun()
    orr.set_use_rerun(cfg.use_rerun)
    orr.init("realtime_mapping")
    orr.spawn()

    owandb = OptionalWandB()
    owandb.set_use_wandb(cfg.use_wandb)
    owandb.init(project="concept-graphs", 
            #    entity="concept-graphs",
                config=cfg_to_dict(cfg),
               )
    cfg = process_cfg(cfg)

    # Initialize the dataset
    dataset = get_dataset(
        dataconfig=cfg.dataset_config,
        start=cfg.start,
        end=cfg.end,
        stride=cfg.stride,
        basedir=cfg.dataset_root,
        sequence=cfg.scene_id,
        desired_height=cfg.image_height,
        desired_width=cfg.image_width,
        device="cpu",
        dtype=torch.float,
    )
    # cam_K = dataset.get_cam_K()

    pre_dataset = get_dataset(
        dataconfig=cfg.dataset_config,
        start=cfg.start,
        end=cfg.end,
        stride=cfg.stride,
        basedir="/home/pchy0316/dataset/before_local_data/Replica",
        sequence=cfg.scene_id,
        desired_height=cfg.image_height,
        desired_width=cfg.image_width,
        device="cpu",
        dtype=torch.float,
    )

    objects = MapObjectList(device=cfg.device)
    before_objects = MapObjectList(device=cfg.device)
    map_edges = MapEdgeMapping(objects)
    before_map_edges = MapEdgeMapping(objects)
    
    # For visualization
    if cfg.vis_render: #cfg.vis_render=False
        view_param = read_pinhole_camera_parameters(cfg.render_camera_path)
        obj_renderer = OnlineObjectRenderer(
            view_param = view_param,
            base_objects = None, 
            gray_map = False,
        )
        frames = []
    # output folder for this mapping experiment
    exp_out_path = get_exp_out_path(cfg.dataset_root, cfg.scene_id, cfg.exp_suffix)

    # output folder of the detections experiment to use
    det_exp_path = get_exp_out_path(cfg.dataset_root, cfg.scene_id, cfg.detections_exp_suffix, make_dir=False)

    # we need to make sure to use the same classes as the ones used in the detections
    detections_exp_cfg = cfg_to_dict(cfg)
    obj_classes = ObjectClasses(
        classes_file_path=detections_exp_cfg['classes_file'], 
        bg_classes=detections_exp_cfg['bg_classes'], 
        skip_bg=detections_exp_cfg['skip_bg']
    )

    # if we need to do detections
    run_detections = check_run_detections(cfg.force_detection, det_exp_path)
    det_exp_pkl_path = get_det_out_path(det_exp_path)
    det_exp_vis_path = get_vis_out_path(det_exp_path)
    
    prev_adjusted_pose = None

    if run_detections:
        print("\n".join(["Running detections..."] * 10))
        det_exp_path.mkdir(parents=True, exist_ok=True)

        ## Initialize the detection models
        detection_model = measure_time(YOLO)('yolov8l-world.pt')
        sam_predictor = SAM('mobile_sam.pt') # UltraLytics SAM # SAM('sam_l.pt') 
        # sam_predictor = measure_time(get_sam_predictor)(cfg) # Normal SAM
        clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
            "ViT-H-14", "laion2b_s32b_b79k"
        )
        clip_model = clip_model.to(cfg.device)
        clip_tokenizer = open_clip.get_tokenizer("ViT-H-14")

        # Set the classes for the detection model
        detection_model.set_classes(obj_classes.get_classes_arr())
    else:
        print("\n".join(["NOT Running detections..."] * 10))

    # openai_client = get_openai_client()

    save_hydra_config(cfg, exp_out_path)
    save_hydra_config(detections_exp_cfg, exp_out_path, is_detection_config=True)

    if cfg.save_objects_all_frames:
        obj_all_frames_out_path = exp_out_path / "saved_obj_all_frames" / f"det_{cfg.detections_exp_suffix}"
        os.makedirs(obj_all_frames_out_path, exist_ok=True)

    exit_early_flag = False
    counter = 0
    
    before_flag = False
    after_flag = False
    
    
    
    # 프레임 진입
    for frame_idx in trange(len(dataset)):
        tracker.curr_frame_idx = frame_idx
        counter+=1
        orr.set_time_sequence("frame", frame_idx)

        # Check if we should exit early only if the flag hasn't been set yet
        if not exit_early_flag and should_exit_early(cfg.exit_early_file):
            print("Exit early signal detected. Skipping to the final frame...")
            exit_early_flag = True

        # If exit early flag is set and we're not at the last frame, skip this iteration
        if exit_early_flag and frame_idx < len(dataset) - 1:
            continue

        # Read info about current frame from dataset
        # color image
        color_path = Path(dataset.color_paths[frame_idx])
        pre_color_path = color_path.parent.parent / "pre_results" / color_path.name
        image_original_pil = Image.open(color_path)
        # color and depth tensors, and camera instrinsics matrix
        color_tensor, depth_tensor, intrinsics, *_ = dataset[frame_idx]

        pre_color_tensor, pre_depth_tensor, *_ = pre_dataset[frame_idx]
        

        pre_depth_array = pre_depth_tensor[...,0].cpu().numpy()
        pre_color_np = pre_color_tensor.cpu().numpy()
        pre_image_rgb = (pre_color_np).astype(np.uint8) # (H, W, 3)
        assert pre_image_rgb.max() > 1, "Image is not in range [0, 255]"
        
        # Covert to numpy and do some sanity checks
        depth_tensor = depth_tensor[..., 0]
        depth_array = depth_tensor.cpu().numpy()
        color_np = color_tensor.cpu().numpy() # (H, W, 3)
        image_rgb = (color_np).astype(np.uint8) # (H, W, 3)
        assert image_rgb.max() > 1, "Image is not in range [0, 255]"

        # Load image detections for the current frame
        raw_gobs = None
        sub_gobs = None # stands for grounded observations
        detections_path = det_exp_pkl_path / (color_path.stem + ".pkl.gz")
        
        vis_save_path_for_vlm = get_vlm_annotated_image_path(det_exp_vis_path, color_path)
        vis_save_path_for_vlm_edges = get_vlm_annotated_image_path(det_exp_vis_path, color_path, w_edges=True)
        
        if run_detections:
            results = None
            # opencv can't read Path objects...
            image = cv2.imread(str(color_path)) # This will in BGR color space
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pre_image = cv2.imread(str(pre_color_path))
            pre_image_rgb = cv2.cvtColor(pre_image, cv2.COLOR_BGR2RGB)
            # Do initial object detection
            results = detection_model.predict(color_path, conf=0.1, verbose=False)
            confidences = results[0].boxes.conf.cpu().numpy()
            detection_class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
            detection_class_labels = [f"{obj_classes.get_classes_arr()[class_id]} {class_idx}" for class_idx, class_id in enumerate(detection_class_ids)]
            xyxy_tensor = results[0].boxes.xyxy
            xyxy_np = xyxy_tensor.cpu().numpy()

            # if there are detections,
            # Get Masks Using SAM or MobileSAM
            # UltraLytics SAM
            if xyxy_tensor.numel() != 0:
                sam_out = sam_predictor.predict(color_path, bboxes=xyxy_tensor, verbose=False)
                masks_tensor = sam_out[0].masks.data

                masks_np = masks_tensor.cpu().numpy()
            else:
                masks_np = np.empty((0, *color_tensor.shape[:2]), dtype=np.float64)

            # Create a detections object that we will save later ### 이거 나중에 저장됨 
            curr_det = sv.Detections(
                xyxy=xyxy_np,
                confidence=confidences,
                class_id=detection_class_ids,
                mask=masks_np,
            )
            
            # Make the edges ### 이부분을 저장하기
            # labels, edges, edge_image, captions = make_vlm_edges_and_captions(image, curr_det, obj_classes, detection_class_labels, det_exp_vis_path, color_path, cfg.make_edges, openai_client)
            labels, edges, edge_image, captions = make_vlm_edges_and_captions(image, curr_det, obj_classes, detection_class_labels, det_exp_vis_path, color_path, False, None)
            image_crops, image_feats, text_feats = compute_clip_features_batched(
                image_rgb, curr_det, clip_model, clip_preprocess, clip_tokenizer, obj_classes.get_classes_arr(), cfg.device)

            # increment total object detections
            tracker.increment_total_detections(len(curr_det.xyxy))

            before_gobs = None
            before_good_gobs = None
            
            frame_name = make_frame_name(color_path.stem, frame_idx)
            before_frame_dir = Path(cfg.before_results_root) / frame_name
            # before_gobs = load_before_frame_as_results(before_frame_dir)
            before_gobs = load_before_by_dataset_idx(frame_idx, cfg) # 여기서 before_gobs cfg.stride 마다 읽어줌
                
            # Save results
            # Convert the detections to a dict. The elements are in np.array
            results = {
                # add new uuid for each detection 
                "xyxy": curr_det.xyxy,
                "confidence": curr_det.confidence,
                "class_id": curr_det.class_id,
                "mask": curr_det.mask,
                "classes": obj_classes.get_classes_arr(),
                "image_crops": image_crops,
                "image_feats": image_feats,
                "text_feats": text_feats,
                "detection_class_labels": detection_class_labels,
                "labels": labels,
                "edges": edges,
                "captions": captions,
            }

            raw_gobs = results 

            # 여기 수정했습니다.
            
            # save the detections if needed  # 중간 결과 필요 없으면 안하기
            if cfg.save_detections:

                vis_save_path = (det_exp_vis_path / color_path.name).with_suffix(".jpg")
                # Visualize and save the annotated image
                annotated_image, labels = vis_result_fast(image, curr_det, obj_classes.get_classes_arr())
                cv2.imwrite(str(vis_save_path), annotated_image)

                depth_image_rgb = cv2.normalize(depth_array, None, 0, 255, cv2.NORM_MINMAX)
                depth_image_rgb = depth_image_rgb.astype(np.uint8)
                depth_image_rgb = cv2.cvtColor(depth_image_rgb, cv2.COLOR_GRAY2BGR)
                annotated_depth_image, labels = vis_result_fast_on_depth(depth_image_rgb, curr_det, obj_classes.get_classes_arr())
                cv2.imwrite(str(vis_save_path).replace(".jpg", "_depth.jpg"), annotated_depth_image)
                cv2.imwrite(str(vis_save_path).replace(".jpg", "_depth_only.jpg"), depth_image_rgb)
                save_detection_results(det_exp_pkl_path / vis_save_path.stem, results)
        else:
            # Support current and old saving formats
            if os.path.exists(det_exp_pkl_path / color_path.stem):
                raw_gobs = load_saved_detections(det_exp_pkl_path / color_path.stem)
            elif os.path.exists(det_exp_pkl_path / f"{int(color_path.stem):06}"):
                raw_gobs = load_saved_detections(det_exp_pkl_path / f"{int(color_path.stem):06}")
            else:
                # if no detections, throw an error
                raise FileNotFoundError(f"No detections found for frame {frame_idx}at paths \n{det_exp_pkl_path / color_path.stem} or \n{det_exp_pkl_path / f'{int(color_path.stem):06}'}.")

        # get pose, this is the untrasformed pose.
        unt_pose = dataset.poses[frame_idx]
        unt_pose = unt_pose.cpu().numpy()

        # Don't apply any transformation otherwise
        adjusted_pose = unt_pose
        
        prev_adjusted_pose = orr_log_camera(intrinsics, adjusted_pose, prev_adjusted_pose, cfg.image_width, cfg.image_height, frame_idx)
        
        orr_log_rgb_image(color_path)
        orr_log_annotated_image(color_path, det_exp_vis_path)
        orr_log_depth_image(depth_tensor)
        orr_log_vlm_image(vis_save_path_for_vlm)
        orr_log_vlm_image(vis_save_path_for_vlm_edges, label="w_edges")

        # resize the observation if needed
        resized_gobs = resize_gobs(raw_gobs, image_rgb)
        # filter the observations
        filtered_gobs = filter_gobs(resized_gobs, image_rgb, 
            skip_bg=cfg.skip_bg,
            BG_CLASSES=obj_classes.get_bg_classes_arr(),
            mask_area_threshold=cfg.mask_area_threshold,
            max_bbox_area_ratio=cfg.max_bbox_area_ratio,
            mask_conf_threshold=cfg.mask_conf_threshold,
        )

        sub_gobs = filtered_gobs

        # resize the observation if needed
        before_resized_gobs = resize_gobs(before_gobs, pre_image_rgb)
        # filter the observations
        before_filtered_gobs = filter_gobs(before_resized_gobs, pre_image_rgb, 
            skip_bg=cfg.skip_bg,
            BG_CLASSES=obj_classes.get_bg_classes_arr(),
            mask_area_threshold=cfg.mask_area_threshold,
            max_bbox_area_ratio=cfg.max_bbox_area_ratio,
            mask_conf_threshold=cfg.mask_conf_threshold,
        )

        before_good_gobs = before_filtered_gobs
        
        # # dis_idx, app_idx 계산 직후에 추가
        # dis_idx, app_idx = match_before_after_masks_robust_gobs(
        #     before_good_gobs, sub_gobs,
        #     tol_px=2, iou_thresh=0.5
        # )

        # 수정된 3D 매칭 코드로 교체
        dis_idx, app_idx = match_before_after_3d(
            before_good_gobs, sub_gobs, 
            adjusted_pose, adjusted_pose, 
            pre_depth_array, depth_array,
            pre_image_rgb,
            image_rgb,
            intrinsics,
            pre_color_path,
            color_path,
            obj_classes,
            frame_idx,
            cfg.obj_pcd_max_points,
            overlap_thresh=0.5
        )


        sub_before_gobs = None
        gobs = None
        sub_before_gobs = subset_gobs(before_good_gobs, dis_idx)
        gobs = subset_gobs(sub_gobs, app_idx)

        # ──[추가 시작]────────────────────────────────────────────

        # 프레임 이름 만들기 (숫자면 frame%06d 포맷, 아니면 그대로 사용)
        # frame_name = f"frame{frame_idx:06d}"

        # # 저장 폴더: exp_out_path/frame_changes
        # change_dir = Path(exp_out_path) / "frame_changes"
        # change_dir.mkdir(parents=True, exist_ok=True)

        # # 프레임별 개별 파일
        # dis_path = change_dir / f"{frame_name}_disappeared.txt"
        # app_path = change_dir / f"{frame_name}_appeared.txt"

        # with open(dis_path, "w") as f:
        #     if len(dis_idx) > 0:
        #         f.write("\n".join(map(str, dis_idx)) + "\n")

        # with open(app_path, "w") as f:
        #     if len(app_idx) > 0:
        #         f.write("\n".join(map(str, app_idx)) + "\n")

        # # 실행 전체 요약 파일 (append 모드)
        # all_dis_path = change_dir / "disappeared_all.txt"
        # all_app_path = change_dir / "appeared_all.txt"

        # with open(all_dis_path, "a") as f:
        #     f.write(f"{frame_name}: " + (",".join(map(str, dis_idx)) if dis_idx else "") + "\n")

        # with open(all_app_path, "a") as f:
        #     f.write(f"{frame_name}: " + (",".join(map(str, app_idx)) if app_idx else "") + "\n")
        # ──[추가 끝]──────────────────────────────────────────────


        before_gobs_flag = False
        while not before_gobs_flag:
            
            before_gobs_flag = True
            
            if len(sub_before_gobs['mask'])==0:
                continue
            
            sub_before_gobs['mask'] = mask_subtract_contained(sub_before_gobs['xyxy'], sub_before_gobs['mask'])
            
            before_obj_pcds_and_bboxes = measure_time(detections_to_obj_pcd_and_bbox)(
                depth_array=pre_depth_array,
                masks=sub_before_gobs['mask'],
                cam_K=intrinsics.cpu().numpy()[:3, :3],  # Camera intrinsics
                image_rgb=pre_image_rgb,
                trans_pose=adjusted_pose,
                min_points_threshold=cfg.min_points_threshold,
                spatial_sim_type=cfg.spatial_sim_type,
                obj_pcd_max_points=cfg.obj_pcd_max_points,
                device=cfg.device,
            )

            for pre_obj in before_obj_pcds_and_bboxes:
                if pre_obj:
                    pre_obj["pcd"] = init_process_pcd(
                        pcd=pre_obj["pcd"],
                        downsample_voxel_size=cfg["downsample_voxel_size"],
                        dbscan_remove_noise=cfg["dbscan_remove_noise"],
                        dbscan_eps=cfg["dbscan_eps"],
                        dbscan_min_points=cfg["dbscan_min_points"],
                    )
                    pre_obj["bbox"] = get_bounding_box(
                        spatial_sim_type=cfg['spatial_sim_type'], 
                        pcd=pre_obj["pcd"],
                    )

            before_detection_list = make_detection_list_from_pcd_and_gobs( # 이게 중요한듯
                before_obj_pcds_and_bboxes, sub_before_gobs, pre_color_path, obj_classes, frame_idx
            )
            
            if len(before_detection_list) == 0:
                continue
            
            if len(before_objects) == 0:
                before_objects.extend(before_detection_list)
                # 크게 기능성에는 영향 없어서 주석처리함
                # tracker.increment_total_objects(len(before_detection_list))
                # owandb.log({
                #         "total_objects_so_far": tracker.get_total_objects(),
                #         "objects_this_frame": len(before_detection_list),
                #     })
                continue 
            before_flag = True
            ### compute similarities and then merge
            # 원본 입니다.
            before_spatial_sim = compute_spatial_similarities(
                spatial_sim_type=cfg['spatial_sim_type'], 
                detection_list=before_detection_list, 
                objects=before_objects,
                downsample_voxel_size=cfg['downsample_voxel_size']
            )
            
            # # 여기 수정 입니다.
            # spatial_sim = compute_spatial_similarities(
            #     spatial_sim_type=cfg['spatial_sim_type'],  # 'giou'
            #     detection_list=detection_list,
            #     objects=objects,
            #     downsample_voxel_size=cfg['downsample_voxel_size']
            # )        

            before_visual_sim = compute_visual_similarities(before_detection_list, before_objects)

            before_agg_sim = aggregate_similarities(
                match_method=cfg['match_method'], 
                phys_bias=cfg['phys_bias'], 
                spatial_sim=before_spatial_sim, 
                visual_sim=before_visual_sim
            )

            # Perform matching of detections to existing objects
            before_match_indices = match_detections_to_objects(
                agg_sim=before_agg_sim, 
                detection_threshold=cfg['sim_threshold']  # Use the sim_threshold from the configuration
            )

            # Now merge the detected objects into the existing objects based on the match indices
            before_objects = merge_obj_matches(
                detection_list=before_detection_list, 
                objects=before_objects, 
                match_indices=before_match_indices,
                downsample_voxel_size=cfg['downsample_voxel_size'], 
                dbscan_remove_noise=cfg['dbscan_remove_noise'], 
                dbscan_eps=cfg['dbscan_eps'], 
                dbscan_min_points=cfg['dbscan_min_points'], 
                spatial_sim_type=cfg['spatial_sim_type'], 
                device=cfg['device']
                # Note: Removed 'match_method' and 'phys_bias' as they do not appear in the provided merge function
            )
            # fix the class names for objects
            # they should be the most popular name, not the first name
            for idx, before_obj in enumerate(before_objects):
                before_temp_class_name = before_obj["class_name"]
                before_curr_obj_class_id_counter = Counter(before_obj['class_id'])
                before_most_common_class_id = before_curr_obj_class_id_counter.most_common(1)[0][0]
                before_most_common_class_name = obj_classes.get_classes_arr()[before_most_common_class_id]
                if before_temp_class_name != before_most_common_class_name:
                    before_obj["class_name"] = before_most_common_class_name

            before_map_edges = process_edges(before_match_indices, sub_before_gobs, len(before_objects), before_objects, before_map_edges, frame_idx) ## 이것도 나중에 수정
            is_before_final_frame = frame_idx == len(dataset) - 1

            if is_before_final_frame:
                print("Final frame detected. Performing final post-processing...")

            # Clean up outlier edges
            edges_to_delete = []
            for curr_map_edge in before_map_edges.edges_by_index.values():
                curr_obj1_idx = curr_map_edge.obj1_idx
                curr_obj2_idx = curr_map_edge.obj2_idx
                obj1_class_name = before_objects[curr_obj1_idx]['class_name'] 
                obj2_class_name = before_objects[curr_obj2_idx]['class_name']
                curr_first_detected = curr_map_edge.first_detected
                curr_num_det = curr_map_edge.num_detections
                if (frame_idx - curr_first_detected > 5) and curr_num_det < 2:
                    edges_to_delete.append((curr_obj1_idx, curr_obj2_idx))
            for edge in edges_to_delete:
                before_map_edges.delete_edge(edge[0], edge[1])
            ### Perform post-processing periodically if told so

            # Denoising
            if processing_needed(
                cfg["denoise_interval"],
                cfg["run_denoise_final_frame"],
                frame_idx,
                is_before_final_frame,
            ):
                before_objects = measure_time(denoise_objects)(
                    downsample_voxel_size=cfg['downsample_voxel_size'], 
                    dbscan_remove_noise=cfg['dbscan_remove_noise'], 
                    dbscan_eps=cfg['dbscan_eps'], 
                    dbscan_min_points=cfg['dbscan_min_points'], 
                    spatial_sim_type=cfg['spatial_sim_type'], 
                    device=cfg['device'], 
                    objects=before_objects
                )

            # Filtering
            if processing_needed(
                cfg["filter_interval"],
                cfg["run_filter_final_frame"],
                frame_idx,
                is_before_final_frame,
            ):
                before_objects = filter_objects(
                    obj_min_points=cfg['obj_min_points'], 
                    obj_min_detections=cfg['obj_min_detections'], 
                    objects=before_objects,
                    map_edges=before_map_edges
                )

            # Merging
            if processing_needed(
                cfg["merge_interval"],
                cfg["run_merge_final_frame"],
                frame_idx,
                is_before_final_frame,
            ):
                before_objects, before_map_edges = measure_time(merge_objects)(
                    merge_overlap_thresh=cfg["merge_overlap_thresh"],
                    merge_visual_sim_thresh=cfg["merge_visual_sim_thresh"],
                    merge_text_sim_thresh=cfg["merge_text_sim_thresh"],
                    objects=before_objects,
                    downsample_voxel_size=cfg["downsample_voxel_size"],
                    dbscan_remove_noise=cfg["dbscan_remove_noise"],
                    dbscan_eps=cfg["dbscan_eps"],
                    dbscan_min_points=cfg["dbscan_min_points"],
                    spatial_sim_type=cfg["spatial_sim_type"],
                    device=cfg["device"],
                    do_edges=cfg["make_edges"],
                    map_edges=before_map_edges
                )




        gobs_flag = False
        while not gobs_flag:
            
            gobs_flag = True
            
            if len(gobs['mask']) == 0: # no detections in this frame
                continue
            
            gobs['mask'] = mask_subtract_contained(gobs['xyxy'], gobs['mask'])

            obj_pcds_and_bboxes = measure_time(detections_to_obj_pcd_and_bbox)(
                depth_array=depth_array,
                masks=gobs['mask'],
                cam_K=intrinsics.cpu().numpy()[:3, :3],  # Camera intrinsics
                image_rgb=image_rgb,
                trans_pose=adjusted_pose,
                min_points_threshold=cfg.min_points_threshold,
                spatial_sim_type=cfg.spatial_sim_type,
                obj_pcd_max_points=cfg.obj_pcd_max_points,
                device=cfg.device,
            )

            for obj in obj_pcds_and_bboxes:
                if obj:
                    obj["pcd"] = init_process_pcd(
                        pcd=obj["pcd"],
                        downsample_voxel_size=cfg["downsample_voxel_size"],
                        dbscan_remove_noise=cfg["dbscan_remove_noise"],
                        dbscan_eps=cfg["dbscan_eps"],
                        dbscan_min_points=cfg["dbscan_min_points"],
                    )
                    obj["bbox"] = get_bounding_box(
                        spatial_sim_type=cfg['spatial_sim_type'], 
                        pcd=obj["pcd"],
                    )

            detection_list = make_detection_list_from_pcd_and_gobs( # 이게 중요한듯
                obj_pcds_and_bboxes, gobs, color_path, obj_classes, frame_idx
            )
        

            if len(detection_list) == 0: # no detections, skip
                continue

            # if no objects yet in the map,
            # just add all the objects from the current frame
            # then continue, no need to match or merge
            if len(objects) == 0:
                objects.extend(detection_list)
                tracker.increment_total_objects(len(detection_list))
                owandb.log({
                        "total_objects_so_far": tracker.get_total_objects(),
                        "objects_this_frame": len(detection_list),
                    })
                continue 
            # 한프레임에 대한것...
            after_flag = True
            ### compute similarities and then merge
            # 원본 입니다.
            spatial_sim = compute_spatial_similarities(
                spatial_sim_type=cfg['spatial_sim_type'], 
                detection_list=detection_list, 
                objects=objects,
                downsample_voxel_size=cfg['downsample_voxel_size']
            )
            
            # # 여기 수정 입니다.
            # spatial_sim = compute_spatial_similarities(
            #     spatial_sim_type=cfg['spatial_sim_type'],  # 'giou'
            #     detection_list=detection_list,
            #     objects=objects,
            #     downsample_voxel_size=cfg['downsample_voxel_size']
            # )        

            visual_sim = compute_visual_similarities(detection_list, objects)

            agg_sim = aggregate_similarities(
                match_method=cfg['match_method'], 
                phys_bias=cfg['phys_bias'], 
                spatial_sim=spatial_sim, 
                visual_sim=visual_sim
            )

            # Perform matching of detections to existing objects
            match_indices = match_detections_to_objects(
                agg_sim=agg_sim, 
                detection_threshold=cfg['sim_threshold']  # Use the sim_threshold from the configuration
            )

            # Now merge the detected objects into the existing objects based on the match indices
            objects = merge_obj_matches(
                detection_list=detection_list, 
                objects=objects, 
                match_indices=match_indices,
                downsample_voxel_size=cfg['downsample_voxel_size'], 
                dbscan_remove_noise=cfg['dbscan_remove_noise'], 
                dbscan_eps=cfg['dbscan_eps'], 
                dbscan_min_points=cfg['dbscan_min_points'], 
                spatial_sim_type=cfg['spatial_sim_type'], 
                device=cfg['device']
                # Note: Removed 'match_method' and 'phys_bias' as they do not appear in the provided merge function
            )
            # fix the class names for objects
            # they should be the most popular name, not the first name
            for idx, obj in enumerate(objects):
                temp_class_name = obj["class_name"]
                curr_obj_class_id_counter = Counter(obj['class_id'])
                most_common_class_id = curr_obj_class_id_counter.most_common(1)[0][0]
                most_common_class_name = obj_classes.get_classes_arr()[most_common_class_id]
                if temp_class_name != most_common_class_name:
                    obj["class_name"] = most_common_class_name

            map_edges = process_edges(match_indices, gobs, len(objects), objects, map_edges, frame_idx) ## 이것도 나중에 수정
            is_final_frame = frame_idx == len(dataset) - 1
            if is_final_frame:
                print("Final frame detected. Performing final post-processing...")

            # Clean up outlier edges
            edges_to_delete = []
            for curr_map_edge in map_edges.edges_by_index.values():
                curr_obj1_idx = curr_map_edge.obj1_idx
                curr_obj2_idx = curr_map_edge.obj2_idx
                obj1_class_name = objects[curr_obj1_idx]['class_name'] 
                obj2_class_name = objects[curr_obj2_idx]['class_name']
                curr_first_detected = curr_map_edge.first_detected
                curr_num_det = curr_map_edge.num_detections
                if (frame_idx - curr_first_detected > 5) and curr_num_det < 2:
                    edges_to_delete.append((curr_obj1_idx, curr_obj2_idx))
            for edge in edges_to_delete:
                map_edges.delete_edge(edge[0], edge[1])
            ### Perform post-processing periodically if told so

            # Denoising
            if processing_needed(
                cfg["denoise_interval"],
                cfg["run_denoise_final_frame"],
                frame_idx,
                is_final_frame,
            ):
                objects = measure_time(denoise_objects)(
                    downsample_voxel_size=cfg['downsample_voxel_size'], 
                    dbscan_remove_noise=cfg['dbscan_remove_noise'], 
                    dbscan_eps=cfg['dbscan_eps'], 
                    dbscan_min_points=cfg['dbscan_min_points'], 
                    spatial_sim_type=cfg['spatial_sim_type'], 
                    device=cfg['device'], 
                    objects=objects
                )

            # Filtering
            if processing_needed(
                cfg["filter_interval"],
                cfg["run_filter_final_frame"],
                frame_idx,
                is_final_frame,
            ):

                objects = filter_objects(
                    obj_min_points=cfg['obj_min_points'], 
                    obj_min_detections=cfg['obj_min_detections'], 
                    objects=objects,
                    map_edges=map_edges
                )

            # Merging
            if processing_needed(
                cfg["merge_interval"],
                cfg["run_merge_final_frame"],
                frame_idx,
                is_final_frame,
            ):
                objects, map_edges = measure_time(merge_objects)(
                    merge_overlap_thresh=cfg["merge_overlap_thresh"],
                    merge_visual_sim_thresh=cfg["merge_visual_sim_thresh"],
                    merge_text_sim_thresh=cfg["merge_text_sim_thresh"],
                    objects=objects,
                    downsample_voxel_size=cfg["downsample_voxel_size"],
                    dbscan_remove_noise=cfg["dbscan_remove_noise"],
                    dbscan_eps=cfg["dbscan_eps"],
                    dbscan_min_points=cfg["dbscan_min_points"],
                    spatial_sim_type=cfg["spatial_sim_type"],
                    device=cfg["device"],
                    do_edges=cfg["make_edges"],
                    map_edges=map_edges
                )
            orr_log_objs_pcd_and_bbox(objects, obj_classes)
            orr_log_edges(objects, map_edges, obj_classes)

            #이거는 프레임마다 중간 기록 덤핑해서 저장하는 거
            if cfg.save_objects_all_frames:
                save_objects_for_frame(
                    obj_all_frames_out_path,
                    frame_idx,
                    objects,
                    cfg.obj_min_detections,
                    adjusted_pose,
                    color_path
                )
            # 디버깅용 시각화
            if cfg.vis_render:
                # render a frame, if needed (not really used anymore since rerun)
                vis_render_image(
                    objects,
                    obj_classes,
                    obj_renderer,
                    image_original_pil,
                    adjusted_pose,
                    frames,
                    frame_idx,
                    color_path,
                    cfg.obj_min_detections,
                    cfg.class_agnostic,
                    cfg.debug_render,
                    is_final_frame,
                    cfg.exp_out_path,
                    cfg.exp_suffix,
                )

            if cfg.periodically_save_pcd and (counter % cfg.periodically_save_pcd_interval == 0):
                # save the pointcloud
                save_pointcloud(
                    exp_suffix=cfg.exp_suffix,
                    exp_out_path=exp_out_path,
                    cfg=cfg,
                    objects=objects,
                    obj_classes=obj_classes,
                    latest_pcd_filepath=cfg.latest_pcd_filepath,
                    create_symlink=True
                )

            owandb.log({
                "frame_idx": frame_idx,
                "counter": counter,
                "exit_early_flag": exit_early_flag,
                "is_final_frame": is_final_frame,
            })

            tracker.increment_total_objects(len(objects))
            tracker.increment_total_detections(len(detection_list))
            owandb.log({
                    "total_objects": tracker.get_total_objects(),
                    "objects_this_frame": len(objects),
                    "total_detections": tracker.get_total_detections(),
                    "detections_this_frame": len(detection_list),
                    "frame_idx": frame_idx,
                    "counter": counter,
                    "exit_early_flag": exit_early_flag,
                    "is_final_frame": is_final_frame,
                    })


            # orr_log_objs_pcd_and_bbox(before_objects, obj_classes)
            # orr_log_edges(before_objects, before_map_edges, obj_classes)

            #이거는 프레임마다 중간 기록 덤핑해서 저장하는 거
            if cfg.save_objects_all_frames:
                save_objects_for_frame(
                    obj_all_frames_out_path,
                    frame_idx,
                    objects,
                    cfg.obj_min_detections,
                    adjusted_pose,
                    color_path
                )
            # 디버깅용 시각화
            if cfg.vis_render:
                # render a frame, if needed (not really used anymore since rerun)
                vis_render_image(
                    objects,
                    obj_classes,
                    obj_renderer,
                    image_original_pil,
                    adjusted_pose,
                    frames,
                    frame_idx,
                    color_path,
                    cfg.obj_min_detections,
                    cfg.class_agnostic,
                    cfg.debug_render,
                    is_final_frame,
                    cfg.exp_out_path,
                    cfg.exp_suffix,
                )

            if cfg.periodically_save_pcd and (counter % cfg.periodically_save_pcd_interval == 0):
                # save the pointcloud
                save_pointcloud(
                    exp_suffix=cfg.exp_suffix,
                    exp_out_path=exp_out_path,
                    cfg=cfg,
                    objects=objects,
                    obj_classes=obj_classes,
                    latest_pcd_filepath=cfg.latest_pcd_filepath,
                    create_symlink=True
                )

            owandb.log({
                "frame_idx": frame_idx,
                "counter": counter,
                "exit_early_flag": exit_early_flag,
                "is_final_frame": is_final_frame,
            })

            tracker.increment_total_objects(len(objects))
            tracker.increment_total_detections(len(detection_list))
            owandb.log({
                    "total_objects": tracker.get_total_objects(),
                    "objects_this_frame": len(objects),
                    "total_detections": tracker.get_total_detections(),
                    "detections_this_frame": len(detection_list),
                    "frame_idx": frame_idx,
                    "counter": counter,
                    "exit_early_flag": exit_early_flag,
                    "is_final_frame": is_final_frame,
                    })
                        


                        
        #### 여기까지가 사라진 객체 생성된 객체 구하는 것이었습니다. 여기까지 작성하였습니다.
        ##########

 
    before_objects = measure_time(denoise_objects)(
                    downsample_voxel_size=cfg['downsample_voxel_size'], 
                    dbscan_remove_noise=cfg['dbscan_remove_noise'], 
                    dbscan_eps=cfg['dbscan_eps'], 
                    dbscan_min_points=cfg['dbscan_min_points'], 
                    spatial_sim_type=cfg['spatial_sim_type'], 
                    device=cfg['device'], 
                    objects=before_objects
                )

            # Filtering
    before_objects = filter_objects(
                    obj_min_points=cfg['obj_min_points'], 
                    obj_min_detections=cfg['obj_min_detections'], 
                    objects=before_objects,
                    map_edges=before_map_edges
                )

            # Merging
    before_objects, before_map_edges = measure_time(merge_objects)(
                    merge_overlap_thresh=cfg["merge_overlap_thresh"],
                    merge_visual_sim_thresh=cfg["merge_visual_sim_thresh"],
                    merge_text_sim_thresh=cfg["merge_text_sim_thresh"],
                    objects=before_objects,
                    downsample_voxel_size=cfg["downsample_voxel_size"],
                    dbscan_remove_noise=cfg["dbscan_remove_noise"],
                    dbscan_eps=cfg["dbscan_eps"],
                    dbscan_min_points=cfg["dbscan_min_points"],
                    spatial_sim_type=cfg["spatial_sim_type"],
                    device=cfg["device"],
                    do_edges=cfg["make_edges"],
                    map_edges=before_map_edges
                )
    

    objects = measure_time(denoise_objects)(
                    downsample_voxel_size=cfg['downsample_voxel_size'], 
                    dbscan_remove_noise=cfg['dbscan_remove_noise'], 
                    dbscan_eps=cfg['dbscan_eps'], 
                    dbscan_min_points=cfg['dbscan_min_points'], 
                    spatial_sim_type=cfg['spatial_sim_type'], 
                    device=cfg['device'], 
                    objects=objects
                )

    objects = filter_objects(
                    obj_min_points=cfg['obj_min_points'], 
                    obj_min_detections=cfg['obj_min_detections'], 
                    objects=objects,
                    map_edges=map_edges
                )

    objects, map_edges = measure_time(merge_objects)(
                    merge_overlap_thresh=cfg["merge_overlap_thresh"],
                    merge_visual_sim_thresh=cfg["merge_visual_sim_thresh"],
                    merge_text_sim_thresh=cfg["merge_text_sim_thresh"],
                    objects=objects,
                    downsample_voxel_size=cfg["downsample_voxel_size"],
                    dbscan_remove_noise=cfg["dbscan_remove_noise"],
                    dbscan_eps=cfg["dbscan_eps"],
                    dbscan_min_points=cfg["dbscan_min_points"],
                    spatial_sim_type=cfg["spatial_sim_type"],
                    device=cfg["device"],
                    do_edges=cfg["make_edges"],
                    map_edges=map_edges
                )
    orr_log_objs_pcd_and_bbox(objects, obj_classes)
    orr_log_edges(objects, map_edges, obj_classes)


    if(after_flag == False):
        objects = MapObjectList([])
    
    if(before_flag == False):
        before_objects = MapObjectList([])
        


    # LOOP OVER -----------------------------------------------------
            
    # ------------------------ on between objects ----------------------- #
    
    # 여기서 너가 위 loop를 통해서 얻은 objects 정보를 활용하여 객체간의 관계를 Rule base로 정하는 on_edges를 생성해줘(여기서 생성하는 map_edges는 무시에 rule base아닌 다른 방법이닌까)
    # ------------------------ on between objects ----------------------- #
    # 규칙 기반 on-edge 생성 (y-up, 바닥 y=0, footprint=XZ)
    # - 바닥 접촉(lower_y < floor_thresh) 객체는 floor에 on
    # - 나머지는 footprint overlap이 충분하고 더 낮은(lower_y) 지지체를 부모로 on
    # - 지지체가 여러 개면: (1) 고정물(is_fixed) 후보 우선, (2) 그중 y-거리(Δy)가 가장 작은 것
    # - 지지체가 없으면 floor로 on

    # 데이터 load하기 SG, footprint, lower 
    
    # sg_objects를 설계: lower_h, footprint(alpha shape)

    
    on_edges = [] # on_edge똑같이 생서
    base = Path("/home/pchy0316/dataset/my_local_data/Replica/room0/poly_lower_SG")
    lower_h, poly_s = load_lower_and_polys(base / "polys_r_mapping_stride10.npz", as_shapely=True) # 여기 수정 높이, poly 저장 # lower_h (24,)
    sg_objects = load_sg_objects(base) # 여기 수정 "is_fixed", "parent", "anchor" 이렇게 하면 edge 생성도 쉽고
    
    #이걸로 우선 scene graph 업데이트 편한 상태로 만들고, 업데이트하기
    
    # 삭제하기 before_objects
    

    # ---- 안전 import ----
    try:
        import alphashape
        HAS_ALPHASHAPE = True
    except Exception:
        HAS_ALPHASHAPE = False
    from shapely.geometry import MultiPoint, MultiPolygon, Polygon

    # ---- 하이퍼파라미터 ----
    floor_thresh = 0.15          # (m) 바닥 접촉 임계 (y 기준)
    overlap_thresh = 0.60        # footprint 교집합 / 위(상부) 객체 footprint 면적
    jitter_std = 4e-3            # 수치 안정화용 지터(약 4mm)
    alphashape_alpha = 1.0       # concave 정도 (scene_graph_processer와 맞춤)

    # ---- 고정물(fixed) 후보 클래스 (소문자 비교; 실험 클래스와 교집합만 사용) ----
    fixed_candidates = {
        "bed", "bookshelf", "cabinet", "kitchen cabinet", "bathroom cabinet", "file cabinet",
        "couch", "sofa", "sofa chair", "fireplace",
        "counter", "kitchen counter", "bathroom counter", "desk", "dining table", "coffee table",
        "end table", "nightstand", "tv stand", "dresser", "wardrobe", "closet",
        "refrigerator", "washing machine", "dishwasher", "oven", "stove",
        "radiator", "fireplace", "piano", "column", "pillar", "stairs", "stair rail",
        "structure", "shower", "toilet", "bathtub", "coffee table"
    }
    hanging_candidates = {"blackboard", "whiteboard", "bulletin board", "calendar", "clock", "mirror", "ceiling", "blinds", "closet door",
                          "picture", "poster", "curtain", "fan", "projector screen", "projector", "wall",
                          "sign", "rack", "coat rack", "soap dispenser", "paper towel roll", "door",
                          "toilet paper dispenser", "toilet paper holder", "handicap bar", "fire alarm",
                          "fire extinguisher", "window", "power outlet", "light switch", "radiator",
                          "mailbox", "rail", "stair rail", "shower head", "ceiling light", "projector",
                          "vent", "range hood"}
    container_candidates = {
        "cabinet", "kitchen cabinet", "bathroom cabinet", "closet", "wardrobe",
        "drawer", "dresser",
        "box", "bin", "basket", "trash can", "bucket",
        "bowl", "cup", "mug", "pot", "pan", "jar",
        "sink", "toilet"
    }
    
    experiment_classes = {c.lower() for c in obj_classes.get_classes_arr()}
    fixed_class_set = {c for c in fixed_candidates if c in experiment_classes}

    # ---- 유틸 ----
    def _coverage(child_poly, anchor_poly) -> float:
        if child_poly is None or anchor_poly is None:
            return 0.0
        try:
            inter = child_poly.intersection(anchor_poly).area
            ca = child_poly.area
            return float(inter / (ca + 1e-9))
        except Exception:
            return 0.0
    
    def _obj_points(obj):
        """open3d PointCloud → (N,3) numpy"""
        pcd = obj.get("pcd", None)
        if pcd is None:
            return None
        if hasattr(pcd, "to_legacy"):
            pcd = pcd.to_legacy()
        pts = np.asarray(pcd.points)
        return pts if pts is not None and pts.size >= 3 else None

    def _lower_y(pts):
        return float(np.min(pts[:, 1]))     # y-up: 높이축 = y

    def _xz_footprint(pts, alpha=alphashape_alpha):
        """
        (x,z) 평면에서 footprint polygon 생성.
        alphashape 있으면 concave, 없으면 convex hull fallback.
        Polygon이 아니거나 면적 0이면 None.
        """
        if pts is None or pts.shape[0] < 3:
            return None
        xz = pts[:, [0, 2]]
        # NaN/Inf 제거
        mask = np.isfinite(xz).all(axis=1)
        xz = xz[mask]
        if xz.shape[0] < 3:
            return None

        # 중복 제거 + 수치 안정화
        try:
            xz_unique = np.unique(xz, axis=0)
        except Exception:
            xz_unique = np.asarray(list({(float(a), float(b)) for a, b in xz}), dtype=float)
        if xz_unique.shape[0] < 3:
            return None
        xz_unique = xz_unique + np.random.normal(0, jitter_std, xz_unique.shape)

        mp = MultiPoint(xz_unique)
        if HAS_ALPHASHAPE and xz_unique.shape[0] >= 4:
            poly = alphashape.alphashape(xz_unique, alpha=alpha)
            if isinstance(poly, MultiPolygon):
                poly = max(poly.geoms, key=lambda p: p.area)
            if poly is None or poly.is_empty:
                return None
        else:
            poly = mp.convex_hull
            if isinstance(poly, MultiPolygon):
                poly = max(poly.geoms, key=lambda p: p.area)

        return poly if isinstance(poly, Polygon) and poly.area > 0 else None

    # ---- 객체 요약 정보 수집 ----
    # idx: objects 리스트 인덱스(0-base). 엣지 JSON에선 object_{idx+1}로 사용.
    before_obj_infos = []  # dict: idx, class_name, lower_y, poly_xz, on_floor(bool), is_fixed(bool)
    
    for idx, before_obj in enumerate(before_objects):
        pts = _obj_points(before_obj)
        if pts is None:
            continue
        ly = _lower_y(pts)
        poly = _xz_footprint(pts)
        if poly is None:
            continue

        cls_name = obj.get("class_name", "").lower()
        on_floor = (ly < floor_thresh)
        # is_fixed: 바닥 접촉 + 고정물 클래스
        is_fixed = (on_floor and (cls_name in fixed_class_set))

        before_obj_infos.append({
            "idx": idx,
            "class_name": cls_name,
            "lower_y": ly,
            "poly": poly,
            "on_floor": on_floor,
            "is_fixed": is_fixed,
        })
        before_obj["is_fixed"] = bool(is_fixed)
        before_obj["lower_y"] = float(ly)

    # ---- 바닥 접촉 객체 → floor on-edge ----
    for info in before_obj_infos:
        if info["on_floor"]:
            on_edges.append({
                "subj": f"object_{info['idx']+1}",
                "rel": "on",
                "obj": "floor"
            })
    
    del_object = []
    # # 위에까지 사라진 객체 바닥인지 확인하고  
    # # lower_h, poly_s 이거 활용해서 local하게 scene graph update 예정
    # for sg_idx, sg_obj in enumerate(sg_objects):
    #     if sg_obj["is_fixed"] == True:
    #         pass
    #         anchor_obj = sg_obj["class_name"]
    #         for dy_idx, dynamic_sg_obj in enumerate(sg_objects):
    #             if(dynamic_sg_obj["anchor"] == anchor_obj):
    #                 poly_s[dy_idx]
    #                 lower_h[dy_idx]
    #         # 아래 주석에 요구상항에 맞게 작성해주세요 
    #         #poly 비교하기 poly_s[sg_idx]와 before_objects와 비교
    #         # 그 이전에 너가 poly 어느 정도 겹치면 on관계 판별했잖아 그거 이용할거야 근데 on 관계를 형성하는건 아니라 어디에 있는건지 파악하는 거지 어떤 anchor 위에 있는지
    #         # 이걸 통해서 어떤 anchor fixed object 위에 있는지 파악하고
    #         # 그 fixed_object에 있는 dynamic object(이것도 sg_objects소속입니다) 중 poly가 어느정도 일치하고, lower 값도 어느정도 일치하는 것이 사라진 객체로 판별 
    #         # 판별했으면 sg_obj["key"]를 del_object에 저장해주기


    # threshold는 cfg에서 가져오되, 없으면 기본값 사용
    anchor_poly_iou_thresh   = 0.30 # SG고정 ↔ before고정
    child_on_anchor_thresh   = 0.10 # 동적이 anchor 위에 있는지
    child_match_poly_iou_thr = 0.10 # SG동적 ↔ before동적
    lower_y_tol              = 0.10 # 3cm

    # 1) before에서 고정물이 아니고 다이나믹 object만 뽑아야 해
    before_dynamic_infos = [
        info for info in before_obj_infos
        if not info.get("is_fixed", False) and (info.get("poly") is not None)
    ]

    # 2) SG에서 anchor(key) → 동적 자식 인덱스 목록 만들기: sg_key_to_children
    sg_key_to_children = defaultdict(list)
    for cidx, cobj in enumerate(sg_objects):
        if cobj.get("is_fixed", False):
            continue 
        parent_tag = None
        a = cobj.get("anchor") or {}
        if isinstance(a, dict) and a.get("tag"):
            parent_tag = a["tag"]
        else:
            p = cobj.get("parent") or {}
            if isinstance(p, dict) and p.get("tag") and p.get("rel") in ("on", "in"):
                parent_tag = p["tag"]
        if parent_tag:
            sg_key_to_children[parent_tag].append(cidx+1)
    
    del_object = []  
    del_object_set = set()

    # SG 고정 앵커 후보(footprint 必)
    sg_anchor_indices = [
        i for i, o in enumerate(sg_objects)
        if o.get("is_fixed", False) and (poly_s[i] is not None)
    ]
    # floor key(있으면 사용)?
    # _floor_key = "floor" if any(o.get("key") == "floor" for o in sg_objects) else None
    
    
    used_dyn_indices = set() # sg_object에서 다이나믹만 남겨둔거
    
    print("생성된 갯수", len(before_dynamic_infos))
    for binfo in before_dynamic_infos:
        bpoly  = binfo.get("poly")
        blower = binfo.get("lower_y")
        if bpoly is None or blower is None:
            continue
        blower = float(blower) 

        # (a) 이 before 동적이 올라가 있을 SG 앵커 추정
        if binfo.get("on_floor", False):
            anchor_key = "floor"
        else:
            best_aidx, best_iou = None, 0.0
            for aidx in sg_anchor_indices:
                cov = _coverage(bpoly, poly_s[aidx])
                print("IOU확인하기", cov)
                if cov > best_iou:
                    best_cov, best_aidx = cov, aidx
            if best_aidx is None or best_cov < child_on_anchor_thresh:
                # 앵커를 확신할 수 없으면 스킵(원하면 floor로 폴백 가능)
                continue
            anchor_key = sg_objects[best_aidx]["key"]
            print("앵커키입니다:", anchor_key)
            #여기까지 굳


            # --- (b) 이 binfo가 올라간 '해당 앵커'의 SG 동적 후보들(1-base → 0-base 변환) ---
            child_list_1base = sg_key_to_children.get(anchor_key, [])
            child_list = [(ci - 1) if isinstance(ci, int) else ci for ci in child_list_1base]
            print(child_list)
            # --- (c) 후보 중 '최고 점수' 1개 greedy 선택 (임계치 미달이면 미선택) ---
            best_idx   = None
            best_score = -1.0
            best_iou   = 0.0
            best_dy    = 0.0

            eps     = 1e-9
            iou_thr = float(child_match_poly_iou_thr)  # 동적↔동적 매칭용 IoU 문턱
            tol     = float(lower_y_tol)               # 높이 차 문턱

            denom_iou = max(1e-9, 1.0 - iou_thr)

            for cidx in child_list:
                if not (0 <= cidx < len(poly_s)):
                    continue
                cpoly = poly_s[cidx]
                if cpoly is None:
                    continue
                clower = float(lower_h[cidx])

                iou = _poly_iou(bpoly, cpoly)
                dy  = abs(blower - clower)
                print("iou입니다", iou)
                print("dy입니다", dy)
                # 문턱 미달 후보 제외
                if iou < iou_thr:
                    continue
                if dy > tol:
                    continue

                # 정규화 점수(항상 0~1): IoU는 문턱 기준으로, 높이는 tol 기준으로
                iou_n = _clamp01((iou - iou_thr) / denom_iou)
                if tol <= 0:
                    h_n = 1.0 if dy <= eps else 0.0
                else:
                    h_n = _clamp01(1.0 - (dy / (tol + eps)))

                # 가중합(튜닝 가능)
                w_iou, w_h = 0.7, 0.3
                score = w_iou * iou_n + w_h * h_n

                if score > best_score:
                    best_score = score
                    best_idx   = cidx
                    best_iou   = iou
                    best_dy    = dy

            # --- (d) 하나 선택됐다면: 그 'SG 동적 객체'를 사라진 것으로 기록 + 후보 목록에서 제거 ---
            if best_idx is not None:
                gone_key = sg_objects[best_idx].get("key", f"object_{best_idx+1}")  # key 없으면 1-base 보정
                if gone_key not in del_object_set:
                    del_object_set.add(gone_key)
                    del_object.append(gone_key)
                # 같은 앵커에서 중복 선택 방지: 원본 맵(1-base)에서 해당 자식 제거
                try:
                    sg_key_to_children[anchor_key].remove(best_idx + 1)
                except ValueError:
                    pass
            # (선택이 없으면 이 binfo는 스킵: 제거되는 것 없음)


        # 방어로직
        best_idx_dy   = None
        best_score_dy = -1.0

        # 문턱/가중치
        iou_thr_dy = float(child_match_poly_iou_thr)   # 예: 0.25 (동적↔동적 IoU 최소)
        tol_dy     = float(lower_y_tol)                # 예: 0.03m (높이 차 허용)
        w_iou_dy, w_h_dy = 0.7, 0.3                       # 점수 가중치(튜닝 가능)
        denom_iou_dy  = max(1e-9, 1.0 - iou_thr_dy)


        for j, sobj in enumerate(sg_objects):
            if sobj.get("is_fixed", False):   # SG의 동적만 대상
                continue
            if j in used_dyn_indices:
                continue
            if j >= len(poly_s):
                continue

            cpoly_dy = poly_s[j]
            if cpoly_dy is None:
                continue
            clower_dy = float(lower_h[j])

            # 문턱 체크
            iou_dy = _poly_iou(bpoly, cpoly_dy)
            if iou_dy < iou_thr_dy:
                continue
            dy_dy = abs(blower - clower_dy)
            if dy_dy > tol_dy:
                continue

            # 0~1 정규화 점수 (음수 방지)
            iou_n_dy = _clamp01((iou_dy - iou_thr_dy) / denom_iou_dy)
            h_n_dy   = _clamp01(1.0 - (dy_dy / (tol_dy + 1e-9)))
            score_dy = w_iou_dy * iou_n_dy + w_h_dy * h_n_dy

            if score_dy > best_score_dy:
                best_score_dy = score_dy
                best_idx_dy   = j

        # 하나 선택됐다면: 그 SG 동적 객체가 '사라진 객체'
        if best_idx_dy is not None:
            gone_key = sg_objects[best_idx_dy].get("key", f"object_{best_idx_dy+1}")  # key 없으면 1-base 보정
            if gone_key not in del_object_set:
                del_object_set.add(gone_key)
                del_object.append(gone_key)
            used_dyn_indices.add(best_idx_dy)


    print("del_object########",len(del_object))
    print(del_object)
    
    del_keys = set(del_object)
    if not del_keys:
        print("[SG-UPDATE] nothing to delete")
    else:
        # 1) 기존 key→index 매핑
        key_to_idx = { o.get("key", f"object_{i+1}"): i for i, o in enumerate(sg_objects) }

        # 2) 지울 인덱스 / 남길 인덱스
        del_idx  = sorted([key_to_idx[k] for k in del_keys if k in key_to_idx])
        keep_idx = [i for i in range(len(sg_objects)) if i not in del_idx]

        # 3) sg_objects 필터링
        sg_objects = [sg_objects[i] for i in keep_idx]

        # 4) poly_s / lower_h도 같은 인덱스로 동기화
        if isinstance(poly_s, list):
            poly_s = [poly_s[i] for i in keep_idx]
        else:
            # 혹시 numpy라면
            poly_s = [poly_s[i] for i in keep_idx]
        if isinstance(lower_h, np.ndarray):
            lower_h = lower_h[keep_idx]
        else:
            lower_h = [lower_h[i] for i in keep_idx]

        # 5) 부모가 사라진 경우 parent 정리(끊기)
        for o in sg_objects:
            p = o.get("parent")
            if isinstance(p, dict) and p.get("tag") in del_keys:
                o["parent"] = None
            a = o.get("anchor")
            if isinstance(a, dict) and a.get("tag") in del_keys:
                o["anchor"] = None

        # 6) sg_key_to_children 재구성(0-base 권장)
        sg_key_to_children = defaultdict(list)
        key_to_idx = { o.get("key", f"object_{i+1}"): i for i, o in enumerate(sg_objects) }
        for cidx, cobj in enumerate(sg_objects):
            if cobj.get("is_fixed", False):
                continue
            parent_tag = None
            a = cobj.get("anchor") or {}
            if isinstance(a, dict) and a.get("tag"):
                parent_tag = a["tag"]
            else:
                p = cobj.get("parent") or {}
                if isinstance(p, dict) and p.get("tag") and p.get("rel") in ("on", "in"):
                    parent_tag = p["tag"]
            if parent_tag in key_to_idx:              # 남아있는 부모만 연결
                sg_key_to_children[parent_tag].append(cidx)

        print(f"[SG-UPDATE] deleted {len(del_idx)} nodes:", sorted(del_keys))
        print(f"[SG-UPDATE] remaining nodes: {len(sg_objects)}")

    # 제거후 중간 결과물
    save_dir = Path("/home/pchy0316/dataset/my_local_data/Replica/room0/mid_results")
    save_path = save_dir / "obj_json_r_mapping_stride10_after_delete.json"

    out_path = save_sg_json_parent_min(save_path, sg_objects)
    print("[SG] saved:", out_path)
    
    
    
    #### 여기
    # ===== 임계값 (cfg 없으면 기본값) =====
    child_on_anchor_thresh   = float(getattr(cfg, "child_on_anchor_thresh", 0.50))    # 고정/바닥 적합도(coverage)
    inside_cov_thresh        = float(getattr(cfg, "inside_cov_thresh", 0.90))         # "in" 판정 coverage (자식 기준)
    child_match_poly_iou_thr = float(getattr(cfg, "child_match_poly_iou_thr", 0.25))  # 동적↔동적 IoU
    lower_y_tol              = float(getattr(cfg, "lower_y_tol", 0.03))               # 3 cm
    floor_thresh             = float(getattr(cfg, "floor_thresh", 0.15))               # 바닥 접촉
    overlap_thresh           = float(getattr(cfg, "overlap_thresh", 0.50))             # on 후보 교차율(자식 기준)
    eps_h                    = float(getattr(cfg, "inside_height_eps", 0.02))          # "in" 높이 여유

    # ===== 유틸 =====
    def _to_set(x):
        try: return {str(s).lower() for s in x}
        except Exception: return set()

    # 컨테이너 클래스
    try:
        container_classes = {s.lower() for s in container_candidates}
    except NameError:
        container_classes = {
            "drawer","cabinet","cupboard","box","basket","bin","bowl","sink",
            "pot","pan","bucket","crate","chest","trash can","trashcan"
        }

    # 제외 클래스
    try:    fixed_cls_names   = _to_set(fixed_candidates)
    except: fixed_cls_names   = set()
    try:    hanging_cls_names = _to_set(hanging_candidates)
    except: hanging_cls_names = set()

    # floor 키
    # _floor_idx = next((i for i,o in enumerate(sg_objects) if o.get("key")=="floor"), None)
    # _floor_key = "floor" if _floor_idx is not None else None

    # 새 key/id
    def _next_object_key_and_id(sg_objs):
        nums = []
        for i, o in enumerate(sg_objs):
            k = o.get("key", f"object_{i+1}")
            m = re.search(r"(\d+)$", k)
            nums.append(int(m.group(1)) if m else (i + 1))
        nxt = (max(nums) if nums else 0) + 1
        return f"object_{nxt}", nxt

    def _top_from_center_extent(center, extent):
        try:    return float(center[1]) + float(extent[1]) * 0.5
        except: return None

    def _bottom_from_center_extent(center, extent):
        try:    return float(center[1]) - float(extent[1]) * 0.5
        except: return None

    def _aabb_from_pts(pts):
        """center/extent 없을 때 pts로 AABB 산출"""
        mn = np.min(pts, axis=0); mx = np.max(pts, axis=0)
        extent = (mx - mn).astype(float)
        center = (mx + mn).astype(float) * 0.5
        vol = float(extent[0] * extent[1] * extent[2])
        return center.tolist(), extent.tolist(), vol

    # ===== 1) 생성 후보 만들기: poly / lower_y / upper_y + 클래스 제외 =====
    gen_candidates = []
    for j, obj in enumerate(objects):
        pts = _obj_points(obj)
        if pts is None:
            continue
        ly   = _lower_y(pts)              # 기존 헬퍼 (y-up)
        poly = _xz_footprint(pts)         # 기존 헬퍼
        if poly is None:
            continue

        # upper_y: center/extent 우선, 없으면 pts로 추정
        c_center = obj.get("center")
        c_extent = obj.get("extent")
        uy = _top_from_center_extent(c_center, c_extent)
        if uy is None:
            try:    uy = float(np.max(pts[:, 1]))
            except: uy = float(ly)

        cls = str(obj.get("class_name","")).lower()

        # 클래스 기반 제외
        if cls in fixed_cls_names or cls in hanging_cls_names:
            continue

        gen_candidates.append({
            "idx": j,
            "class_name": cls,
            "lower_y": float(ly),
            "upper_y": float(uy),
            "poly": poly,
            "pts": pts,   # bbox 추정용
        })

    # ===== 1.5) 중복 제거: 기존 SG와 거의 동일(높이 유사 + IoU↑) 후보 제거 =====
    dedup_iou_thr = max(0.5, child_match_poly_iou_thr)
    remain = []
    for c in gen_candidates:
        cpoly, clower = c["poly"], c["lower_y"]
        is_dup = False
        for sidx, sobj in enumerate(sg_objects):
            if sobj.get("key") == "floor":   # 바닥 제외
                continue
            sp = poly_s[sidx] if 0 <= sidx < len(poly_s) else None
            if sp is None:
                continue
            slower = float(lower_h[sidx])
            if abs(clower - slower) > lower_y_tol:
                continue
            if _poly_iou(cpoly, sp) >= dedup_iou_thr:
                is_dup = True
                break
        if not is_dup:
            remain.append(c)

    # ===== 2) 부모 후보(전체 SG) 중 1개 선택 → on / in 결정 =====
    created_keys  = []
    created_edges = []

    for c in remain:
        cpoly   = c["poly"]
        clower  = c["lower_y"]
        cupper  = c["upper_y"]
        carea   = cpoly.area if cpoly is not None else 0.0

        best_idx   = None
        best_key   = None
        best_rel   = None
        best_score = -inf

        # 후보 탐색: 고정/바닥(coverage) 우선 평가, 동적(IoU)도 허용
        for sidx, sobj in enumerate(sg_objects):
            sp = poly_s[sidx] if 0 <= sidx < len(poly_s) else None
            if sp is None:
                continue

            slower = float(lower_h[sidx])
            dy = clower - slower
            if dy < 0:
                continue  # 부모는 항상 더 낮아야 함

            # 교차율(자식 기준)
            inter_ok = False
            if cpoly is not None and sp is not None and carea > 0:
                try:
                    inter_area = cpoly.intersection(sp).area
                    inter_rate = inter_area / carea
                    inter_ok = (inter_rate >= overlap_thresh)
                except Exception:
                    inter_ok = False

            if not inter_ok and not (sobj.get("key") == "floor"):
                continue

            # 적합도
            if sobj.get("is_fixed", False) or sobj.get("key") == "floor":
                fit = _coverage(cpoly, sp)             # 자식기준 coverage
                thr = child_on_anchor_thresh
            else:
                fit = _poly_iou(cpoly, sp)
                thr = child_match_poly_iou_thr

            if fit < thr:
                continue

            # ----- 관계(on/in) 결정 -----
            parent_cls = (sobj.get("class_name") or sobj.get("object_tag") or "").lower()
            p_center   = sobj.get("center")
            p_extent   = sobj.get("extent")
            p_top      = _top_from_center_extent(p_center, p_extent)
            p_bottom   = _bottom_from_center_extent(p_center, p_extent)
            if (p_top is None or p_bottom is None) and p_extent is not None:
                try:
                    ey = float(p_extent[1])
                    p_top    = slower + 0.5 * ey
                    p_bottom = slower - 0.5 * ey
                except Exception:
                    pass

            cov = _coverage(cpoly, sp)  # 자식 기준 coverage
            rel = "on"                  # 기본

            # 컨테이너 + coverage + 수직 범위 → "in"
            if (parent_cls in container_classes) and (cov >= inside_cov_thresh) \
            and (p_top is not None and p_bottom is not None):
                if (cupper <= p_top + eps_h) and (clower >= p_bottom - eps_h):
                    rel = "in"

            # 점수(0~1): 적합도 정규화 + 높이 근접(작을수록 좋음)
            denom = max(1e-9, 1.0 - thr)
            fit_n = _clamp01((fit - thr) / denom)
            h_n   = _clamp01(1.0 - abs(dy) / (lower_y_tol + 1e-9))
            score = 0.8 * fit_n + 0.2 * h_n

            if score > best_score:
                best_score = score
                best_idx   = sidx
                best_key   = sobj.get("key", f"object_{sidx+1}")
                best_rel   = rel

        # 부모 없으면 floor/on 또는 부모 없음  (floor는 항상 있다고 가정)
        if best_idx is None:
            best_key, best_rel = ("floor", "on") if (clower <= floor_thresh) else (None, None)


        # ===== 3) SG에 새 노드 추가 (bbox_* 포함) =====
        new_key, new_id = _next_object_key_and_id(sg_objects)

        # bbox_center/extent/volume: 원본에 있으면 사용, 없으면 pts로 계산
        src = objects[c["idx"]]
        bbox_center = src.get("center")
        bbox_extent = src.get("extent")
        bbox_volume = src.get("volume")
        if (bbox_center is None) or (bbox_extent is None) or (bbox_volume is None):
            bc, be, bv = _aabb_from_pts(c["pts"])
            bbox_center = bbox_center if bbox_center is not None else bc
            bbox_extent = bbox_extent if bbox_extent is not None else be
            bbox_volume = bbox_volume if bbox_volume is not None else bv

        new_obj = {
            "key": new_key,
            "id":  new_id,
            "object_tag": c["class_name"],
            "object_caption": [],
            "bbox_extent": bbox_extent,
            "bbox_center": bbox_center,
            "bbox_volume": bbox_volume,
            "is_fixed": False,
            "parent": {"rel": best_rel, "tag": best_key} if best_key is not None else {"rel": None, "tag": None},

            # 내부 호환 필드(있어도 무방)
            "class_name": c["class_name"],
            "extent": bbox_extent,
            "center": bbox_center,
            "volume": bbox_volume,
            "lower_y": clower,
            "upper_y": cupper,
            "anchor": src.get("anchor"),
        }

        # 메모리 동기화
        sg_objects.append(new_obj)
        if isinstance(poly_s, np.ndarray):
            poly_s = list(poly_s)
        poly_s.append(cpoly)
        lower_h = np.append(lower_h, [clower]) if isinstance(lower_h, np.ndarray) else (lower_h + [clower])

        if best_key is not None and best_rel is not None:
            created_edges.append({"subj": new_key, "rel": best_rel, "obj": best_key})
        created_keys.append(new_key)

    # ===== 4) 저장 =====
    save_path2 = save_dir / "obj_json_r_mapping_stride10_after_SG_Update.json"
    out_path2 = save_sg_json_parent_min(save_path2, sg_objects)
    print(f"[SG][create] +{len(created_keys)} nodes, saved:", out_path2)


    # # ---- 비-바닥 객체 → 지지체 탐색 ----
    # # 후보 우선순위: 고정물 지지체 > 비고정 지지체; 각 그룹 내에서는 Δy(= upper.lower_y - base.lower_y) 작은 순
    # for info_top in obj_infos:
    #     if info_top["on_floor"]:
    #         continue  # 이미 floor 처리됨

    #     poly_top = info_top["poly"]
    #     area_top = poly_top.area if poly_top else 0.0
    #     if area_top <= 0:
    #         # footprint가 없으면 안전하게 floor로
    #         on_edges.append({
    #             "subj": f"object_{info_top['idx']+1}",
    #             "rel": "on",
    #             "obj": "floor"
    #         })
    #         continue

    #     candidates_fixed = []
    #     candidates_others = []

    #     for info_base in obj_infos:
    #         if info_base["idx"] == info_top["idx"]:
    #             continue

    #         # 더 낮은(lower_y) 지지체만
    #         if info_base["lower_y"] >= info_top["lower_y"]:
    #             continue

    #         poly_base = info_base["poly"]
    #         if poly_base is None:
    #             continue
    #         if not poly_top.intersects(poly_base):
    #             continue

    #         inter_area = poly_top.intersection(poly_base).area
    #         inter_rate = (inter_area / area_top) if area_top > 0 else 0.0
    #         if inter_rate < overlap_thresh:
    #             continue

    #         dy = info_top["lower_y"] - info_base["lower_y"]  # 양수: 위-아래 수직거리(y)
    #         item = (dy, info_base)  # 정렬 키: 수직거리

    #         if info_base["is_fixed"]:
    #             candidates_fixed.append(item)
    #         else:
    #             candidates_others.append(item)

    #     chosen_parent = None
    #     if candidates_fixed:
    #         candidates_fixed.sort(key=lambda x: x[0])
    #         chosen_parent = candidates_fixed[0][1]
    #     elif candidates_others:
    #         candidates_others.sort(key=lambda x: x[0])
    #         chosen_parent = candidates_others[0][1]

    #     if chosen_parent is not None:
    #         on_edges.append({
    #             "subj": f"object_{info_top['idx']+1}",
    #             "rel": "on",
    #             "obj": f"object_{chosen_parent['idx']+1}"
    #         })
    #     else:
    #         # 지지체가 없으면 floor fallback
    #         on_edges.append({
    #             "subj": f"object_{info_top['idx']+1}",
    #             "rel": "on",
    #             "obj": "floor"
    #         })

    # 저장 (프로젝트 유틸 호출 시그니처 그대로 사용)
    # save_on_edges_simple(exp_suffix="final_test", exp_out_path=exp_out_path, edges=on_edges)
    # save_clip_feats_npy(exp_suffix = "inital", out_path = exp_out_path, objects=objects, key="clip_ft")
    # save_text_feats_npy(exp_suffix = "inital", out_path = exp_out_path, objects=objects, key="text_ft")


    # Consolidate captions 
    for object in objects:
        obj_captions = object['captions'][:20]
        # consolidated_caption = consolidate_captions(openai_client, obj_captions)
        consolidated_caption = []
        object['consolidated_caption'] = consolidated_caption


    # Consolidate captions 
    for before_object in before_objects:
        before_obj_captions = object['captions'][:20]
        # consolidated_caption = consolidate_captions(openai_client, obj_captions)
        before_consolidated_caption = []
        before_object['consolidated_caption'] = before_consolidated_caption
    

    # # Save the pointcloud
    # if cfg.save_pcd:
    #     save_pointcloud(
    #         exp_suffix=cfg.exp_suffix,
    #         exp_out_path=exp_out_path,
    #         cfg=cfg,
    #         objects=objects,
    #         obj_classes=obj_classes,
    #         latest_pcd_filepath=cfg.latest_pcd_filepath,
    #         create_symlink=True,
    #         edges=map_edges
    #     )

    before_suffix = f"{cfg.exp_suffix}_before"
    if cfg.save_json:
        save_obj_json(
            exp_suffix=cfg.exp_suffix,
            exp_out_path=exp_out_path,
            objects=objects
        )
        
        save_obj_json(
            exp_suffix=before_suffix, 
            exp_out_path=exp_out_path,
            objects=before_objects
        )
        
    #     save_edge_json(
    #         exp_suffix=cfg.exp_suffix,
    #         exp_out_path=exp_out_path,
    #         objects=objects,
    #         edges=map_edges
    #     )

    # # Save metadata if all frames are saved
    # if cfg.save_objects_all_frames:
    #     save_meta_path = obj_all_frames_out_path / f"meta.pkl.gz"
    #     with gzip.open(save_meta_path, "wb") as f:
    #         pickle.dump({
    #             'cfg': cfg,
    #             'class_names': obj_classes.get_classes_arr(),
    #             'class_colors': obj_classes.get_class_color_dict_by_index(),
    #         }, f)

    # #마지막에 각 이미지마다 annotation된 정보를 저장하려고 할때
    # if run_detections:
    #     if cfg.save_video:
    #         save_video_detections(det_exp_path)

    # owandb.finish()

if __name__ == "__main__":
    main()