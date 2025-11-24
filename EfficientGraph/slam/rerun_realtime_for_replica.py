'''
The script is used to model Grounded SAM detections in 3D, it assumes the tag2text classes are avaialable. It also assumes the dataset has Clip features saved for each object/mask.
'''

# Standard library imports
import os
import copy
import uuid
from pathlib import Path
import pickle
import gzip

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
from EfficientGraph.utils.ious import mask_subtract_contained
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
    save_obj_json_temp, 
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
from EfficientGraph.slam.slam_classes import MapEdgeMapping, MapObjectList
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
    resize_gobs
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

# ---- Flexible rule-edge logger (schema-agnostic + optional floor handling) ----


SPECIAL_TAGS = {"floor", "ceiling", "wall"}  # 필요시 추가/수정

def _vec3(v):
    try:
        if hasattr(v, "tolist"):
            v = v.tolist()
        return [float(v[0]), float(v[1]), float(v[2])]
    except Exception:
        return [0.0, 0.0, 0.0]

def _object_anchor(objects, idx):
    """bbox.center > center > pcd.mean > (0,0,0)"""
    o = objects[idx]
    if "bbox" in o and o["bbox"] is not None:
        b = o["bbox"]
        if hasattr(b, "get_center"):
            return _vec3(np.asarray(b.get_center()))
        if isinstance(b, dict) and "center" in b:
            return _vec3(b["center"])
    if "center" in o:
        return _vec3(o["center"])
    if "pcd" in o and o["pcd"] is not None and hasattr(o["pcd"], "points"):
        try:
            pts = np.asarray(o["pcd"].points)
            if pts.size > 0:
                return _vec3(pts.mean(axis=0))
        except Exception:
            pass
    return [0.0, 0.0, 0.0]

def _idx_by_curr_num_map(objects):
    # curr_obj_num -> index 매핑
    m = {}
    for i, o in enumerate(objects):
        curr = o.get("curr_obj_num")
        if curr is not None:
            m[int(curr)] = i
    return m

def _parse_obj_ref(ref, n_objects, idx_by_curr):
    """
    다양한 ref를 0-base 인덱스로 변환:
    - int: [우선] 0-base 인덱스로 시도, [대안] curr_obj_num로 시도
    - str: 'object_3' -> 2, 숫자문자열 '3' -> curr_obj_num 3 → idx, 그 외는 None
    - dict: {'idx': i} 또는 {'curr': num} 또는 {'num': num}
    """
    # special은 여기서 처리하지 않음(호출부에서 별도 처리)
    if isinstance(ref, int):
        # 0-base 인덱스 시도
        if 0 <= ref < n_objects:
            return ref
        # curr_obj_num 시도
        if ref in idx_by_curr:
            return idx_by_curr[ref]
        # 1-base 인덱스일 가능성
        if 1 <= ref <= n_objects and (ref-1) not in idx_by_curr.values():
            return ref - 1
        return None

    if isinstance(ref, str):
        s = ref.strip()
        s_low = s.lower()
        # 숫자 추출
        digits = "".join(ch for ch in s if ch.isdigit())
        if "object_" in s_low and digits:
            i = int(digits) - 1
            return i if 0 <= i < n_objects else None
        if digits:
            # 숫자만이면 curr_obj_num으로 해석 우선
            num = int(digits)
            if num in idx_by_curr:
                return idx_by_curr[num]
            # 인덱스로도 시도
            if 0 <= num < n_objects:
                return num
            if 1 <= num <= n_objects:
                return num - 1
        return None

    if isinstance(ref, dict):
        if "idx" in ref and isinstance(ref["idx"], int):
            i = ref["idx"]
            if 0 <= i < n_objects:
                return i
        for k in ("curr", "num", "curr_obj_num"):
            if k in ref and isinstance(ref[k], int):
                num = int(ref[k])
                if num in idx_by_curr:
                    return idx_by_curr[num]
        # 기타 포맷은 여기서 해석 안 함
        return None

    return None

def _floor_value_from_objects(objects, axis='z', default=0.0):
    if not objects:
        return default
    ax = {'x':0,'y':1,'z':2}[axis]
    vals = []
    for i in range(len(objects)):
        a = _object_anchor(objects, i)
        vals.append(a[ax])
    return min(vals) if vals else default

def _project_to_plane(p, axis='z', plane_val=0.0):
    x, y, z = p
    if axis == 'z':
        return [x, y, plane_val]
    if axis == 'y':
        return [x, plane_val, z]
    if axis == 'x':
        return [plane_val, y, z]
    return [x, y, z]

def orr_log_rule_edges_from_json(
    orr, objects, obj_classes, json_path_or_data,
    base_entity_path="world/rule_edges",   # 원본 world/edges와 분리
    clear=True,                            # 이 트리만 초기화
    default_num_dets=2,                    # <=1 skip 우회
    handle_special="skip",                 # "skip" | "project" (floor 등 표시)
    floor_axis='z',                        # y-up이면 'y'
    floor_value=None                       # None이면 객체로 추정, 숫자면 고정
):
    """
    다양한 스키마를 허용:
    - 엣지 사전의 키 후보:
        subj / subject / s / head / source
        obj  / object  / o / tail / target
        rel  / relation / r / type / label
        num_dets / count / n
        endpoints: [[x,y,z],[x,y,z]] 있으면 그대로 사용
        subj_point / obj_point: 한쪽만 있으면 나머지는 앵커
    - special: 'floor','wall','ceiling' (handle_special="skip" or "project")
    """
    # 데이터 준비
    if isinstance(json_path_or_data, (str, Path)):
        p = Path(json_path_or_data)
        if not p.exists():
            print(f"[rule_edges] not found: {p}")
            return
        with open(p) as f:
            edges = json.load(f)
        src_name = p.name
    else:
        edges = json_path_or_data
        src_name = "<in-memory>"

    if clear:
        orr.log(base_entity_path, orr.Clear(recursive=True))

    n_obj = len(objects)
    idx_by_curr = _idx_by_curr_num_map(objects)
    floor_val = (_floor_value_from_objects(objects, axis=floor_axis, default=0.0)
                 if floor_value is None else float(floor_value))

    def _get(e, keys, default=None):
        for k in keys:
            if k in e:
                return e[k]
        return default

    added = 0
    for e in edges:
        # 1) raw fields
        subj_raw = _get(e, ("subj","subject","s","head","source","from"))
        obj_raw  = _get(e, ("obj","object","o","tail","target","to"))
        rel_type = str(_get(e, ("rel","relation","r","type","label"), "rel")).replace(" ", "_")
        num_dets = int(_get(e, ("num_dets","count","n"), default_num_dets))

        # 2) endpoints override
        endpoints = _get(e, ("endpoints","line","pts"))
        if endpoints and isinstance(endpoints, (list, tuple)) and len(endpoints) == 2:
            p1 = _vec3(endpoints[0])
            p2 = _vec3(endpoints[1])
            endpoints = [p1, p2]
            si = oi = None  # 라벨/색상 계산에만 객체 정보 필요
        else:
            # special?
            subj_is_special = isinstance(subj_raw, str) and subj_raw.lower() in SPECIAL_TAGS
            obj_is_special  = isinstance(obj_raw,  str) and obj_raw.lower()  in SPECIAL_TAGS

            # 3) 인덱스 해석
            si = None if subj_is_special else _parse_obj_ref(subj_raw, n_obj, idx_by_curr)
            oi = None if obj_is_special  else _parse_obj_ref(obj_raw,  n_obj, idx_by_curr)

            if handle_special == "skip" and (subj_is_special or obj_is_special):
                continue
            if (si is None and not subj_is_special) or (oi is None and not obj_is_special):
                continue
            if (not subj_is_special) and (not obj_is_special) and si == oi:
                continue

            # 4) 한쪽/양쪽 좌표 만들기 (subj_point/obj_point가 있으면 우선 사용)
            subj_pt = _get(e, ("subj_point","subject_point"))
            obj_pt  = _get(e, ("obj_point","object_point"))
            if subj_pt is not None:
                p_subj = _vec3(subj_pt)
            elif subj_is_special:
                # special은 plane 위 한 점(상대쪽 투영)으로 그려야 하니 일단 None → 나중 처리
                p_subj = None
            else:
                p_subj = _object_anchor(objects, si)

            if obj_pt is not None:
                p_obj = _vec3(obj_pt)
            elif obj_is_special:
                p_obj = None
            else:
                p_obj = _object_anchor(objects, oi)

            # special을 plane으로 투영해서 수직 라인 구성
            if handle_special == "project" and (subj_is_special or obj_is_special):
                if subj_is_special and p_obj is not None:
                    p_plane = _project_to_plane(p_obj, axis=floor_axis, plane_val=floor_val)
                    endpoints = [p_plane, p_obj]
                elif obj_is_special and p_subj is not None:
                    p_plane = _project_to_plane(p_subj, axis=floor_axis, plane_val=floor_val)
                    endpoints = [p_plane, p_subj]
                else:
                    # 한쪽도 앵커가 없으면 스킵
                    continue
            else:
                # 둘 다 object일 때 기본 라인
                endpoints = [p_subj, p_obj]

        # 라벨/색 준비 (네 스타일 그대로)
        if si is not None:
            obj1_label = f"{objects[si].get('curr_obj_num', (si+1))}"
            obj1_cls   = str(objects[si].get('class_name','Object')).replace(" ", "_")
        else:
            # special/좌표 기반만 있을 때 대체 라벨
            obj1_label = str(subj_raw).upper() if isinstance(subj_raw, str) else "S"
            obj1_cls   = obj1_label

        if oi is not None:
            obj2_label = f"{objects[oi].get('curr_obj_num', (oi+1))}"
            obj2_cls   = str(objects[oi].get('class_name','Object')).replace(" ", "_")
        else:
            obj2_label = str(obj_raw).upper() if isinstance(obj_raw, str) else "O"
            obj2_cls   = obj2_label

        edge_label_by_curr_num = f"{obj1_label}_{rel_type}_{obj2_label}"
        full_label = f"{obj1_label}_{obj1_cls}__{rel_type}__{obj2_label}_{obj2_cls}_({num_dets})"
        name_label = f"{obj1_cls}__{rel_type}__{obj2_cls}"

        # 색상(가능하면 obj 쪽 클래스 색)
        obj_2_color = None
        try:
            if oi is not None:
                obj_2_color = obj_classes.get_class_color(objects[oi]['class_name'])
        except Exception:
            obj_2_color = None

        # 네가 준 orr_log_edges 경로/라벨 5종 그대로
        orr.log(
            f"{base_entity_path}/edges_no_labels/{edge_label_by_curr_num}",
            orr.LineStrips3D(endpoints, colors=[obj_2_color] if obj_2_color is not None else None),
            orr.AnyValues(full_label=full_label)
        )
        orr.log(
            f"{base_entity_path}/edges_w_num_det_labels/{edge_label_by_curr_num}",
            orr.LineStrips3D(endpoints, labels=[f"{num_dets}"],
                             colors=[obj_2_color] if obj_2_color is not None else None),
            orr.AnyValues(full_label=full_label)
        )
        orr.log(
            f"{base_entity_path}/edges_w_rel_type_labels/{edge_label_by_curr_num}",
            orr.LineStrips3D(endpoints, labels=[rel_type],
                             colors=[obj_2_color] if obj_2_color is not None else None),
            orr.AnyValues(full_label=full_label)
        )
        orr.log(
            f"{base_entity_path}/edges_w_full_labels/{edge_label_by_curr_num}",
            orr.LineStrips3D(endpoints, labels=[full_label],
                             colors=[obj_2_color] if obj_2_color is not None else None),
            orr.AnyValues(full_label=full_label)
        )
        orr.log(
            f"{base_entity_path}/edges_w_names/{edge_label_by_curr_num}",
            orr.LineStrips3D(endpoints, labels=[name_label],
                             colors=[obj_2_color] if obj_2_color is not None else None),
            orr.AnyValues(full_label=full_label)
        )

        added += 1

    print(f"[rerun] logged {added} rule edges at '{base_entity_path}' from {src_name}")

#----------poly 저장하기-----------------------
def save_polys_compact(
    objects,          # MapObjectList (list-like of dicts)
    obj_infos,        # 위에서 만든 요약 리스트( idx, poly, lower_y, is_fixed 등 )
    out_npz_path,     # 예: exp_out_path / f"polys_{cfg.exp_suffix}.npz"
    quantize=True,    # True면 정수 양자화 저장
    scale=100.0       # 1m * 100 = 1cm 정밀도 (필요시 1000=1mm)
):
    """
    objects 순서와 동일하게 폴리곤(외곽선 XZ)을 이어붙여 저장.
    - quantize=True: int16로 저장, verts_q / scale 포함
    - quantize=False: float32로 저장, verts 포함
    항상 offsets(M+1), lower_y(M), has_poly(M) 포함
    """
    M = len(objects)
    offsets = [0]
    verts_list = []
    has_poly = np.zeros(M, dtype=np.uint8)
    lower_y = np.full(M, np.nan, dtype=np.float32)

    # obj_infos를 인덱스 키로 빠르게 조회
    infos_by_idx = {info["idx"]: info for info in obj_infos}

    for i in range(M):
        info = infos_by_idx.get(i, None)
        if info is not None:
            lower_y[i] = float(info.get("lower_y", np.nan))
            poly = info.get("poly", None)
        else:
            poly = None

        if poly is not None:
            # 외곽 좌표 (마지막=첫점 중복 제거)
            coords = np.array(poly.exterior.coords[:-1], dtype=np.float64)  # (N, 2) with (x, z)
            # 안전 장치: 유효 꼭짓점 3개 미만이면 무시
            if coords.shape[0] >= 3:
                has_poly[i] = 1
                # 이어붙이기
                verts_list.append(coords.astype(np.float32))  # float32로 먼저 모음
                offsets.append(offsets[-1] + coords.shape[0])
            else:
                offsets.append(offsets[-1])
        else:
            offsets.append(offsets[-1])

    if len(verts_list) > 0:
        verts = np.concatenate(verts_list, axis=0)  # (K, 2)
    else:
        verts = np.zeros((0, 2), dtype=np.float32)

    offsets = np.asarray(offsets, dtype=np.int32)

    # 저장
    out_npz_path.parent.mkdir(parents=True, exist_ok=True)
    if quantize:
        # 정수 양자화 (예: 1cm 단위): float * scale → round → int16
        verts_q = np.round(verts * scale).astype(np.int16)
        np.savez(
            out_npz_path,
            verts_q=verts_q,
            offsets=offsets,
            lower_y=lower_y,
            has_poly=has_poly,
            scale=np.array([scale], dtype=np.float32),
        )
    else:
        np.savez(
            out_npz_path,
            verts=verts.astype(np.float32),
            offsets=offsets,
            lower_y=lower_y,
            has_poly=has_poly,
        )

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

    objects = MapObjectList(device=cfg.device) 
    map_edges = MapEdgeMapping(objects)

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
        image_original_pil = Image.open(color_path)
        # color and depth tensors, and camera instrinsics matrix
        color_tensor, depth_tensor, intrinsics, *_ = dataset[frame_idx]

        # Covert to numpy and do some sanity checks
        depth_tensor = depth_tensor[..., 0]
        depth_array = depth_tensor.cpu().numpy()
        color_np = color_tensor.cpu().numpy() # (H, W, 3)
        image_rgb = (color_np).astype(np.uint8) # (H, W, 3)
        assert image_rgb.max() > 1, "Image is not in range [0, 255]"

        # Load image detections for the current frame
        raw_gobs = None
        gobs = None # stands for grounded observations
        detections_path = det_exp_pkl_path / (color_path.stem + ".pkl.gz")
        
        vis_save_path_for_vlm = get_vlm_annotated_image_path(det_exp_vis_path, color_path)
        vis_save_path_for_vlm_edges = get_vlm_annotated_image_path(det_exp_vis_path, color_path, w_edges=True)
        
        if run_detections:
            results = None
            # opencv can't read Path objects...
            image = cv2.imread(str(color_path)) # This will in BGR color space
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

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
            
            # [CALLER FIX 1] 감지 0개면 CLIP 스킵 (torch.cat([]) 예방)
            if getattr(curr_det, "xyxy", None) is None or len(curr_det.xyxy) == 0:
                image_crops, image_feats, text_feats = [], None, []
            else:
                try:
                    image_crops, image_feats, text_feats = compute_clip_features_batched(
                        image_rgb, curr_det, clip_model, clip_preprocess, clip_tokenizer,
                        obj_classes.get_classes_arr(), cfg.device
                    )
                    # [CALLER FIX 2] 내부 필터링으로 전처리 0건이면 안전 스킵
                    if image_feats is None:
                        image_crops, image_feats, text_feats = [], None, []
                except RuntimeError as e:
                    # [CALLER FIX 3] torch.cat([]) 류 에러만 잡고 스킵, 그 외는 재발생
                    if "expected a non-empty list of Tensors" in str(e):
                        image_crops, image_feats, text_feats = [], None, []
                    else:
                        raise
            
            #office0때문에 넣은거                

            # increment total object detections
            tracker.increment_total_detections(len(curr_det.xyxy))

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

        gobs = filtered_gobs

        if len(gobs['mask']) == 0: # no detections in this frame
            continue

        # this helps make sure things like pillows on couches are separate objects
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
            # continue 
        # 한프레임에 대한것...
        
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
    # LOOP OVER -----------------------------------------------------
            
    # ------------------------ on between objects ----------------------- #
    
    # 여기서 너가 위 loop를 통해서 얻은 objects 정보를 활용하여 객체간의 관계를 Rule base로 정하는 on_edges를 생성해줘(여기서 생성하는 map_edges는 무시에 rule base아닌 다른 방법이닌까)
    # ------------------------ on between objects ----------------------- #
    # 규칙 기반 on-edge 생성 (y-up, 바닥 y=0, footprint=XZ)
    # - 바닥 접촉(lower_y < floor_thresh) 객체는 floor에 on
    # - 나머지는 footprint overlap이 충분하고 더 낮은(lower_y) 지지체를 부모로 on
    # - 지지체가 여러 개면: (1) 고정물(is_fixed) 후보 우선, (2) 그중 y-거리(Δy)가 가장 작은 것
    # - 지지체가 없으면 floor로 on

    # ------------------------ Rule-based edges (on / in) ----------------------- #
    # 이 블록은 기존 "on between objects" 전부를 교체합니다.

    on_edges = []
    in_edges = []

    # ---- 안전 import ----
    try:
        import alphashape
        HAS_ALPHASHAPE = True
    except Exception:
        HAS_ALPHASHAPE = False
    from shapely.geometry import MultiPoint, MultiPolygon, Polygon
    # inside 판정용
    try:
        from scipy.spatial import Delaunay
        HAS_DELAUNAY = True
    except Exception:
        HAS_DELAUNAY = False
    import open3d as o3d


    # ------------------------ Rule-based edges (on / in) for Replica (z-up) ----------------------- #
    # Replica: z 가 높이축, 바닥 평면 z = -1.47
    on_edges = []
    in_edges = []

    # ---- 안전 import ----
    try:
        import alphashape
        HAS_ALPHASHAPE = True
    except Exception:
        HAS_ALPHASHAPE = False
    from shapely.geometry import MultiPoint, MultiPolygon, Polygon
    # inside 판정용
    try:
        from scipy.spatial import Delaunay
        HAS_DELAUNAY = True
    except Exception:
        HAS_DELAUNAY = False
    import open3d as o3d

    # ---- 좌표/바닥 설정 ----
    FLOOR_Z = -1.47            # Replica floor plane
    floor_thresh = 0.15        # (m) 바닥 접촉 임계 (z 기준; lower_z가 floor에 얼마나 가까운지)

    # ---- 기타 하이퍼파라미터 ----
    overlap_thresh = 0.60      # footprint 교집합 / 위(상부) 객체 footprint 면적
    jitter_std = 4e-3          # 수치 안정화용 지터(약 4mm)
    alphashape_alpha = 1.0     # concave 정도 (scene_graph_processor와 맞춤)
    inside_thresh = 0.95       # DOVSG 기본값과 동일: 자식 포인트 중 부모 내부 비율 임계

    # ---- 고정물/벽걸이/컨테이너 후보 ----
    fixed_candidates = {
        "bed", "bookshelf", "cabinet", "kitchen cabinet", "bathroom cabinet", "file cabinet",
        "couch", "sofa", "sofa chair", "fireplace",
        "counter", "kitchen counter", "bathroom counter", "desk", "dining table", "coffee table",
        "end table", "nightstand", "tv stand", "dresser", "wardrobe", "closet",
        "refrigerator", "washing machine", "dishwasher", "oven", "stove",
        "radiator", "fireplace", "piano", "column", "pillar", "stairs", "stair rail",
        "structure", "shower", "toilet", "bathtub"
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
    container_class_set = {c for c in container_candidates if c in experiment_classes}

    # ---- 유틸 ----
    def _obj_points(obj):
        """open3d PointCloud → (N,3) numpy"""
        pcd = obj.get("pcd", None)
        if pcd is None:
            return None
        if hasattr(pcd, "to_legacy"):
            pcd = pcd.to_legacy()
        pts = np.asarray(pcd.points)
        return pts if pts is not None and pts.size >= 3 else None

    def _lower_z(pts):
        """z-up: 높이축 = z → 객체 최저 z"""
        return float(np.min(pts[:, 2]))

    def _xy_footprint(pts, alpha=alphashape_alpha):
        """
        (x,y) 평면에서 footprint polygon 생성 (z-up에서 ground plane은 XY).
        alphashape 있으면 concave, 없으면 convex hull fallback.
        Polygon이 아니거나 면적 0이면 None.
        """
        if pts is None or pts.shape[0] < 3:
            return None
        xy = pts[:, [0, 1]]
        mask = np.isfinite(xy).all(axis=1)
        xy = xy[mask]
        if xy.shape[0] < 3:
            return None

        # 중복 제거 + 수치 안정화
        try:
            xy_unique = np.unique(xy, axis=0)
        except Exception:
            xy_unique = np.asarray(list({(float(a), float(b)) for a, b in xy}), dtype=float)
        if xy_unique.shape[0] < 3:
            return None
        xy_unique = xy_unique + np.random.normal(0, jitter_std, xy_unique.shape)

        mp = MultiPoint(xy_unique)
        if HAS_ALPHASHAPE and xy_unique.shape[0] >= 4:
            poly = alphashape.alphashape(xy_unique, alpha=alpha)
            if isinstance(poly, MultiPolygon):
                poly = max(poly.geoms, key=lambda p: p.area)
            if poly is None or poly.is_empty:
                return None
        else:
            poly = mp.convex_hull
            if isinstance(poly, MultiPolygon):
                poly = max(poly.geoms, key=lambda p: p.area)

        return poly if isinstance(poly, Polygon) and poly.area > 0 else None

    # 부모/자식 기록 헬퍼
    def record_parent(child_idx: int, rel: str, parent_kind: str, parent_idx: int | None = None,
                      score: float | None = None, meta: dict | None = None):
        meta = meta or {}
        ch = objects[child_idx]
        ch["parent"] = {
            "rel": rel,            # "on" | "in"
            "kind": parent_kind,   # "object" | "floor"
            "idx": parent_idx,     # object 인덱스 or None (floor)
            "score": score,        # inside ratio / overlap 등
            "meta": meta,          # {"overlap":..., "dz":...} 등
        }
        if parent_kind == "object" and parent_idx is not None and 0 <= parent_idx < len(objects):
            par = objects[parent_idx]
            children = par.get("children", [])
            children.append({"idx": child_idx, "rel": rel})
            par["children"] = children

    # ---- 객체 요약 정보 수집 (z-up) ----
    # note: 파이프라인 호환을 위해 'lower_y' 라는 키에 실제 lower_z 값을 저장
    obj_infos = []   # dict: idx, class_name, lower_y(=lower_z), poly_xy, on_floor(bool), is_fixed(bool)
    fixed_set = []
    lower_y_set = []  # 실제로는 z 최소값 리스트
    for idx, obj in enumerate(objects):
        pts = _obj_points(obj)
        if pts is None:
            continue
        lz = _lower_z(pts)
        poly = _xy_footprint(pts)
        if poly is None:
            continue

        cls_name = obj.get("class_name", "").lower()
        is_hanging = (cls_name in hanging_candidates)
        # 바닥 접촉: 최저 z 가 바닥 z 에서 floor_thresh 이내인 경우
        on_floor = ((lz - FLOOR_Z) <= floor_thresh) and (not is_hanging)
        # is_fixed: 바닥 접촉 + 고정물 클래스
        is_fixed = (on_floor and (cls_name in fixed_class_set))
        if is_fixed:
            fixed_set.append(idx)

        obj["is_fixed"] = bool(is_fixed)
        obj["lower_y"] = float(lz)           # ← lower_y 키에 z값을 저장(호환성)
        lower_y_set.append(float(lz))

        obj_infos.append({
            "idx": idx,
            "class_name": cls_name,
            "lower_y": lz,                   # 내부 계산도 이 키를 사용 (실제로는 z)
            "poly": poly,                    # XY footprint
            "on_floor": on_floor,
            "is_fixed": is_fixed,
        })

    # poly 정보 저장(메모리/속도 최적화 포맷) — 좌표 성분만 바뀌었을 뿐 사용법 동일
    # poly_npz_path = exp_out_path / f"polys_{cfg.exp_suffix}.npz"
    # save_polys_compact(objects, obj_infos, poly_npz_path, quantize=True, scale=100.0)

    # ---- 바닥 접촉 객체 → floor on-edge ----
    for info in obj_infos:
        if info["on_floor"]:
            subj_idx = info["idx"]
            on_edges.append({
                "subj": f"object_{subj_idx+1}",
                "rel": "on",
                "obj": "floor"
            })
            record_parent(
                child_idx=subj_idx, rel="on", parent_kind="floor",
                parent_idx=None, score=None, meta={"reason": "on_floor"}
            )

    # ---- 비-바닥 객체 → 지지체 탐색(on) (z-up) ----
    # 우선순위: 고정물 지지체 > 비고정 지지체; 각 그룹 내에서는 Δz(= upper.lower_z - base.lower_z) 작은 순
    for info_top in obj_infos:
        if info_top["on_floor"]:
            continue
        if info_top["class_name"] in hanging_candidates:
            continue  # 벽/천장 걸이류 제외

        poly_top = info_top["poly"]
        area_top = poly_top.area if poly_top else 0.0
        if area_top <= 0:
            top_idx = info_top["idx"]
            on_edges.append({"subj": f"object_{top_idx+1}", "rel": "on", "obj": "floor"})
            record_parent(child_idx=top_idx, rel="on", parent_kind="floor")
            continue

        candidates_fixed = []   # (dz, inter_rate, info_base)
        candidates_others = []  # (dz, inter_rate, info_base)

        for info_base in obj_infos:
            if info_base["idx"] == info_top["idx"]:
                continue
            if info_base["class_name"] in hanging_candidates:
                continue
            # 더 낮은(lower_z) 지지체만
            if info_base["lower_y"] >= info_top["lower_y"]:
                continue

            poly_base = info_base["poly"]
            if poly_base is None:
                continue
            if not poly_top.intersects(poly_base):
                continue

            inter_area = poly_top.intersection(poly_base).area
            inter_rate = (inter_area / area_top) if area_top > 0 else 0.0
            if inter_rate < overlap_thresh:
                continue

            dz = info_top["lower_y"] - info_base["lower_y"]  # 양수: 위-아래 수직거리(z)
            item = (dz, inter_rate, info_base)

            if info_base["is_fixed"]:
                candidates_fixed.append(item)
            else:
                candidates_others.append(item)

        chosen_parent = None
        chosen_meta = None
        if candidates_fixed:
            candidates_fixed.sort(key=lambda x: x[0])  # dz 오름차순
            dz, inter_rate, base = candidates_fixed[0]
            chosen_parent = base
            chosen_meta = {"overlap": float(inter_rate), "dz": float(dz), "priority": "fixed"}
        elif candidates_others:
            candidates_others.sort(key=lambda x: x[0])
            dz, inter_rate, base = candidates_others[0]
            chosen_parent = base
            chosen_meta = {"overlap": float(inter_rate), "dz": float(dz), "priority": "non_fixed"}

        top_idx = info_top["idx"]
        if chosen_parent is not None:
            par_idx = chosen_parent["idx"]
            on_edges.append({"subj": f"object_{top_idx+1}", "rel": "on", "obj": f"object_{par_idx+1}"})
            record_parent(
                child_idx=top_idx, rel="on", parent_kind="object", parent_idx=par_idx,
                score=chosen_meta.get("overlap", None), meta=chosen_meta
            )
        else:
            on_edges.append({"subj": f"object_{top_idx+1}", "rel": "on", "obj": "floor"})
            record_parent(child_idx=top_idx, rel="on", parent_kind="floor")

    # ---- inside(in) 관계 (Delaunay/AABB) ----
    def _get_pts(idx):
        return _obj_points(objects[idx])

    def _inside_ratio_delaunay(parent_pts, child_pts):
        if parent_pts is None or child_pts is None: return None
        if parent_pts.shape[0] < 4 or child_pts.shape[0] == 0: return None
        try:
            tri = Delaunay(parent_pts)
            mask = tri.find_simplex(child_pts) > 0
            return float(np.count_nonzero(mask)) / float(child_pts.shape[0])
        except Exception:
            return None

    def _inside_ratio_bbox(parent_bbox, child_pts):
        if parent_bbox is None or child_pts is None or child_pts.shape[0] == 0: return None
        try:
            aabb = parent_bbox.get_axis_aligned_bounding_box()
            idxs = aabb.get_point_indices_within_bounding_box(o3d.utility.Vector3dVector(child_pts))
            return float(len(idxs)) / float(child_pts.shape[0])
        except Exception:
            return None

    pts_cache, bbox_cache = {}, {}

    for info_child in obj_infos:
        child_idx = info_child["idx"]
        if info_child["class_name"] in hanging_candidates:
            continue

        child_pts = pts_cache.get(child_idx)
        if child_pts is None:
            child_pts = _get_pts(child_idx); pts_cache[child_idx] = child_pts
        if child_pts is None or child_pts.shape[0] == 0:
            continue

        best_parent, best_score = None, -1.0

        for info_parent in obj_infos:
            parent_idx = info_parent["idx"]
            if parent_idx == child_idx: continue
            if info_parent["class_name"] not in container_class_set: continue

            parent_pts = pts_cache.get(parent_idx)
            if parent_pts is None:
                parent_pts = _get_pts(parent_idx); pts_cache[parent_idx] = parent_pts

            parent_bbox = bbox_cache.get(parent_idx)
            if parent_bbox is None:
                parent_bbox = objects[parent_idx].get("bbox", None); bbox_cache[parent_idx] = parent_bbox

            score = None
            if HAS_DELAUNAY:
                score = _inside_ratio_delaunay(parent_pts, child_pts)
            if score is None and parent_bbox is not None:
                score = _inside_ratio_bbox(parent_bbox, child_pts)
            if score is None:
                continue

            if score > best_score:
                best_score, best_parent = score, parent_idx

        if best_parent is not None and best_score >= inside_thresh:
            subj_tag = f"object_{child_idx+1}"
            obj_tag  = f"object_{best_parent+1}"

            on_edges = [e for e in on_edges if not (e["rel"] == "on" and e["subj"] == subj_tag)]
            in_edges.append({"subj": subj_tag, "rel": "in", "obj": obj_tag})
            record_parent(
                child_idx=child_idx, rel="in", parent_kind="object",
                parent_idx=best_parent, score=float(best_score),
                meta={"inside_ratio": float(best_score)}
            )

    # ---- Anchor 지정 (동적 노드 → 최종 고정물) ----
    def _find_anchor_idx(node_idx: int) -> int | None:
        visited = set(); cur = node_idx
        while True:
            if cur in visited: return None
            visited.add(cur)
            par = objects[cur].get("parent", None)
            if par is None: return None
            if par["kind"] == "floor" or par["idx"] is None: return None
            pidx = int(par["idx"])
            if pidx < 0 or pidx >= len(objects): return None
            if objects[pidx].get("is_fixed", False): return pidx
            cur = pidx

    for info in obj_infos:
        i = info["idx"]
        if not objects[i].get("is_fixed", False):
            aidx = _find_anchor_idx(i)
            objects[i]["anchor"] = None if aidx is None else {
                "idx": aidx,
                "name": objects[aidx].get("class_name", f"object_{aidx+1}"),
                "tag": f"object_{aidx+1}",
            }

    # ---- 결과 저장 ----
    def _save_edges(name: str, edges_list: list):
        out_path = Path(exp_out_path) / f"{name}_{cfg.exp_suffix}.json"
        out_path.parent.mkdir(exist_ok=True, parents=True)
        with open(out_path, "w") as f:
            json.dump(edges_list, f, indent=2)
        print(f"[rulebase] Saved {name} → {out_path}")

    all_rule_edges = on_edges + in_edges
    _save_edges("on_edges_rule", on_edges)
    _save_edges("in_edges_rule", in_edges)
    _save_edges("rule_edges_all", all_rule_edges)
    

    # ---------------------------------------------------------------------------------------------- #

    if cfg.use_rerun:

        json_path = Path(exp_out_path) / f"rule_edges_all_{cfg.exp_suffix}.json"
        orr_log_rule_edges_from_json(
            orr, objects, obj_classes, json_path,
            base_entity_path="world/rule_edges",  # 기존 world/edges와 분리
            clear=True,                           # 이 트리만 초기화
            default_num_dets=2,                   # <=1 skip 우회
            handle_special="skip",                # floor 등 임의 태그는 기본 스킵
            floor_axis="z",                       # 필요시 'y'로
            floor_value=None
        )

    # -------------------------------------------------------------------------- #

    #fixed_object용 저장하기
    def save_fixed_list(exp_suffix, exp_out_path, fixed_set):
        out = Path(exp_out_path) / f"fixed_list_{exp_suffix}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump([int(x) for x in fixed_set], f)
        print(f"saved → {out}")

    def save_lower_list(exp_suffix, exp_out_path, lower_y_set):
        out = Path(exp_out_path) / f"lower_list_{exp_suffix}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump([float(x) for x in lower_y_set], f)
        print(f"saved → {out}")



    #fixed_object용 불러오기 불러오는 코드 --> fixed_set = load_fixed_list(cfg.exp_suffix, exp_out_path)
    def load_fixed_list(exp_suffix, exp_out_path):
        p = Path(exp_out_path) / f"fixed_list_{exp_suffix}.json"
        with open(p) as f:
            return [int(x) for x in json.load(f)]


    save_fixed_list(cfg.exp_suffix, exp_out_path, fixed_set)
    save_lower_list(cfg.exp_suffix, exp_out_path, lower_y_set)
    # 저장 (프로젝트 유틸 호출 시그니처 그대로 사용)
    # save_on_edges_simple(exp_suffix="test1", exp_out_path=exp_out_path, edges=on_edges)
    # save_clip_feats_npy(exp_suffix = "inital", out_path = exp_out_path, objects=objects, key="clip_ft")

    # save_text_feats_npy(exp_suffix = "inital", out_path = exp_out_path, objects=objects, key="text_ft")


    # Consolidate captions 
    for object in objects:
        obj_captions = object['captions'][:20]
        # consolidated_caption = consolidate_captions(openai_client, obj_captions)
        consolidated_caption = []
        object['consolidated_caption'] = consolidated_caption


    # Save the pointcloud
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

    if cfg.save_json:
        # save_obj_json(
        #     exp_suffix=cfg.exp_suffix,
        #     exp_out_path=exp_out_path,
        #     objects=objects
        # )
        
        save_obj_json_temp(
            exp_suffix=cfg.exp_suffix,
            exp_out_path=exp_out_path,
            objects=objects
        )
        
        
        # save_edge_json(
        #     exp_suffix=cfg.exp_suffix,
        #     exp_out_path=exp_out_path,
        #     objects=objects,
        #     edges=map_edges
        # )

    # Save metadata if all frames are saved
    if cfg.save_objects_all_frames:
        save_meta_path = obj_all_frames_out_path / f"meta.pkl.gz"
        with gzip.open(save_meta_path, "wb") as f:
            pickle.dump({
                'cfg': cfg,
                'class_names': obj_classes.get_classes_arr(),
                'class_colors': obj_classes.get_class_color_dict_by_index(),
            }, f)

    #마지막에 각 이미지마다 annotation된 정보를 저장하려고 할때
    if run_detections:
        if cfg.save_video:
            save_video_detections(det_exp_path)

    owandb.finish()

if __name__ == "__main__":
    main()
