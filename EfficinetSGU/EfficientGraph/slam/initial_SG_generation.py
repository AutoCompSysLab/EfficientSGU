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

def _log_rule_edges_json(orr, objects, obj_classes, exp_out_path, exp_suffix, name="rule_edges_all"):
    """
    {exp_out_path}/{name}_{exp_suffix}.json 파일을 읽어서
    [{'subj':'object_i','rel':'on|in|under','obj':'object_j'}, ...] 형태를
    임시 MapEdgeMapping으로 변환 후 rerun에 로깅한다.
    """
    p = Path(exp_out_path) / f"{name}_{exp_suffix}.json"
    if not p.exists():
        print(f"[rulebase] {p} not found — skip")
        return

    with open(p) as f:
        edges = json.load(f)

    tmp_edges = MapEdgeMapping(objects)  # 빈 edge 컨테이너
    added = 0
    for e in edges:
        subj, obj = e.get("subj"), e.get("obj")
        if not (isinstance(subj, str) and isinstance(obj, str)):
            continue
        try:
            si = int(subj.split("_")[-1]) - 1
            oi = int(obj.split("_")[-1]) - 1
        except Exception:
            continue
        if 0 <= si < len(objects) and 0 <= oi < len(objects):
            # 보유 API에 맞춰 add/update (프로젝트마다 이름이 조금 다를 수 있어요)
            if hasattr(tmp_edges, "add_edge"):
                try:
                    # 가장 흔한 시그니처 가정
                    tmp_edges.add_edge(si, oi, relation=e.get("rel", "on"))
                except TypeError:
                    # relation 인자 없는 구현일 경우
                    tmp_edges.add_edge(si, oi)
                    # get_edge / edges_by_index 가 있으면 레이블 주석
                    if hasattr(tmp_edges, "get_edge"):
                        edge = tmp_edges.get_edge(si, oi)
                        if edge is not None:
                            setattr(edge, "relation", e.get("rel", "on"))
            elif hasattr(tmp_edges, "add_or_update_edge"):
                tmp_edges.add_or_update_edge(si, oi)
            added += 1

    print(f"[rerun] logging {added} rule edges from {p.name}")
    # 기본 edge 로거 재활용
    orr_log_edges(objects, tmp_edges, obj_classes)  # 기존 호출과 동일하게 사용

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


    # ---- 하이퍼파라미터 ----
    floor_thresh = 0.15          # (m) 바닥 접촉 임계 (y 기준)
    overlap_thresh = 0.60        # footprint 교집합 / 위(상부) 객체 footprint 면적
    jitter_std = 4e-3            # 수치 안정화용 지터(약 4mm)
    alphashape_alpha = 1.0       # concave 정도 (scene_graph_processer와 맞춤)
    inside_thresh = 0.95         # DOVSG 기본값과 동일: 자식 포인트 중 부모 내부 비율 임계

    # ---- 고정물(fixed) 후보 클래스 (소문자 비교; 실험 클래스와 교집합만 사용) ----
    fixed_candidates = {
        "bed", "bookshelf", "cabinet", "kitchen cabinet", "bathroom cabinet", "file cabinet", "couch", "sofa", "sofa chair", "fireplace"
        "counter", "kitchen counter", "bathroom counter", "desk", "dining table", "coffee table",
        "end table", "nightstand", "tv stand", "dresser", "wardrobe", "closet",
        "refrigerator", "washing machine", "dishwasher", "oven", "stove",
        "radiator", "fireplace", "piano", "column", "pillar", "stairs", "stair rail",
        "structure", "shower", "toilet", "bathtub"
    }
    hanging_candidates = {"blackboard", "whiteboard", "bulletin board", "calendar", "clock", "mirror", "picture", "poster", "curtain", "fan",
                          "projector screen", "projector", "sign", "rack", "coat rack", "soap dispenser", "paper towel roll", "door",
                          "toilet paper dispenser", "toilet paper holder", "handicap bar", "fire alarm", "fire extinguisher", "window", 
                          "power outlet", "light switch", "radiator", "mailbox", "rail", "stair rail", "shower head", "ceiling light",
                          "projector", "vent", "range hood"}
    
    # inside 부모 후보(컨테이너 계열; 실험 클래스와 교집합만 사용)
    container_candidates = {
        # 수납/가구
        "cabinet", "kitchen cabinet", "bathroom cabinet", "closet", "wardrobe",
        "drawer", "dresser",
        # 용기/가전 내공간
        "box", "bin", "basket", "trash can", "bucket",
        #"fridge", "refrigerator", "microwave", "oven", "dishwasher", "washing machine",
        # 주방/그릇류
        "bowl", "cup", "mug", "pot", "pan", "jar",
        # 위생
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
        mask = np.isfinite(xz).all(axis=1)  # NaN/Inf 제거
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

    # 부모/자식 기록 헬퍼
    def record_parent(child_idx: int, rel: str, parent_kind: str, parent_idx: int | None = None,
                      score: float | None = None, meta: dict | None = None):
        """
        parent_kind: "object" | "floor"
        rel: "on" | "under"
        """
        meta = meta or {}
        ch = objects[child_idx]
        ch["parent"] = {
            "rel": rel,                   # "on" | "under"
            "kind": parent_kind,          # "object" | "floor"
            "idx": parent_idx,            # object 인덱스 or None (floor)
            "score": score,               # inside ratio / overlap 등
            "meta": meta,                 # {"overlap":..., "dy":...} 등
        }
        # 역참조(부모 → 자식)
        if parent_kind == "object" and parent_idx is not None and 0 <= parent_idx < len(objects):
            par = objects[parent_idx]
            children = par.get("children", [])
            children.append({"idx": child_idx, "rel": rel})
            par["children"] = children

    # ---- 객체 요약 정보 수집 ----
    # idx: objects 리스트 인덱스(0-base). 엣지 JSON에선 object_{idx+1}로 사용.
    obj_infos = []  # dict: idx, class_name, lower_y, poly_xz, on_floor(bool), is_fixed(bool)
    fixed_set = []
    lower_y_set = []
    for idx, obj in enumerate(objects):
        pts = _obj_points(obj)
        if pts is None:
            continue
        ly = _lower_y(pts)
        poly = _xz_footprint(pts)
        if poly is None:
            continue

        cls_name = obj.get("class_name", "").lower()
        is_hanging = (cls_name in hanging_candidates)
        on_floor = (ly < floor_thresh) and (not is_hanging)
        # is_fixed: 바닥 접촉 + 고정물 클래스
        is_fixed = (on_floor and (cls_name in fixed_class_set))
        if is_fixed: fixed_set.append(idx)
        
        obj["is_fixed"] = bool(is_fixed)   # objects에 is_fixed 추가(정적 노드 플래그)
        obj["lower_y"] = float(ly)
        lower_y_set.append(float(ly))
        # print("lower길이입니다########",float(ly))
        
        obj_infos.append({
            "idx": idx,
            "class_name": cls_name,
            "lower_y": ly,
            "poly": poly,
            "on_floor": on_floor,
            "is_fixed": is_fixed,
        })

    # poly 정보 저장(메모리/속도 최적화 포맷)
    poly_npz_path = exp_out_path / f"polys_{cfg.exp_suffix}.npz"
    save_polys_compact(objects, obj_infos, poly_npz_path, quantize=True, scale=100.0)

    # ---- 바닥 접촉 객체 → floor on-edge (+부모 기록) ----
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

    # ---- 비-바닥 객체 → 지지체 탐색(on) ----
    # 후보 우선순위: 고정물 지지체 > 비고정 지지체; 각 그룹 내에서는 Δy(= upper.lower_y - base.lower_y) 작은 순
    for info_top in obj_infos:
        if info_top["on_floor"]:
            continue  # 이미 floor 처리됨
        
        if info_top["class_name"] in hanging_candidates:
            continue  # 벽/천장 걸이류는 제외

        poly_top = info_top["poly"]
        area_top = poly_top.area if poly_top else 0.0
        if area_top <= 0:
            # footprint가 없으면 안전하게 floor로
            top_idx = info_top["idx"]
            on_edges.append({
                "subj": f"object_{top_idx+1}",
                "rel": "on",
                "obj": "floor"
            })
            record_parent(child_idx=top_idx, rel="on", parent_kind="floor")
            continue

        candidates_fixed = []   # (dy, inter_rate, info_base)
        candidates_others = []  # (dy, inter_rate, info_base)

        for info_base in obj_infos:
            if info_base["idx"] == info_top["idx"]:
                continue
            
            # 오탈자 수정: is_haning -> is_hanging
            if info_base["class_name"] in hanging_candidates:
                continue

            # 더 낮은(lower_y) 지지체만
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

            dy = info_top["lower_y"] - info_base["lower_y"]  # 양수: 위-아래 수직거리(y)
            item = (dy, inter_rate, info_base)  # 정렬 키: 수직거리

            if info_base["is_fixed"]:
                candidates_fixed.append(item)
            else:
                candidates_others.append(item)

        chosen_parent = None
        chosen_meta = None
        if candidates_fixed:
            candidates_fixed.sort(key=lambda x: x[0])  # dy 오름차순
            dy, inter_rate, base = candidates_fixed[0]
            chosen_parent = base
            chosen_meta = {"overlap": float(inter_rate), "dy": float(dy), "priority": "fixed"}
        elif candidates_others:
            candidates_others.sort(key=lambda x: x[0])
            dy, inter_rate, base = candidates_others[0]
            chosen_parent = base
            chosen_meta = {"overlap": float(inter_rate), "dy": float(dy), "priority": "non_fixed"}

        top_idx = info_top["idx"]
        if chosen_parent is not None:
            par_idx = chosen_parent["idx"]
            on_edges.append({
                "subj": f"object_{top_idx+1}",
                "rel": "on",
                "obj": f"object_{par_idx+1}"
            })
            record_parent(
                child_idx=top_idx, rel="on", parent_kind="object", parent_idx=par_idx,
                score=chosen_meta.get("overlap", None), meta=chosen_meta
            )
        else:
            # 지지체가 없으면 floor fallback
            on_edges.append({
                "subj": f"object_{top_idx+1}",
                "rel": "on",
                "obj": "floor"
            })
            record_parent(child_idx=top_idx, rel="on", parent_kind="floor")

    # ---- inside(in) 관계 (DOVSG 스타일) ----
    # 부모 점군으로 Delaunay(3D) → 자식 점군 내부 비율 >= inside_thresh이면 "under"으로 대체/추가
    def _get_pts(idx):
        return _obj_points(objects[idx])

    def _inside_ratio_delaunay(parent_pts, child_pts):
        if parent_pts is None or child_pts is None:
            return None
        if parent_pts.shape[0] < 4 or child_pts.shape[0] == 0:
            return None
        try:
            tri = Delaunay(parent_pts)
            mask = tri.find_simplex(child_pts) > 0
            return float(np.count_nonzero(mask)) / float(child_pts.shape[0])
        except Exception:
            return None  # qhull 실패 등

    def _inside_ratio_bbox(parent_bbox, child_pts):
        if parent_bbox is None or child_pts is None or child_pts.shape[0] == 0:
            return None
        try:
            aabb = parent_bbox.get_axis_aligned_bounding_box()
            idxs = aabb.get_point_indices_within_bounding_box(o3d.utility.Vector3dVector(child_pts))
            return float(len(idxs)) / float(child_pts.shape[0])
        except Exception:
            return None

    # 캐시
    pts_cache = {}
    bbox_cache = {}

    for info_child in obj_infos:
        child_idx = info_child["idx"]
        if info_child["class_name"] in hanging_candidates:
            continue  # 벽걸이류는 제외

        child_pts = pts_cache.get(child_idx)
        if child_pts is None:
            child_pts = _get_pts(child_idx)
            pts_cache[child_idx] = child_pts
        if child_pts is None or child_pts.shape[0] == 0:
            continue

        best_parent = None
        best_score = -1.0

        for info_parent in obj_infos:
            parent_idx = info_parent["idx"]
            if parent_idx == child_idx:
                continue
            # inside 부모 후보: 컨테이너 계열만
            if info_parent["class_name"] not in container_class_set:
                continue

            parent_pts = pts_cache.get(parent_idx)
            if parent_pts is None:
                parent_pts = _get_pts(parent_idx)
                pts_cache[parent_idx] = parent_pts

            parent_bbox = bbox_cache.get(parent_idx)
            if parent_bbox is None:
                parent_bbox = objects[parent_idx].get("bbox", None)
                bbox_cache[parent_idx] = parent_bbox

            score = None
            if HAS_DELAUNAY:
                score = _inside_ratio_delaunay(parent_pts, child_pts)
            if score is None and parent_bbox is not None:
                score = _inside_ratio_bbox(parent_bbox, child_pts)
            if score is None:
                continue

            if score > best_score:
                best_score = score
                best_parent = parent_idx

        if best_parent is not None and best_score >= inside_thresh:
            subj_tag = f"object_{child_idx+1}"
            obj_tag  = f"object_{best_parent+1}"

            # 같은 주어의 on-edge 제거(in이 우선)
            on_edges = [e for e in on_edges if not (e["rel"] == "on" and e["subj"] == subj_tag)]

            # in-edge 추가
            in_edges.append({"subj": subj_tag, "rel": "under", "obj": obj_tag})

            # 노드 부모 갱신(덮어쓰기)
            record_parent(
                child_idx=child_idx, rel="under", parent_kind="object",
                parent_idx=best_parent, score=float(best_score),
                meta={"inside_ratio": float(best_score)}
            )

    # ---- Anchor 지정(동적 노드 → 최종 고정물 소속) ----
    # 정의: is_fixed=False 인 노드는 부모 체인을 따라가 첫 is_fixed=True 노드를 anchor로 저장.
    def _find_anchor_idx(node_idx: int) -> int | None:
        visited = set()
        cur = node_idx
        while True:
            if cur in visited:
                return None  # cycle 방지
            visited.add(cur)
            par = objects[cur].get("parent", None)
            if par is None:
                return None
            if par["kind"] == "floor" or par["idx"] is None:
                return None
            pidx = int(par["idx"])
            if pidx < 0 or pidx >= len(objects):
                return None
            if objects[pidx].get("is_fixed", False):
                return pidx
            cur = pidx

    for info in obj_infos:
        i = info["idx"]
        if not objects[i].get("is_fixed", False):
            aidx = _find_anchor_idx(i)
            if aidx is not None:
                objects[i]["anchor"] = {
                    "idx": aidx,
                    "name": objects[aidx].get("class_name", f"object_{aidx+1}"),
                    "tag": f"object_{aidx+1}",
                }
            else:
                # 명시적으로 없는 경우 None으로 표기(원하면 생략 가능)
                objects[i]["anchor"] = None
                


    # ---- 결과 저장(선택) ----
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
    # -------------------------------------------------------------------------- #

    # >>> 여기 추가 <<<
    if cfg.use_rerun:
        _log_rule_edges_json(orr, objects, obj_classes, exp_out_path, cfg.exp_suffix, "rule_edges_all")
        
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
    save_on_edges_simple(exp_suffix="test1", exp_out_path=exp_out_path, edges=on_edges)
    # save_clip_feats_npy(exp_suffix = "inital", out_path = exp_out_path, objects=objects, key="clip_ft")

    # save_text_feats_npy(exp_suffix = "inital", out_path = exp_out_path, objects=objects, key="text_ft")


    # # Consolidate captions 
    # for object in objects:
    #     obj_captions = object['captions'][:20]
    #     # consolidated_caption = consolidate_captions(openai_client, obj_captions)
    #     consolidated_caption = []
    #     object['consolidated_caption'] = consolidated_caption


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

    owandb.finish()

if __name__ == "__main__":
    main()
