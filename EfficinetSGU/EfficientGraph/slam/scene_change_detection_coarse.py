# import os, glob, re, shutil
# from pathlib import Path
# import numpy as np
# import cv2
# from skimage.metrics import structural_similarity as ssim

# # ---------------- utils ----------------
# def ensure_dir(p: str): os.makedirs(p, exist_ok=True)

# def downsample_rgbd(I, D, size):
#     I_ds = cv2.resize(I, size, interpolation=cv2.INTER_AREA)
#     D_ds = cv2.resize(D, size, interpolation=cv2.INTER_NEAREST)
#     return I_ds, D_ds

# def exposure_align_gray(g1, g2, valid=None):
#     if valid is None: valid = np.ones_like(g1, dtype=bool)
#     x = g2[valid].astype(np.float32).reshape(-1,1)
#     y = g1[valid].astype(np.float32).reshape(-1,1)
#     if x.size < 10:  # 안전장치
#         return g2, 1.0, 0.0
#     A = np.concatenate([x, np.ones_like(x)], axis=1)
#     a, b = np.linalg.lstsq(A, y, rcond=None)[0].reshape(-1)
#     g2p = np.clip(a*g2 + b, 0, 255).astype(g2.dtype)
#     return g2p, float(a), float(b)

# def chroma_delta_map(I1u8, I2u8, eps=1e-6):
#     f1 = I1u8.astype(np.float32)/255.0
#     f2 = I2u8.astype(np.float32)/255.0
#     s1 = f1.sum(axis=2, keepdims=True) + eps
#     s2 = f2.sum(axis=2, keepdims=True) + eps
#     r1g1 = f1[...,:2]/s1
#     r2g2 = f2[...,:2]/s2
#     return np.abs(r2g2 - r1g1).sum(axis=2)  # L1

# def get_tag(path: str, fallback_idx=None) -> str:
#     base = os.path.basename(path)
#     m = re.search(r'(\d{6,})', base)
#     if m: return f"{int(m.group(1))%1_000_000:06d}"
#     if fallback_idx is not None: return f"{int(fallback_idx):06d}"
#     raise ValueError(f"Cannot infer tag from: {path}")

# def load_pairs(rgbA, depA, rgbB, depB, rgb_ext="png", dep_ext="png"):
#     rA = sorted(glob.glob(os.path.join(rgbA, f"*.{rgb_ext}")))
#     rB = sorted(glob.glob(os.path.join(rgbB, f"*.{rgb_ext}")))
#     dA = sorted(glob.glob(os.path.join(depA, f"*.{dep_ext}")))
#     dB = sorted(glob.glob(os.path.join(depB, f"*.{dep_ext}")))
#     n = min(len(rA), len(rB), len(dA), len(dB))
#     for i in range(n):
#         yield rA[i], dA[i], rB[i], dB[i]

# def find_pose(rgbB_path, pose_dir, sorted_pose_list, idx_fallback=None):
#     cand = Path(pose_dir) / (Path(rgbB_path).stem + ".txt")
#     if cand.exists(): return str(cand)
#     if idx_fallback is not None and idx_fallback < len(sorted_pose_list):
#         return sorted_pose_list[idx_fallback]
#     return None

# # ---------------- SSIM (global + patches) ----------------
# def ssim_global_and_patches(g1, g2, patch=32, stride=32, win_size=7):
#     s_glob = float(ssim(g1, g2, win_size=win_size, gaussian_weights=True))
#     H, W = g1.shape
#     vals = []
#     for y in range(0, H - patch + 1, stride):
#         for x in range(0, W - patch + 1, stride):
#             w = min(win_size, patch-1)
#             s_loc = ssim(g1[y:y+patch, x:x+patch], g2[y:y+patch, x:x+patch],
#                          win_size=w, gaussian_weights=True)
#             vals.append(1.0 - float(s_loc))
#     mean_local = float(np.mean(vals)) if vals else (1.0 - s_glob)
#     return s_glob, mean_local

# # ---------------- Coarse score (Eq.1 & Eq.2) ----------------
# def coarse_change_score(Ia_rgb, Da_u16, Ib_rgb, Db_u16,
#                         wc=0.2, wd=0.5, ws=0.3, lam=0.3, Tc=0.01,
#                         alpha_m=0.003, beta=0.005,
#                         multiscale=((256,256),(128,128)),
#                         patch=32, stride=32):
#     S_list, cC_list, cD_list, cS_list = [], [], [], []
#     for size in multiscale:
#         Ia, Da = downsample_rgbd(Ia_rgb, Da_u16, size)
#         Ib, Db = downsample_rgbd(Ib_rgb, Db_u16, size)

#         Da_m = Da.astype(np.float32) / 1000.0
#         Db_m = Db.astype(np.float32) / 1000.0
#         valid = (Da_m > 0) & (Db_m > 0)

#         z = np.minimum(Da_m, Db_m)
#         tau = alpha_m + beta * z
#         depth_change = (np.abs(Db_m - Da_m) > tau) & valid
#         c_depth = float(depth_change.mean()) if valid.any() else 0.0

#         gA = cv2.cvtColor(Ia, cv2.COLOR_RGB2GRAY)
#         gB = cv2.cvtColor(Ib, cv2.COLOR_RGB2GRAY)
#         gB_adj, _, _ = exposure_align_gray(gA, gB, valid)
#         s_glob, mean_1m_ssim = ssim_global_and_patches(gA, gB_adj, patch=patch, stride=stride)
#         c_ssim = (1.0 - s_glob) + lam * mean_1m_ssim

#         c_color = float(chroma_delta_map(Ia, Ib)[valid].mean()) if valid.any() else \
#                   float(chroma_delta_map(Ia, Ib).mean())

#         S = wc*c_color + wd*c_depth + ws*c_ssim

#         S_list.append(S); cC_list.append(c_color); cD_list.append(c_depth); cS_list.append(c_ssim)

#     S = float(np.mean(S_list))
#     return {
#         "S": S,
#         "c_color": float(np.mean(cC_list)),
#         "c_depth": float(np.mean(cD_list)),
#         "c_ssim": float(np.mean(cS_list)),
#         "changed_before": bool(S > Tc),
#         "changed": bool(S > Tc)   # coarse 단계에서는 동일 트리거 사용
#     }

# # ---------------- main ----------------
# if __name__ == "__main__":
#     # ===== 입력: A(과거), B(변경) =====
#     A_rgb   = "/home/pchy0316/baseline_project/concept-graphs-ali-dev/ai2thor/data/FloorPlan217_change1/color"
#     A_depth = "/home/pchy0316/baseline_project/concept-graphs-ali-dev/ai2thor/data/FloorPlan217_change1/depth"
#     B_rgb   = "/home/pchy0316/baseline_project/concept-graphs-ali-dev/ai2thor/data/FloorPlan217_change2/color"
#     B_depth = "/home/pchy0316/baseline_project/concept-graphs-ali-dev/ai2thor/data/FloorPlan217_change2/depth"
#     B_pose  = "/home/pchy0316/baseline_project/concept-graphs-ali-dev/ai2thor/data/FloorPlan217_change2/pose"

#     # ===== 출력 루트 =====
#     OUT_ROOT = "/home/pchy0316/dataset/my_local_data/Replica/room0_temp"
#     PRE_DIR        = os.path.join(OUT_ROOT, "pre_results")          # A만
#     RESULT_DIR     = os.path.join(OUT_ROOT, "results")              # B만
#     BEFORE_TMP_DIR = os.path.join(OUT_ROOT, "before_results_temp")  # (있을 수도 있음)
#     BEFORE_DIR     = os.path.join(OUT_ROOT, "before_results")

#     for d in [PRE_DIR, RESULT_DIR, BEFORE_DIR]:
#         ensure_dir(d)

#     pose_list = sorted(glob.glob(os.path.join(B_pose, "*.txt")))

#     cnt_changed, cnt_before = 0, 0

#     for idx, (a_rgb_p, a_dep_p, b_rgb_p, b_dep_p) in enumerate(load_pairs(A_rgb, A_depth, B_rgb, B_depth)):
#         Ia = cv2.cvtColor(cv2.imread(a_rgb_p), cv2.COLOR_BGR2RGB)
#         Ib = cv2.cvtColor(cv2.imread(b_rgb_p), cv2.COLOR_BGR2RGB)
#         Da = cv2.imread(a_dep_p, cv2.IMREAD_UNCHANGED)
#         Db = cv2.imread(b_dep_p, cv2.IMREAD_UNCHANGED)
#         if Da is None or Db is None:
#             print(f"[WARN] depth read fail at {idx}")
#             continue

#         out = coarse_change_score(Ia, Da, Ib, Db)

#         print(f"[Frame {idx:04d}] {os.path.basename(a_rgb_p)} vs {os.path.basename(b_rgb_p)}")
#         print(f"   S={out['S']:.4f} | rd={out['c_depth']:.4f} | ssimc={out['c_ssim']:.4f} | changed={out['changed']}")
#         print(f"   S={out['S']:.4f} | changed_before={out['changed_before']}")

#         if out["changed"]:        cnt_changed += 1
#         if out["changed_before"]: cnt_before  += 1
#         print(f"이전에 활용했던거: {cnt_changed}, 이번에 활용한 거: {cnt_before}")

#         tag = get_tag(b_rgb_p, fallback_idx=idx)

#         # ---- A 저장 (pre_results) : A RGB/DEPTH만 ----
#         cv2.imwrite(os.path.join(PRE_DIR, f"frame{tag}.jpg"), cv2.cvtColor(Ia, cv2.COLOR_RGB2BGR))
#         Da_u16 = Da.astype(np.uint16) if Da.dtype != np.uint16 else Da
#         cv2.imwrite(os.path.join(PRE_DIR, f"depth{tag}.png"), Da_u16)

#         # ---- B 저장 (results) : B RGB/DEPTH/POSE만 ----
#         cv2.imwrite(os.path.join(RESULT_DIR, f"frame{tag}.jpg"), cv2.cvtColor(Ib, cv2.COLOR_RGB2BGR))
#         Db_u16 = Db.astype(np.uint16) if Db.dtype != np.uint16 else Db
#         cv2.imwrite(os.path.join(RESULT_DIR, f"depth{tag}.png"), Db_u16)

#         pose_src = find_pose(b_rgb_p, B_pose, pose_list, idx_fallback=idx)
#         if pose_src is None:
#             print(f"[WARN] pose not found for {os.path.basename(b_rgb_p)}")
#         else:
#             pose_dst = os.path.join(RESULT_DIR, f"pose{tag}.txt")
#             ensure_dir(os.path.dirname(pose_dst))
#             shutil.copy2(pose_src, pose_dst)

#         # ---- (옵션) 이전 detection 산출물 동기화 ----
#         src_folder = os.path.join(BEFORE_TMP_DIR, f"frame{tag}")
#         dst_folder = os.path.join(BEFORE_DIR,     f"frame{tag}")
#         if os.path.isdir(src_folder):
#             if os.path.exists(dst_folder):
#                 shutil.rmtree(dst_folder)
#             shutil.copytree(src_folder, dst_folder)

#     print("\n[Done]")

import os, glob, re, shutil, csv, json
from pathlib import Path
import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim

# ---------------- utils ----------------
def ensure_dir(p: str): os.makedirs(p, exist_ok=True)

def downsample_rgbd(I, D, size):
    I_ds = cv2.resize(I, size, interpolation=cv2.INTER_AREA)
    D_ds = cv2.resize(D, size, interpolation=cv2.INTER_NEAREST)
    return I_ds, D_ds

def exposure_align_gray(g1, g2, valid=None):
    if valid is None: valid = np.ones_like(g1, dtype=bool)
    x = g2[valid].astype(np.float32).reshape(-1,1)
    y = g1[valid].astype(np.float32).reshape(-1,1)
    if x.size < 10:
        return g2, 1.0, 0.0
    A = np.concatenate([x, np.ones_like(x)], axis=1)
    a, b = np.linalg.lstsq(A, y, rcond=None)[0].reshape(-1)
    g2p = np.clip(a*g2 + b, 0, 255).astype(g2.dtype)
    return g2p, float(a), float(b)

def chroma_delta_map(I1u8, I2u8, eps=1e-6):
    f1 = I1u8.astype(np.float32)/255.0
    f2 = I2u8.astype(np.float32)/255.0
    s1 = f1.sum(axis=2, keepdims=True) + eps
    s2 = f2.sum(axis=2, keepdims=True) + eps
    r1g1 = f1[...,:2]/s1
    r2g2 = f2[...,:2]/s2
    return np.abs(r2g2 - r1g1).sum(axis=2)

def get_tag(path: str, fallback_idx=None) -> str:
    base = os.path.basename(path)
    m = re.search(r'(\d{6,})', base)
    if m: return f"{int(m.group(1))%1_000_000:06d}"
    if fallback_idx is not None: return f"{int(fallback_idx):06d}"
    raise ValueError(f"Cannot infer tag from: {path}")

def load_pairs(rgbA, depA, rgbB, depB, rgb_ext="png", dep_ext="png"):
    rA = sorted(glob.glob(os.path.join(rgbA, f"*.{rgb_ext}")))
    rB = sorted(glob.glob(os.path.join(rgbB, f"*.{rgb_ext}")))
    dA = sorted(glob.glob(os.path.join(depA, f"*.{dep_ext}")))
    dB = sorted(glob.glob(os.path.join(depB, f"*.{dep_ext}")))
    n = min(len(rA), len(rB), len(dA), len(dB))
    for i in range(n):
        yield rA[i], dA[i], rB[i], dB[i]

def find_pose(rgbB_path, pose_dir, sorted_pose_list, idx_fallback=None):
    cand = Path(pose_dir) / (Path(rgbB_path).stem + ".txt")
    if cand.exists(): return str(cand)
    if idx_fallback is not None and idx_fallback < len(sorted_pose_list):
        return sorted_pose_list[idx_fallback]
    return None

# ---- before 산출물 복사(+필요시 리네이밍) -----------------
def sync_before_artifacts(orig_tag: str, new_tag: str, src_root: str, dst_root: str):
    """
    1) src_root/frame{orig_tag}/ 이 있으면 -> dst_root/frame{new_tag}/ 로 폴더째 복사
    2) 그 외 src_root 내 파일 중 orig_tag를 포함하는 파일은 이름의 tag를 new_tag 로 치환해 복사
    """
    ensure_dir(dst_root)

    src_folder = os.path.join(src_root, f"frame{orig_tag}")
    if os.path.isdir(src_folder):
        dst_folder = os.path.join(dst_root, f"frame{new_tag}")
        if os.path.exists(dst_folder):
            shutil.rmtree(dst_folder)
        shutil.copytree(src_folder, dst_folder)
        return True

    patterns = [f"*{orig_tag}.*", f"*{orig_tag}_*.*", f"*_{orig_tag}.*"]
    copied = False
    for pat in patterns:
        for src in glob.glob(os.path.join(src_root, pat)):
            dst_name = os.path.basename(src).replace(orig_tag, new_tag)
            shutil.copy2(src, os.path.join(dst_root, dst_name))
            copied = True
    return copied

# ---------------- SSIM (global + patches) ----------------
def ssim_global_and_patches(g1, g2, patch=32, stride=32, win_size=7):
    s_glob = float(ssim(g1, g2, win_size=win_size, gaussian_weights=True))
    H, W = g1.shape
    vals = []
    for y in range(0, H - patch + 1, stride):
        for x in range(0, W - patch + 1, stride):
            w = min(win_size, patch-1)
            s_loc = ssim(g1[y:y+patch, x:x+patch], g2[y:y+patch, x:x+patch],
                         win_size=w, gaussian_weights=True)
            vals.append(1.0 - float(s_loc))
    mean_local = float(np.mean(vals)) if vals else (1.0 - s_glob)
    return s_glob, mean_local

# ---------------- Coarse score (Eq.1 & Eq.2) ----------------
def coarse_change_score(Ia_rgb, Da_u16, Ib_rgb, Db_u16,
                        wc=0.03, wd=0.32, ws=0.65, lam=0.95, Tc=0.035,
                        alpha_m=0.003, beta=0.045,
                        multiscale=((256,256),(128,128)),
                        patch=32, stride=32):
    S_list, cC_list, cD_list, cS_list = [], [], [], []
    for size in multiscale:
        Ia, Da = downsample_rgbd(Ia_rgb, Da_u16, size)
        Ib, Db = downsample_rgbd(Ib_rgb, Db_u16, size)

        Da_m = Da.astype(np.float32) / 1000.0
        Db_m = Db.astype(np.float32) / 1000.0
        valid = (Da_m > 0) & (Db_m > 0)

        z = np.minimum(Da_m, Db_m)
        tau = alpha_m + beta * z
        depth_change = (np.abs(Db_m - Da_m) > tau) & valid
        c_depth = float(depth_change.mean()) if valid.any() else 0.0

        gA = cv2.cvtColor(Ia, cv2.COLOR_RGB2GRAY)
        gB = cv2.cvtColor(Ib, cv2.COLOR_RGB2GRAY)
        gB_adj, _, _ = exposure_align_gray(gA, gB, valid)
        s_glob, mean_1m_ssim = ssim_global_and_patches(gA, gB_adj, patch=patch, stride=stride)
        c_ssim = (1.0 - s_glob) + lam * mean_1m_ssim

        c_color = float(chroma_delta_map(Ia, Ib)[valid].mean()) if valid.any() else \
                  float(chroma_delta_map(Ia, Ib).mean())

        S = wc*c_color + wd*c_depth + ws*c_ssim

        S_list.append(S); cC_list.append(c_color); cD_list.append(c_depth); cS_list.append(c_ssim)

    S = float(np.mean(S_list))
    return {
        "S": S,
        "c_color": float(np.mean(cC_list)),
        "c_depth": float(np.mean(cD_list)),
        "c_ssim": float(np.mean(cS_list)),
        "changed_before": bool(S > Tc),
        "changed": bool(S > Tc)   # coarse 단계에서는 동일 트리거
    }

# ---------------- main ----------------
if __name__ == "__main__":
    # ===== 입력: A(과거), B(변경) =====
    A_rgb   = "/home/pchy0316/baseline_project/concept-graphs-ali-dev/ai2thor/data/FloorPlan217_change1/color"
    A_depth = "/home/pchy0316/baseline_project/concept-graphs-ali-dev/ai2thor/data/FloorPlan217_change1/depth"
    B_rgb   = "/home/pchy0316/baseline_project/concept-graphs-ali-dev/ai2thor/data/FloorPlan217_change2/color"
    B_depth = "/home/pchy0316/baseline_project/concept-graphs-ali-dev/ai2thor/data/FloorPlan217_change2/depth"
    B_pose  = "/home/pchy0316/baseline_project/concept-graphs-ali-dev/ai2thor/data/FloorPlan217_change2/pose"

    # ===== 출력 루트 =====
    OUT_ROOT = "/home/pchy0316/dataset/my_local_data/Replica/scene0"
    PRE_DIR        = os.path.join(OUT_ROOT, "pre_results")          # A만 (변화 프레임만, 재인덱싱)
    RESULT_DIR     = os.path.join(OUT_ROOT, "results")              # B만 (변화 프레임만, 재인덱싱)
    BEFORE_DIR     = os.path.join(OUT_ROOT, "before_results")       # 이전 detection 산출물
    BEFORE_SRC_DIR = os.path.join(OUT_ROOT, "before_results_temp")  # (네가 저장해둔 실제 위치로 바꿔도 됨)

    for d in [PRE_DIR, RESULT_DIR, BEFORE_DIR]:
        ensure_dir(d)

    # 저장 모드: 'changed' 또는 'changed_before'
    SAVE_KEY = "changed"          # True인 프레임만 저장
    # 매핑 기록 파일(원태그->신태그)도 남겨 둠
    MAP_CSV = os.path.join(OUT_ROOT, "coarse_changed_mapping.csv")
    with open(MAP_CSV, "w", newline="") as f:
        csv.writer(f).writerow(["orig_tag", "new_tag", "S", "c_color", "c_depth", "c_ssim"])

    pose_list = sorted(glob.glob(os.path.join(B_pose, "*.txt")))

    out_idx = 0  # 000000부터 재인덱싱
    num_total = 0
    num_saved = 0

    LOG_TXT = os.path.join(OUT_ROOT, "coarse_scores.txt")
    with open(LOG_TXT, "w") as log_f:   # 실행할 때마다 새로 작성
        log_f.write("frame_idx\tS\tc_color\tc_depth\tc_ssim\tchanged\n")

    for idx, (a_rgb_p, a_dep_p, b_rgb_p, b_dep_p) in enumerate(load_pairs(A_rgb, A_depth, B_rgb, B_depth)):
        num_total += 1
        Ia = cv2.cvtColor(cv2.imread(a_rgb_p), cv2.COLOR_BGR2RGB)
        Ib = cv2.cvtColor(cv2.imread(b_rgb_p), cv2.COLOR_BGR2RGB)
        Da = cv2.imread(a_dep_p, cv2.IMREAD_UNCHANGED)
        Db = cv2.imread(b_dep_p, cv2.IMREAD_UNCHANGED)
        if Da is None or Db is None:
            print(f"[WARN] depth read fail at {idx}")
            continue

        out = coarse_change_score(Ia, Da, Ib, Db)
        

        # --- 콘솔에 출력 ---
        msg = (f"[Frame {idx:06d}] "
            f"S={out['S']:.6f}, "
            f"c_color={out['c_color']:.6f}, "
            f"c_depth={out['c_depth']:.6f}, "
            f"c_ssim={out['c_ssim']:.6f}, "
            f"changed={out['changed']}")
        print(msg)

        # --- txt 파일에도 같은 내용 저장 ---
        with open(LOG_TXT, "a") as log_f:
            log_f.write(f"{idx:06d}\t{out['S']:.6f}\t{out['c_color']:.6f}\t"
                        f"{out['c_depth']:.6f}\t{out['c_ssim']:.6f}\t{int(out['changed'])}\n")




        orig_tag = get_tag(b_rgb_p, fallback_idx=idx)  # 원래 태그(로그/매핑용)
        print(f"[Frame {idx:04d}] {os.path.basename(a_rgb_p)} vs {os.path.basename(b_rgb_p)} "
              f"-> S={out['S']:.4f} rd={out['c_depth']:.4f} cssim={out['c_ssim']:.4f} {SAVE_KEY}={out[SAVE_KEY]}")

        if not out[SAVE_KEY]:
            continue  # <<<<<<<<<<<<<< 변한 프레임만 저장

        new_tag = f"{out_idx:06d}"   # 재인덱싱된 태그
        out_idx += 1
        num_saved += 1

        # ---- A 저장 (pre_results) : A RGB/DEPTH ----
        cv2.imwrite(os.path.join(PRE_DIR, f"frame{new_tag}.jpg"), cv2.cvtColor(Ia, cv2.COLOR_RGB2BGR))
        Da_u16 = Da.astype(np.uint16) if Da.dtype != np.uint16 else Da
        cv2.imwrite(os.path.join(PRE_DIR, f"depth{new_tag}.png"), Da_u16)

        # ---- B 저장 (results) : B RGB/DEPTH/POSE ----
        cv2.imwrite(os.path.join(RESULT_DIR, f"frame{new_tag}.jpg"), cv2.cvtColor(Ib, cv2.COLOR_RGB2BGR))
        Db_u16 = Db.astype(np.uint16) if Db.dtype != np.uint16 else Db
        cv2.imwrite(os.path.join(RESULT_DIR, f"depth{new_tag}.png"), Db_u16)

        pose_src = find_pose(b_rgb_p, B_pose, pose_list, idx_fallback=idx)
        if pose_src is None:
            print(f"[WARN] pose not found for {os.path.basename(b_rgb_p)}")
        else:
            pose_dst = os.path.join(RESULT_DIR, f"pose{new_tag}.txt")
            ensure_dir(os.path.dirname(pose_dst))
            shutil.copy2(pose_src, pose_dst)

        # ---- before 산출물 동기화 (orig_tag -> new_tag로 이동/리네임) ----
        if os.path.isdir(BEFORE_SRC_DIR):
            sync_before_artifacts(orig_tag, new_tag, BEFORE_SRC_DIR, BEFORE_DIR)

        # ---- 매핑 기록 ----
        with open(MAP_CSV, "a", newline="") as f:
            csv.writer(f).writerow([orig_tag, new_tag, out["S"], out["c_color"], out["c_depth"], out["c_ssim"]])

    print(f"\n[Done] total={num_total}, saved={num_saved}, out_index_max={out_idx-1:06d}")
