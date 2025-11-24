import numpy as np
import os, glob, cv2, re
from pathlib import Path
from skimage.metrics import structural_similarity as ssim
import shutil


def get_tag_from_path(path: str, default_idx: int = None) -> str:
    """
    파일/경로에서 6자리 프레임 번호를 추출.
    - 'frame000123.jpg', 'depth000123.png', 'pose000123.txt' 등에서 000123 추출
    - 경로 어디에든 'frame\\d+' 패턴이 있으면 잡음
    - 실패 시 default_idx 사용
    """
    base = os.path.basename(path)
    m = re.search(r'(?:frame|depth|pose)?(\d{6,})', base)
    if not m:
        m = re.search(r'frame(\d{6,})', path)
    if m:
        return f"{int(m.group(1)) % 1_000_000:06d}"
    if default_idx is not None:
        return f"{int(default_idx):06d}"
    raise ValueError(f"Cannot infer frame tag from: {path}")

CAM = {"w": 640, "h": 480, "fx": 240.0, "fy": 240.0, "cx": 320.0, "cy": 240.0, "scale": 1000.0}

# ---------- helpers ----------
def downsample_rgbd(I, D, size):
    I_ds = cv2.resize(I, size, interpolation=cv2.INTER_AREA)
    D_ds = cv2.resize(D, size, interpolation=cv2.INTER_NEAREST)
    return I_ds, D_ds

def exposure_align_gray(g1, g2, valid=None):
    if valid is None: valid = np.ones_like(g1, dtype=bool)
    x = g2[valid].astype(np.float32).reshape(-1,1)
    y = g1[valid].astype(np.float32).reshape(-1,1)
    if x.size < 10: return g2, 1.0, 0.0
    A = np.concatenate([x, np.ones_like(x)], axis=1)
    p = np.linalg.lstsq(A, y, rcond=None)[0].squeeze()
    a, b = float(p[0]), float(p[1])
    g2p = np.clip(a*g2 + b, 0, 255).astype(g2.dtype)
    return g2p, a, b

def chroma_delta_map(I1u8, I2u8, eps=1e-6):
    f1 = I1u8.astype(np.float32)/255.0
    f2 = I2u8.astype(np.float32)/255.0
    s1 = f1.sum(axis=2, keepdims=True) + eps
    s2 = f2.sum(axis=2, keepdims=True) + eps
    r1g1 = f1[...,:2]/s1
    r2g2 = f2[...,:2]/s2
    return np.abs(r2g2 - r1g1).sum(axis=2)  # (H,W) L1

def block_reduce_mean(arr, block):
    H, W = arr.shape[:2]
    return cv2.resize(arr.astype(np.float32), (W//block, H//block), interpolation=cv2.INTER_AREA)

# ---------- pose I/O ----------
def parse_pose_txt_to_4x4(path):
    """
    텍스트 파일에서 실수 16개를 추출해 4x4(row-major) 행렬로 반환.
    (4줄×4열이든 한 줄 16개든 상관없이 숫자만 인식)
    """
    with open(path, "r", encoding="utf-8") as f:
        txt = f.read()
    nums = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+(?:[eE][-+]?\d+)?", txt)
    vals = [float(x) for x in nums]
    if len(vals) < 16:
        raise ValueError(f"Pose file '{path}' has only {len(vals)} numbers (<16).")
    if len(vals) > 16:
        print(f"[WARN] Pose '{path}' has {len(vals)} numbers (>16). Using first 16.")
    M = np.array(vals[:16], dtype=np.float64).reshape(4, 4)
    return M

def append_pose_line(traj_path, M4x4):
    """4x4 행렬을 row-major 16개로 평탄화해 traj.txt에 한 줄로 append."""
    flat = M4x4.reshape(-1)
    line = " ".join(f"{v:.12f}" for v in flat)
    with open(traj_path, "a", encoding="utf-8") as f:
        f.write(line + "\n")

def find_pose_for_rgb(rgb_path, pose_dir, pose_list_sorted, idx_fallback=None):
    """
    1) stem 매칭(예: 000123.png ↔ 000123.txt) 시도
    2) 실패 시 정렬된 pose_list의 idx_fallback 사용
    """
    stem = Path(rgb_path).stem
    cand = Path(pose_dir) / f"{stem}.txt"
    if cand.exists():
        return str(cand)
    if idx_fallback is not None and idx_fallback < len(pose_list_sorted):
        return pose_list_sorted[idx_fallback]
    return None

# ---------- change detection ----------
def change_with_local_multiscale(
    I1_rgb, D1_mm, I2_rgb, D2_mm,
    wd=0.5, ws=0.3, wc=0.2, T=0.01,
    alpha_m=0.003, beta=0.005,
    rd_fast=0.005, ssim_fast=0.99,
    chroma_tau=0.05,
    block=32,
    ssim_pix_thr=0.98,
    ssim_block_thr=0.95,
    depth_block_ratio_thr=0.15,
    rgb_block_ratio_thr=0.15,
    low_ssim_pix_ratio_thr=0.01,
    high_chroma_pix_ratio_thr=0.01,
):
    # ---------- global @ 256 ----------
    I1s, D1s = downsample_rgbd(I1_rgb, D1_mm, (256,256))
    I2s, D2s = downsample_rgbd(I2_rgb, D2_mm, (256,256))
    D1m = D1s.astype(np.float32) / CAM["scale"]
    D2m = D2s.astype(np.float32) / CAM["scale"]
    valid = (D1m>0) & (D2m>0)
    z = np.minimum(D1m, D2m)
    tau = alpha_m + beta*z
    depth_mask = (np.abs(D2m-D1m) > tau) & valid
    rd = float(depth_mask.mean()) if valid.any() else 0.0

    g1 = cv2.cvtColor(I1s, cv2.COLOR_RGB2GRAY)
    g2 = cv2.cvtColor(I2s, cv2.COLOR_RGB2GRAY)
    g2p, a_align, b_align = exposure_align_gray(g1, g2, valid)
    s_global, ssim_map = ssim(g1, g2p, win_size=7, gaussian_weights=True, full=True)

    chroma_map = chroma_delta_map(I1s, I2s)
    chroma_high_mask = (chroma_map > chroma_tau) & valid
    dc_mean = float(chroma_map[valid].mean()) if valid.any() else 0.0

    S = wd*rd + ws*(1.0 - float(s_global)) + wc*dc_mean

    # ---------- local @ 256 ----------
    low_ssim_ratio_256  = float((ssim_map < ssim_pix_thr).mean())
    high_chroma_ratio_256 = float(chroma_high_mask.mean())
    depth_block_ratio_256 = block_reduce_mean(depth_mask.astype(np.float32), block)
    ssim_block_mean_256   = block_reduce_mean(ssim_map, block)
    chroma_block_ratio_256= block_reduce_mean(chroma_high_mask.astype(np.float32), block)

    local_trigger_256 = (
        (ssim_block_mean_256.min() < ssim_block_thr) or
        (depth_block_ratio_256.max() > depth_block_ratio_thr) or
        (chroma_block_ratio_256.max() > rgb_block_ratio_thr) or
        (low_ssim_ratio_256  > low_ssim_pix_ratio_thr) or
        (high_chroma_ratio_256 > high_chroma_pix_ratio_thr)
    )

    # ---------- local @ 128 ----------
    I1t, D1t = downsample_rgbd(I1_rgb, D1_mm, (128,128))
    I2t, D2t = downsample_rgbd(I2_rgb, D2_mm, (128,128))
    D1tm = D1t.astype(np.float32)/CAM["scale"]
    D2tm = D2t.astype(np.float32)/CAM["scale"]
    valid_t = (D1tm>0) & (D2tm>0)
    zt = np.minimum(D1tm, D2tm)
    taut = alpha_m + beta*zt
    depth_mask_t = (np.abs(D2tm-D1tm) > taut) & valid_t

    g1t = cv2.cvtColor(I1t, cv2.COLOR_RGB2GRAY)
    g2t = cv2.cvtColor(I2t, cv2.COLOR_RGB2GRAY)
    g2tp, _, _ = exposure_align_gray(g1t, g2t, valid_t)
    s_t, ssim_map_t = ssim(g1t, g2tp, win_size=7, gaussian_weights=True, full=True)
    chroma_map_t = chroma_delta_map(I1t, I2t)
    chroma_high_mask_t = (chroma_map_t > chroma_tau) & valid_t

    low_ssim_ratio_128   = float((ssim_map_t < ssim_pix_thr).mean())
    high_chroma_ratio_128= float(chroma_high_mask_t.mean())
    depth_block_ratio_128= block_reduce_mean(depth_mask_t.astype(np.float32), block//2)
    ssim_block_mean_128  = block_reduce_mean(ssim_map_t, block//2)
    chroma_block_ratio_128 = block_reduce_mean(chroma_high_mask_t.astype(np.float32), block//2)

    local_trigger_128 = (
        (ssim_block_mean_128.min() < ssim_block_thr) or
        (depth_block_ratio_128.max() > depth_block_ratio_thr) or
        (chroma_block_ratio_128.max() > rgb_block_ratio_thr) or
        (low_ssim_ratio_128  > low_ssim_pix_ratio_thr) or
        (high_chroma_ratio_128 > high_chroma_pix_ratio_thr)
    )

    changed = (S > T) or (rd > rd_fast) or (s_global < ssim_fast) or local_trigger_256 or local_trigger_128
    changed_before = (S > T)

    return {
        "changed_before" : bool(changed_before),
        "changed": bool(changed),
        "S": float(S),
        "rd_global": float(rd),
        "ssim_global": float(s_global),
        "dc_mean": float(dc_mean),
        "valid_coverage": float(valid.mean()),
        "fast_or": {"rd_fast": rd_fast, "ssim_fast": ssim_fast},
        "local_256": {
            "min_block_ssim": float(ssim_block_mean_256.min()),
            "max_block_depth_ratio": float(depth_block_ratio_256.max()),
            "max_block_rgb_ratio": float(chroma_block_ratio_256.max()),
            "low_ssim_ratio": float(low_ssim_ratio_256),
            "high_chroma_ratio": float(high_chroma_ratio_256),
        },
        "local_128": {
            "min_block_ssim": float(ssim_block_mean_128.min()),
            "max_block_depth_ratio": float(depth_block_ratio_128.max()),
            "max_block_rgb_ratio": float(chroma_block_ratio_128.max()),
            "low_ssim_ratio": float(low_ssim_ratio_128),
            "high_chroma_ratio": float(high_chroma_ratio_128),
        },
        "params": {
            "T": T, "wd": wd, "ws": ws, "wc": wc,
            "alpha_m": alpha_m, "beta": beta,
            "chroma_tau": chroma_tau,
            "block": block, "ssim_pix_thr": ssim_pix_thr,
            "ssim_block_thr": ssim_block_thr,
            "depth_block_ratio_thr": depth_block_ratio_thr,
            "rgb_block_ratio_thr": rgb_block_ratio_thr,
            "low_ssim_pix_ratio_thr": low_ssim_pix_ratio_thr,
            "high_chroma_pix_ratio_thr": high_chroma_pix_ratio_thr,
        },
        "exposure_align": {"a": float(a_align), "b": float(b_align)},
    }

# ---------- sequence loader ----------
def load_sequence_pairs(rgb_dir1, depth_dir1, rgb_dir2, depth_dir2, rgb_ext="png", depth_ext="png"):
    rgb1 = sorted(glob.glob(os.path.join(rgb_dir1, f"*.{rgb_ext}")))
    rgb2 = sorted(glob.glob(os.path.join(rgb_dir2, f"*.{rgb_ext}")))
    d1   = sorted(glob.glob(os.path.join(depth_dir1, f"*.{depth_ext}")))
    d2   = sorted(glob.glob(os.path.join(depth_dir2, f"*.{depth_ext}")))
    n = min(len(rgb1), len(rgb2), len(d1), len(d2))
    for i in range(n):
        yield rgb1[i], d1[i], rgb2[i], d2[i]

def clamp01(x): 
    return 0.0 if x < 0 else (1.0 if x > 1 else x)

def hinge_up(x, thr, m):   # x ↑가 변화, thr 위로 갈수록 1
    return clamp01(1.0 if m <= 0 else (x - thr) / m)

def hinge_down(x, thr, m): # x ↓가 변화, thr 아래로 갈수록 1
    return clamp01(1.0 if m <= 0 else (thr - x) / m)

def local_scale_score(L, P):
    if not L: return 0.0
    # P의 임계 재사용
    ssim_thr  = P["ssim_block_thr"]
    d_thr     = P["depth_block_ratio_thr"]
    rgb_thr   = P["rgb_block_ratio_thr"]
    low_thr   = P["low_ssim_pix_ratio_thr"]
    chr_thr   = P["high_chroma_pix_ratio_thr"]
    # 각 항목을 0~1로 정규화(임계 근방을 m_*로 선형 스케일링)
    s_ssim  = hinge_down(L.get("min_block_ssim", 1.0), ssim_thr, m_ssim)
    s_depth = hinge_up  (L.get("max_block_depth_ratio", 0.0), d_thr,   m_ratio)
    s_rgb   = hinge_up  (L.get("max_block_rgb_ratio",   0.0), rgb_thr, m_ratio)
    s_low   = hinge_up  (L.get("low_ssim_ratio",        0.0), low_thr, m_ratio)
    s_chr   = hinge_up  (L.get("high_chroma_ratio",     0.0), chr_thr, m_ratio)
    return (w_l_ssim*s_ssim + w_l_depth*s_depth + w_l_rgb*s_rgb + w_l_low*s_low + w_l_chroma*s_chr) / Wsum



def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

# ---------- main ----------
if __name__ == "__main__":
    # 입력(읽기) 경로 #change1이 original, change2가 변화하는것
    seqA_rgb   = "/home/pchy0316/baseline_project/concept-graphs-ali-dev/ai2thor/data/FloorPlan217_change1/color"
    seqA_depth = "/home/pchy0316/baseline_project/concept-graphs-ali-dev/ai2thor/data/FloorPlan217_change1/depth"
    seqB_rgb   = "/home/pchy0316/baseline_project/concept-graphs-ali-dev/ai2thor/data/FloorPlan217_change2/color"
    seqB_depth = "/home/pchy0316/baseline_project/concept-graphs-ali-dev/ai2thor/data/FloorPlan217_change2/depth"
    # 오타 주의: change2
    seqB_pose  = "/home/pchy0316/baseline_project/concept-graphs-ali-dev/ai2thor/data/FloorPlan217_change2/pose"

    # 출력 베이스
    out_dir = "/home/pchy0316/dataset/my_local_data/Replica/room0".strip()
    ensure_dir(out_dir)

    # 서브 폴더
    pre_dir         = os.path.join(out_dir, "pre_results")          # seqA_rgb + seqA_depth
    result_dir      = os.path.join(out_dir, "results")               # seqB_rgb + seqB_depth + pose
    before_temp_dir = os.path.join(out_dir, "before_results_temp")  # 소스(원본 번호)
    before_dir      = os.path.join(out_dir, "before_results")       # 타깃(저장 번호)
    
    
    B_pre_dir         = os.path.join(out_dir, "B_pre_results")          # seqA_rgb + seqA_depth
    B_result_dir      = os.path.join(out_dir, "B_results")               # seqB_rgb + seqB_depth + pose
    B_before_temp_dir = os.path.join(out_dir, "B_before_results_temp")  # 소스(원본 번호)
    B_before_dir      = os.path.join(out_dir, "B_before_results")       # 타깃(저장 번호)

    for d in [pre_dir, result_dir, before_dir]:
        ensure_dir(d)

    # pose 파일 목록
    pose_list_sorted = sorted(glob.glob(os.path.join(seqB_pose, "*.txt")))

    # 변경 프레임 누적 카운트 -> 저장 번호
    save_idx = 0
    cnt = 0
    cnt_before = 0
    for idx, (rgb1_path, depth1_path, rgb2_path, depth2_path) in enumerate(
        load_sequence_pairs(seqA_rgb, seqA_depth, seqB_rgb, seqB_depth)
    ):
        # 읽기
        I1 = cv2.cvtColor(cv2.imread(rgb1_path), cv2.COLOR_BGR2RGB)  # before RGB
        I2 = cv2.cvtColor(cv2.imread(rgb2_path), cv2.COLOR_BGR2RGB)  # after  RGB
        D1 = cv2.imread(depth1_path, cv2.IMREAD_UNCHANGED)           # before Depth (비교용)
        D2 = cv2.imread(depth2_path, cv2.IMREAD_UNCHANGED)           # after  Depth

        out = change_with_local_multiscale(I1, D1, I2, D2)

        # ===== Local → Global 통합 스코어 =====
        p = out["params"]
        L256 = out.get("local_256", {})
        L128 = out.get("local_128", {})

        # 하이퍼파라미터(없으면 기본값 사용)
        alpha   = p.get("alpha_local", 0.40)     # 최종 S에서 로컬 비중
        beta256 = p.get("beta256", 0.50)         # 256 스케일 비중
        beta128 = p.get("beta128", 0.50)         # 128 스케일 비중
        m_ssim  = p.get("margin_ssim",  0.15)    # SSIM 완충 구간(작을수록 민감)
        m_ratio = p.get("margin_ratio", 0.10)    # ratio류 완충 구간
        T_total = p.get("T_total", p["T"])       # 최종 임계값(없으면 기존 T 재사용)

        # 항목별 가중치(합=1 권장)
        w_l_ssim   = p.get("w_l_ssim",    0.30)  # min_block_ssim (작아질수록 변화)
        w_l_depth  = p.get("w_l_depth",   0.25)  # max_block_depth_ratio (커질수록 변화)
        w_l_rgb    = p.get("w_l_rgb",     0.20)  # max_block_rgb_ratio
        w_l_low    = p.get("w_l_lowssim", 0.15)  # low_ssim_ratio
        w_l_chroma = p.get("w_l_chroma",  0.10)  # high_chroma_ratio
        Wsum = max(w_l_ssim + w_l_depth + w_l_rgb + w_l_low + w_l_chroma, 1e-9)

        S_local_256 = local_scale_score(L256, p)
        S_local_128 = local_scale_score(L128, p)
        S_local = (beta256 * S_local_256 + beta128 * S_local_128) / max(beta256 + beta128, 1e-9)

        S_global = out["S"]
        S_total  = (1 - alpha) * S_global + alpha * S_local

        # 결과를 out에 기록(로깅/판정에 활용 가능)
        out["S_local_256"] = S_local_256
        out["S_local_128"] = S_local_128
        out["S_local"]     = S_local
        out["S_total"]     = S_total
        out["changed_total"] = bool(S_total > T_total)

        if out['changed']:
            cnt = cnt+1
        if out['changed_before']:
            cnt_before = cnt_before+1
        # 나중에 결과 표시
        print(f"[Frame {idx:04d}] {os.path.basename(rgb1_path)} vs {os.path.basename(rgb2_path)}")
        print(f"   S={out['S']:.4f} | rd={out['rd_global']:.4f} | ssim={out['ssim_global']:.3f} | changed={out['changed']}") #여기에 추가로 change_before 누적되는 횟수 print
        print(f"   S={out['S']:.4f} | changed_before={out['changed_before']}") #여기에 추가로 changed_before 누적되는 횟수 print        
        print(f"이전에 활용했던거:  {cnt}, 이번에 활용한 거: {cnt_before}")
        
        
        # flag_changed = False
        # while(not flag_changed):
            
        #     flag_changed = True
            
        #     if not out["changed"]:
        #         continue

        #     # 원본 번호(파일명/경로에서)와 저장 번호(변화 프레임 카운트) 분리
        #     src_tag  = get_tag_from_path(rgb2_path, default_idx=idx)   # 예: 000008
        #     save_tag = f"{save_idx:06d}"                               # 예: 000003

        #     # ---------- pre_results: seqA_rgb + seqB_depth ----------
        #     pre_rgb_out   = os.path.join(pre_dir,    f"frame{save_tag}.jpg")
        #     pre_depth_out = os.path.join(pre_dir,    f"depth{save_tag}.png")

        #     cv2.imwrite(pre_rgb_out, cv2.cvtColor(I1, cv2.COLOR_RGB2BGR))
        #     if D2 is None:
        #         print(f"[WARN] D2 is None at idx={idx}. Skip writing pre_results depth.")
        #     else:
        #         D2_u16 = D2.astype(np.uint16) if D2.dtype != np.uint16 else D2
        #         cv2.imwrite(pre_depth_out, D2_u16)

        #     # ---------- result: seqB_rgb + seqB_depth + pose ----------
        #     res_rgb_out   = os.path.join(result_dir, f"frame{save_tag}.jpg")
        #     res_depth_out = os.path.join(result_dir, f"depth{save_tag}.png")
        #     res_pose_out  = os.path.join(result_dir, f"pose{save_tag}.txt")  # ★ 포즈 저장 위치

        #     cv2.imwrite(res_rgb_out, cv2.cvtColor(I2, cv2.COLOR_RGB2BGR))
        #     if D2 is not None:
        #         cv2.imwrite(res_depth_out, D2_u16)

        #     # 포즈 복사 → result/pose{save_tag}.txt
        #     pose_path = find_pose_for_rgb(rgb2_path, seqB_pose, pose_list_sorted, idx_fallback=idx)
        #     if pose_path is None:
        #         print(f"[WARN] Pose file not found for RGB '{os.path.basename(rgb2_path)}' (idx {idx}). Skipping pose copy.")
        #     else:
        #         shutil.copy2(pose_path, res_pose_out)

        #     # ---------- before_results_temp/frame{src_tag} → before_results/frame{save_tag} ----------
        #     src_folder = os.path.join(before_temp_dir, f"frame{src_tag}")
        #     dst_folder = os.path.join(before_dir,     f"frame{save_tag}")
        #     if os.path.isdir(src_folder):
        #         if os.path.exists(dst_folder):
        #             shutil.rmtree(dst_folder)
        #         shutil.copytree(src_folder, dst_folder)
        #     else:
        #         print(f"[WARN] Temp folder not found for src_tag={src_tag}: {src_folder}")

        #     save_idx += 1

        # print(f"\n[Done] Saved {save_idx} changed frames into:")
        # print(f"  - pre_results/ (seqA_rgb → frame{save_tag}.jpg, seqB_depth → depth{save_tag}.png)")
        # print(f"  - result/      (seqB_rgb → frame{save_tag}.jpg, seqB_depth → depth{save_tag}.png, pose{save_tag}.txt)")
        # print(f"  - before_results/ (before_results_temp/frame{{src_tag}} → frame{{save_tag}})")

        flag_changed = False
        while(not flag_changed):
            
            flag_changed = True
            
            if not out["changed"]:
                continue

            # 원본 번호(파일명/경로에서)와 저장 번호(변화 프레임 카운트) 분리
            src_tag  = get_tag_from_path(rgb2_path, default_idx=idx)   # 예: 000008
            save_tag = f"{save_idx:06d}"                               # 예: 000003

            # ---------- pre_results: seqA_rgb + seqA_depth ----------
            pre_rgb_out   = os.path.join(pre_dir,    f"frame{src_tag}.jpg")
            pre_depth_out = os.path.join(pre_dir,    f"depth{src_tag}.png")

            cv2.imwrite(pre_rgb_out, cv2.cvtColor(I1, cv2.COLOR_RGB2BGR))
            if D2 is None:
                print(f"[WARN] D2 is None at idx={idx}. Skip writing pre_results depth.")
            else:
                D2_u16 = D2.astype(np.uint16) if D1.dtype != np.uint16 else D2
                cv2.imwrite(pre_depth_out, D2_u16)

            # ---------- result: seqB_rgb + seqB_depth + pose ----------
            res_rgb_out   = os.path.join(result_dir, f"frame{src_tag}.jpg")
            res_depth_out = os.path.join(result_dir, f"depth{src_tag}.png")
            res_pose_out  = os.path.join(result_dir, f"pose{src_tag}.txt")  # ★ 포즈 저장 위치

            cv2.imwrite(res_rgb_out, cv2.cvtColor(I2, cv2.COLOR_RGB2BGR))
            if D2 is not None:
                cv2.imwrite(res_depth_out, D2_u16)

            # 포즈 복사 → result/pose{save_tag}.txt
            pose_path = find_pose_for_rgb(rgb2_path, seqB_pose, pose_list_sorted, idx_fallback=idx)
            if pose_path is None:
                print(f"[WARN] Pose file not found for RGB '{os.path.basename(rgb2_path)}' (idx {idx}). Skipping pose copy.")
            else:
                shutil.copy2(pose_path, res_pose_out)

            # # ---------- before_results_temp/frame{src_tag} → before_results/frame{save_tag} ---------- # 여기 수정
            # src_folder = os.path.join(before_temp_dir, f"frame{src_tag}")
            # dst_folder = os.path.join(before_dir,     f"frame{src_tag}")
            # if os.path.isdir(src_folder):
            #     if os.path.exists(dst_folder):
            #         shutil.rmtree(dst_folder)
            #     shutil.copytree(src_folder, dst_folder)
            # else:
            #     print(f"[WARN] Temp folder not found for src_tag={src_tag}: {src_folder}")

            # save_idx += 1


        flag_changed_before = False
        while(not flag_changed_before):
            
            flag_changed_before = True

            if not out["changed_before"]:
                continue

            # 원본 번호(파일명/경로에서)와 저장 번호(변화 프레임 카운트) 분리
            src_tag  = get_tag_from_path(rgb2_path, default_idx=idx)   # 예: 000008
            save_tag = f"{save_idx:06d}"                               # 예: 000003

            # ---------- pre_results: seqA_rgb + seqB_depth ----------
            pre_rgb_out   = os.path.join(B_pre_dir,    f"frame{src_tag}.jpg")
            pre_depth_out = os.path.join(B_pre_dir,    f"depth{src_tag}.png")

            cv2.imwrite(pre_rgb_out, cv2.cvtColor(I1, cv2.COLOR_RGB2BGR))
            if D2 is None:
                print(f"[WARN] D2 is None at idx={idx}. Skip writing pre_results depth.")
            else:
                D2_u16 = D2.astype(np.uint16) if D2.dtype != np.uint16 else D2
                cv2.imwrite(pre_depth_out, D2_u16)

            # ---------- result: seqB_rgb + seqB_depth + pose ----------
            res_rgb_out   = os.path.join(B_result_dir, f"frame{src_tag}.jpg")
            res_depth_out = os.path.join(B_result_dir, f"depth{src_tag}.png")
            res_pose_out  = os.path.join(B_result_dir, f"pose{src_tag}.txt")  # ★ 포즈 저장 위치

            cv2.imwrite(res_rgb_out, cv2.cvtColor(I2, cv2.COLOR_RGB2BGR))
            if D2 is not None:
                cv2.imwrite(res_depth_out, D2_u16)

            # 포즈 복사 → result/pose{save_tag}.txt
            pose_path = find_pose_for_rgb(rgb2_path, seqB_pose, pose_list_sorted, idx_fallback=idx)
            if pose_path is None:
                print(f"[WARN] Pose file not found for RGB '{os.path.basename(rgb2_path)}' (idx {idx}). Skipping pose copy.")
            else:
                shutil.copy2(pose_path, res_pose_out)

            # ---------- before_results_temp/frame{src_tag} → before_results/frame{save_tag} ----------
            src_folder = os.path.join(before_temp_dir, f"frame{src_tag}")
            dst_folder = os.path.join(before_dir,     f"frame{src_tag}")
            if os.path.isdir(src_folder):
                if os.path.exists(dst_folder):
                    shutil.rmtree(dst_folder)
                shutil.copytree(src_folder, dst_folder)
            else:
                print(f"[WARN] Temp folder not found for src_tag={src_tag}: {src_folder}")

            save_idx += 1

            # print(f"\n[Done] Saved {save_idx} changed frames into:")
            # print(f"  - pre_results/ (seqA_rgb → frame{save_tag}.jpg, seqB_depth → depth{save_tag}.png)")
            # print(f"  - result/      (seqB_rgb → frame{save_tag}.jpg, seqB_depth → depth{save_tag}.png, pose{save_tag}.txt)")
            # print(f"  - before_results/ (before_results_temp/frame{{src_tag}} → frame{{save_tag}})")

# import cv2
# import numpy as np

# def get_tag_from_path(path: str, default_idx: int = None) -> str:
#     """
#     파일/경로에서 6자리 프레임 번호를 추출.
#     - 'frame000123.jpg', 'depth000123.png', 'pose000123.txt' 등에서 000123 추출
#     - 경로 어디에든 'frame\\d+' 패턴이 있으면 잡음
#     - 실패 시 default_idx 사용
#     """
#     base = os.path.basename(path)
#     m = re.search(r'(?:frame|depth|pose)?(\d{6,})', base)
#     if not m:
#         m = re.search(r'frame(\d{6,})', path)
#     if m:
#         return f"{int(m.group(1)) % 1_000_000:06d}"
#     if default_idx is not None:
#         return f"{int(default_idx):06d}"
#     raise ValueError(f"Cannot infer frame tag from: {path}")

