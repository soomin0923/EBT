# data_utils.py

import os
import pickle
import logging
from pathlib import Path
import numpy as np

logger = logging.getLogger(__name__)

# 전극 그리드 정의
FATIG_GRID = {
    'FP1': (0, 1), 'FP2': (0, 3),
    'F7': (1, 0), 'F3': (1, 1), 'Fz': (1, 2), 'F4': (1, 3), 'F8': (1, 4),
    'FT7': (1.5, 0), 'FC3': (1.5, 1), 'FCz': (1.5, 2), 'FC4': (1.5, 3), 'FT8': (1.5, 4),
    'T3': (2, 0), 'C3': (2, 1), 'Cz': (2, 2), 'C4': (2, 3), 'T4': (2, 4),
    'TP7': (2.5, 0), 'CP3': (2.5, 1), 'CPz': (2.5, 2), 'CP4': (2.5, 3), 'TP8': (2.5, 4),
    'T5': (3, 0), 'P3': (3, 1), 'Pz': (3, 2), 'P4': (3, 3), 'T6': (3, 4),
    'O1': (4, 1), 'Oz': (4, 2), 'O2': (4, 3)
}

MWL_GRID = {
    'Fp1': (0, 1), 'Fp2': (0, 3),
    'F7': (1, 0), 'F3': (1, 1), 'Fz': (1, 2), 'F4': (1, 3), 'F8': (1, 4),
    'T3': (2, 0), 'C3': (2, 1), 'Cz': (2, 2), 'C4': (2, 3), 'T4': (2, 4),
    'T5': (3, 0), 'P3': (3, 1), 'Pz': (3, 2), 'P4': (3, 3), 'T6': (3, 4),
    'O1': (4, 1), 'Oz': (4, 2), 'O2': (4, 3)
}

STEW_GRID = {
    'AF3': (0, 1), 'AF4': (0, 3),
    'F7': (1, 0),  'F3': (1, 1), 'F4': (1, 3), 'F8': (1, 4),
    'T7': (2, 0),  'FC5': (2, 1), 'FC6': (2, 3), 'T8': (2, 4),
    'P7': (3, 0),  'P8': (3, 4),
    'O1': (4, 1),  'O2': (4, 3),
}
STEW_CHANNELS = ['AF3','AF4','F7','F3','F4','F8','FC5','FC6','T7','T8','P7','P8','O1','O2']

def _canon_eeg(name: str) -> str:
    """채널명 정규화: 대소문자/공백 통일 (예: 'af3' -> 'AF3')"""
    return (name or "").strip().upper()

def load_electrode_info(data_info_path, dataset):
    """
    data_info_path: info.pkl 경로
    dataset: 'FATIG' or 'MWL' or 'STEW'
    """
    dataset = str(dataset).upper()

    # ⭐ 1. STEW는 pickle 파일이 아니라 ratings.txt라서, 여기서 바로 return 하고
    #      아래 pickle.load 부분을 건너뛰게 만듦
    if dataset == 'STEW':
        return STEW_CHANNELS, STEW_GRID

    # ⭐ 2. FATIG / MWL만 pickle을 사용
    with open(data_info_path, 'rb') as f:
        info = pickle.load(f)

    electrode_names = info.get('BL', [])

    if dataset == 'FATIG':
        grid = FATIG_GRID
    elif dataset == 'MWL':
        grid = MWL_GRID
    else:
        grid = MWL_GRID  # 기본값

    return electrode_names, grid


def get_electrode_positions(electrode_names, grid_dict):
    """
    전극 이름 리스트를 받아 해당하는 위치 좌표를 반환
    """
    positions = []
    for name in electrode_names:
        if name in grid_dict:
            positions.append(grid_dict[name])
        else:
            # 기본 위치 (0, 0)로 설정
            positions.append((0, 0))
    return np.array(positions)

def get_electrode_positions(electrode_names, grid_dict):
    positions = []
    for name in electrode_names:
        key = _canon_eeg(name)
        if key in grid_dict:
            positions.append(grid_dict[key])
        else:
            positions.append((0, 0))
    return np.array(positions)


def compute_electrode_distances(positions):
    """
    전극 위치들 간의 유클리드 거리 행렬 계산
    positions: (n_electrodes, 2) array
    returns: (n_electrodes, n_electrodes) distance matrix
    """
    n_electrodes = len(positions)
    distances = np.zeros((n_electrodes, n_electrodes))
    
    for i in range(n_electrodes):
        for j in range(n_electrodes):
            distances[i, j] = np.sqrt(
                (positions[i][0] - positions[j][0])**2 + 
                (positions[i][1] - positions[j][1])**2
            )
    
    return distances

def get_grid_shape_from_grid_dict(grid_dict):
    max_row = int(max(row for row, col in grid_dict.values()))
    max_col = int(max(col for row, col in grid_dict.values()))
    return (max_row + 1, max_col + 1)
def map_eeg_to_grid(eeg_data, grid_dict, grid_shape, channel_names):
    """
    eeg_data: (channels, time)
    grid_dict: electrode_name → (row, col)
    grid_shape: (H, W)
    channel_names: 실제 데이터의 채널명 리스트
    """
    mapped = np.zeros((grid_shape[0], grid_shape[1], eeg_data.shape[1]), dtype=np.float32)
    name2idx = {_canon_eeg(ch): i for i, ch in enumerate(channel_names)}
    canon_grid = {_canon_eeg(n): (int(r), int(c)) for n, (r, c) in grid_dict.items()}

    for name, (row, col) in canon_grid.items():
        if name not in name2idx:
            continue
        mapped[row, col, :] = eeg_data[name2idx[name]]
    return mapped
def map_eeg_to_grid_2d(eeg_ch_t, grid, grid_shape=None, channel_names=None, fill_value=0.0):
    import numpy as np, math, logging
    logger = logging.getLogger(__name__)

    x = np.asarray(eeg_ch_t)
    if x.ndim != 2:
        raise ValueError(f"Expected 2D (C,T) array, got shape={x.shape}")
    if x.shape[0] > x.shape[1]:
        x = x.T
    C, T = x.shape

    eeg_avg = x.mean(axis=1).astype(np.float32)

    if (channel_names is None) or (len(channel_names) < C):
        channel_names = [f"Ch{i+1}" for i in range(C)]
    if len(channel_names) > C:
        channel_names = channel_names[:C]
    name2idx = {_canon_eeg(n): i for i, n in enumerate(channel_names)}

    if grid_shape is None:
        if grid:
            rows = [int(math.floor(r)) for (r, c) in grid.values()]
            cols = [int(math.floor(c)) for (r, c) in grid.values()]
            H = max(rows) + 1 if rows else C
            W = max(cols) + 1 if cols else 1
        else:
            H = int(np.ceil(np.sqrt(C)))
            W = int(np.ceil(C / H))
    else:
        H, W = grid_shape

    mapped = np.full((H, W), fill_value, dtype=np.float32)
    used = 0
    canon_grid = {_canon_eeg(n): (r, c) for n, (r, c) in (grid or {}).items()}
    for name, (r, c) in canon_grid.items():
        if name not in name2idx:
            continue
        idx = name2idx[name]
        rr, cc = int(round(r)), int(round(c))
        if 0 <= rr < H and 0 <= cc < W:
            mapped[rr, cc] = eeg_avg[idx]
            used += 1

    if used < max(3, int(0.5 * C)):
        W = int(np.ceil(np.sqrt(C)))
        H = int(np.ceil(C / W))
        mapped = np.full((H, W), fill_value, dtype=np.float32)
        k = 0
        for i in range(H):
            for j in range(W):
                if k < C:
                    mapped[i, j] = eeg_avg[k]
                    k += 1
        logger.warning(f"[map_eeg_to_grid_2d] Fallback auto-grid used (mapped={used}, C={C}).")

    return mapped

def load_subject_data(file_path):
    """단일 피클 파일에서 subject별 EEG 데이터와 레이블을 로드"""
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    subject_data = []
    subject_labels = []
    if 'data' in data and 'label' in data:
        if isinstance(data['data'], list) and len(data['data']) > 0:
            for batch_data, batch_labels in zip(data['data'], data['label']):
                if isinstance(batch_data, np.ndarray) and batch_data.ndim == 3:
                    for trial_idx in range(batch_data.shape[0]):
                        subject_data.append(batch_data[trial_idx])
                        label = batch_labels[trial_idx]
                        if isinstance(label, np.ndarray):
                            label = label.item() if label.size == 1 else label.flatten()[0]
                        subject_labels.append(label)
                else:
                    for d, l in zip(batch_data, batch_labels):
                        subject_data.append(d)
                        if isinstance(l, np.ndarray):
                            l = l.item() if l.size == 1 else l.flatten()[0]
                        subject_labels.append(l)
    return np.array(subject_data), np.array(subject_labels)

def load_all_subjects(data_dir, pattern="sub*.pkl"):
    """
    지정된 디렉터리에서 모든 subject 파일을 찾아 concat 후 반환
    - 우선 .pkl을 찾고, 없으면 STEW 형식(.txt)을 자동 처리
    """
    data_dir = Path(data_dir)

    # 1) PKL 우선
    pkl_files = sorted(list(data_dir.glob(pattern)))
    if pkl_files:
        all_data, all_labels, all_subject_ids = [], [], []
        for file_path in pkl_files:
            subject_id = file_path.stem
            data, labels = load_subject_data(file_path)  # 기존 함수 그대로 사용
            all_data.append(data)
            all_labels.append(labels)
            all_subject_ids.extend([subject_id] * len(labels))
        X = np.concatenate(all_data, axis=0)
        y = np.concatenate(all_labels, axis=0)
        subject_ids = np.array(all_subject_ids)
        return X, y, subject_ids

    # 2) PKL이 없다면: STEW(.txt) 폴백
    txt_hi = sorted(data_dir.glob("sub*_hi.txt"))
    txt_lo = sorted(data_dir.glob("sub*_lo.txt"))
    if txt_hi or txt_lo:
        # 필요시 epoch_len/overlap 조절 가능
        return load_stew_text_dataset(str(data_dir), epoch_len=512, overlap=0.5)

    # 3) 아무것도 없으면 에러
    raise FileNotFoundError(f"No files found in {data_dir}")

def normalize_per_subject(X, subs):
    """
    주제별(subject)로 채널 단위 평균-표준편차 정규화
    - X: (N, channels, time_steps), subs: 각 샘플의 subject ID 벡터
    """
    X_norm = np.zeros_like(X)
    unique_subjects = np.unique(subs)

    for subj in unique_subjects:
        mask = (subs == subj)
        X_subj = X[mask]

        for ch_idx in range(X_subj.shape[1]):
            mean = np.mean(X_subj[:, ch_idx, :])
            std = np.std(X_subj[:, ch_idx, :])
            X_norm[mask, ch_idx, :] = (X_subj[:, ch_idx, :] - mean) / (std + 1e-8)

    return X_norm


# === STEW: 텍스트 데이터셋 지원 =========================================
from pathlib import Path
import numpy as np

def _stew_read_matrix(txt_path: Path) -> np.ndarray:
    """
    txt: 공백 구분 실수, 일반적으로 (time, channels).
    반환: (channels, time)
    """
    arr = np.loadtxt(str(txt_path), dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[:, None]  # (T,) -> (T,1)
    # 보편적으로 시간(행)이 채널(열)보다 훨씬 큼. (T,C) -> (C,T)로 전치
    if arr.shape[0] >= arr.shape[1]:
        arr = arr.T
    return arr  # (C, T)

def _segment_trials(data_ch_t: np.ndarray, epoch_len: int = 1000, step: int | None = None) -> np.ndarray:
    """
    data_ch_t: (channels, time)
    epoch_len: 잘라낼 길이(타임스텝)
    step: 슬라이딩 간격(기본: epoch_len, 즉 non-overlap)
    반환: (N, channels, epoch_len)
    """
    C, T = data_ch_t.shape
    if step is None or step <= 0:
        step = epoch_len
    if T < epoch_len:
        return np.empty((0, C, epoch_len), dtype=np.float32)

    chunks = []
    for s in range(0, T - epoch_len + 1, step):
        chunks.append(data_ch_t[:, s:s+epoch_len])
    if not chunks:
        return np.empty((0, C, epoch_len), dtype=np.float32)
    return np.stack(chunks, axis=0)

def load_stew_text_dataset(
    root_dir: str,
    epoch_len: int = 1000,
    overlap: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    STEW 텍스트 데이터셋 로더
    - root_dir: ./data_processed/data_eeg_STEW
    - 파일명 규칙: subXX_hi.txt, subXX_lo.txt
    - 레이블: hi=1, lo=0
    - 반환: X (N, C, T), y (N,), subject_ids (N,)
    """
    root = Path(root_dir)
    hi_files = sorted(root.glob("sub*_hi.txt"))
    lo_files = sorted(root.glob("sub*_lo.txt"))

    # subject 리스트 수집
    subj_ids = sorted(set([p.stem.split("_")[0] for p in hi_files + lo_files]))
    X_list, y_list, sid_list = [], [], []

    step = epoch_len - int(epoch_len * overlap) if (0.0 < overlap < 1.0) else epoch_len

    for sid in subj_ids:
        hi_path = root / f"{sid}_hi.txt"
        lo_path = root / f"{sid}_lo.txt"

        if hi_path.exists():
            mat_hi = _stew_read_matrix(hi_path)  # (C,T)
            seg_hi = _segment_trials(mat_hi, epoch_len=epoch_len, step=step)  # (N,C,T)
            if seg_hi.size:
                X_list.append(seg_hi)
                y_list.append(np.ones((seg_hi.shape[0],), dtype=np.int64))
                sid_list.extend([sid]*seg_hi.shape[0])

        if lo_path.exists():
            mat_lo = _stew_read_matrix(lo_path)  # (C,T)
            seg_lo = _segment_trials(mat_lo, epoch_len=epoch_len, step=step)
            if seg_lo.size:
                X_list.append(seg_lo)
                y_list.append(np.zeros((seg_lo.shape[0],), dtype=np.int64))
                sid_list.extend([sid]*seg_lo.shape[0])

    if not X_list:
        raise FileNotFoundError(f"No STEW txt files found under {root}")

    X = np.concatenate(X_list, axis=0)  # (N,C,T)
    y = np.concatenate(y_list, axis=0)  # (N,)
    subject_ids = np.array(sid_list)
    return X, y, subject_ids

def make_auto_grid(n_channels: int):
    """
    전극명이 없을 때 'Ch1..ChN'으로 이름을 만들고,
    대략적인 정사각 배치의 그리드를 생성.
    반환: (electrode_names, grid_dict)
    """
    names = [f"Ch{i+1}" for i in range(n_channels)]
    W = int(np.ceil(np.sqrt(n_channels)))
    H = int(np.ceil(n_channels / W))
    grid = {}
    k = 0
    for r in range(H):
        for c in range(W):
            if k < n_channels:
                grid[names[k]] = (r, c)
                k += 1
    return names, grid
# --- STEW 텍스트 지원 유틸 ---
from pathlib import Path
import numpy as np

def _stew_read_matrix(txt_path: Path) -> np.ndarray:
    """공백 구분 실수 텍스트 -> (channels, time)"""
    arr = np.loadtxt(str(txt_path), dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[:, None]  # (T,) -> (T,1)
    # 보통 (time, channels)이므로 전치하여 (channels, time)로 맞춤
    if arr.shape[0] >= arr.shape[1]:
        arr = arr.T
    return arr  # (C, T)

def _segment_trials(data_ch_t: np.ndarray, epoch_len: int = 1000, step: int | None = None) -> np.ndarray:
    """(C,T) → (N,C,epoch_len) 슬라이싱"""
    C, T = data_ch_t.shape
    if step is None or step <= 0:
        step = epoch_len
    if T < epoch_len:
        return np.empty((0, C, epoch_len), dtype=np.float32)
    chunks = []
    for s in range(0, T - epoch_len + 1, step):
        chunks.append(data_ch_t[:, s:s+epoch_len])
    return np.stack(chunks, axis=0) if chunks else np.empty((0, C, epoch_len), dtype=np.float32)

def load_stew_text_dataset(data_dir, epoch_len=1500, overlap=0.5):
    import os, glob
    import numpy as np

    # (선택) ratings.txt 읽기 – 필요하면 여기서 subject 별 rating을 활용
    ratings_path = os.path.join(data_dir, 'ratings.txt')
    if os.path.exists(ratings_path):
        ratings = np.loadtxt(ratings_path, delimiter=',', dtype=int)
        # ratings: (subject_id, condition, score) 형식이라고 가정
        # subject_id -> score 매핑 만들고 쓰고 싶으면 여기에서 처리

    hi_files = sorted(glob.glob(os.path.join(data_dir, 'sub*_hi.txt')))
    lo_files = sorted(glob.glob(os.path.join(data_dir, 'sub*_lo.txt')))

    X_list, y_list, subj_list = [], [], []

    def add_segments(arr, label, subj_id):
        step = int(epoch_len * (1 - overlap))
        for start in range(0, arr.shape[0] - epoch_len + 1, step):
            seg = arr[start:start+epoch_len, :]      # (time, ch)
            seg = seg.T.astype(np.float32)          # (ch, time) 로 맞추기
            X_list.append(seg)
            y_list.append(label)
            subj_list.append(subj_id)

    # hi = 1, lo = 0 예시
    for path in hi_files:
        basename = os.path.basename(path)      # sub01_hi.txt
        subj_id = int(basename[3:5])          # '01' -> 1
        data = np.loadtxt(path)               # (T, ch)
        add_segments(data, label=1, subj_id=subj_id)

    for path in lo_files:
        basename = os.path.basename(path)      # sub01_lo.txt
        subj_id = int(basename[3:5])
        data = np.loadtxt(path)
        add_segments(data, label=0, subj_id=subj_id)

    X = np.stack(X_list, axis=0)              # (N, ch, time)
    y = np.array(y_list, dtype=np.int64)
    subject_ids = np.array(subj_list, dtype=np.int64)

    return X, y, subject_ids

