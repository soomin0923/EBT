# dataset.py

import numpy as np
import torch
from torch.utils.data import Dataset
from scipy import signal
from augmentations import EEGAugmentation
from data_utils import FATIG_GRID, MWL_GRID, map_eeg_to_grid_2d, get_grid_shape_from_grid_dict

class EnhancedEEGDataset(Dataset):
    """
    electrode_names: (channels,) 데이터 채널 이름 리스트 (dataset_info.pkl에서 BL로 불러오기)
    수정: 3D spatial-temporal → 2D spatial map으로 변경
    """
    def __init__(self, X, y, electrode_names, grid, fs=128, stft_params=None, eps=1e-8, augment=False):
        self.fs = fs
        self.eps = eps
        self.augment = augment
        self.eeg_augment = EEGAugmentation(fs=fs) if augment else None
        self.raw = []
        self.stft = []
        self.spatial_maps = []  # 수정: 3D → 2D
        self.labels = torch.LongTensor(y)

        if stft_params is None:
            seq_len = X.shape[2]
            if seq_len <= 500:
                nperseg, noverlap = 64, 32
            elif seq_len <= 1000:
                nperseg, noverlap = 128, 64
            else:
                nperseg, noverlap = 256, 128
        else:
            nperseg = stft_params.get('nperseg', 128)
            noverlap = stft_params.get('noverlap', 64)

        grid_shape = get_grid_shape_from_grid_dict(grid)

        for i in range(X.shape[0]):
            data = X[i]  # (channels, time)
            self.raw.append(torch.FloatTensor(data))

            # STFT
            t_tensor = torch.FloatTensor(data)
            S = torch.stft(
                t_tensor,
                n_fft=nperseg,
                hop_length=nperseg - noverlap,
                win_length=nperseg,
                window=torch.hann_window(nperseg),
                return_complex=False
            )
            real, imag = S.unbind(-1)
            mag = torch.sqrt(real ** 2 + imag ** 2 + self.eps)
            log_mag = torch.log(mag + self.eps)
            cmin = log_mag.amin(dim=(1, 2), keepdim=True)
            cmax = log_mag.amax(dim=(1, 2), keepdim=True)
            norm = (log_mag - cmin) / (cmax - cmin + self.eps)
            self.stft.append(norm)

            # 수정: 2D Spatial map 생성 (시간축 평균화)
            spatial_map_2d = map_eeg_to_grid_2d(data, grid, grid_shape, channel_names=electrode_names)
            self.spatial_maps.append(torch.FloatTensor(spatial_map_2d))

        self.raw = torch.stack(self.raw)
        self.stft = torch.stack(self.stft)
        self.spatial_maps = torch.stack(self.spatial_maps, dim=0)  # (N, H, W)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        raw = self.raw[idx]
        stft = self.stft[idx]
        spatial_map = self.spatial_maps[idx]  # (H, W)
        label = self.labels[idx]

        if self.augment:
            raw_np = raw.numpy()
            raw_np = self.eeg_augment(raw_np, label.item())
            raw = torch.FloatTensor(raw_np)

            shift = np.random.randint(-10, 10)
            if shift > 0:
                raw = torch.cat([raw[:, shift:], torch.zeros_like(raw[:, :shift])], dim=1)
            elif shift < 0:
                raw = torch.cat([torch.zeros_like(raw[:, :abs(shift)]), raw[:, :shift]], dim=1)
            noise_level = np.random.uniform(0.01, 0.05)
            raw = raw + noise_level * torch.randn_like(raw)

        return raw, stft, spatial_map, label