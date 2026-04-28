import numpy as np


class EEGAugmentation:
    """
    EEG 신경생리학적 특성을 보존하는 안전한 데이터 증강 클래스
    
    제거된 위험 증강:
    - channel_shuffle: 공간 정보 파괴
    - spectral_augmentation: 주파수 대역의 생리학적 의미 파괴
    - time_warping: 주파수 특성 왜곡
    
    수정된 증강:
    - amplitude_scaling: 범위 축소 (±10% → ±5%)
    """
    def __init__(self, fs=128, apply_prob=0.5):
        self.fs = fs
        self.apply_prob = apply_prob

    def __call__(self, data, label=None):
        augmented_data = data.copy()

        if label is not None:
            augmentations = self._get_label_specific_augmentations(label)
        else:
            augmentations = self._get_default_augmentations()

        for aug_func, prob in augmentations:
            if np.random.rand() < prob:
                augmented_data = aug_func(augmented_data)

        return augmented_data

    def _get_label_specific_augmentations(self, label):
        """레이블별 증강 - 공간정보 파괴 증강 제거"""
        if label == 0:  # Alert/Low workload
            return [
                (self.time_shift, 0.6),
                (self.gaussian_noise, 0.4),
                (self.channel_shuffle, 0.3),  # ❌ 제거: 공간정보 파괴
            ]
        else:  # Fatigued/High workload
            return [
                (self.time_shift, 0.3),
                (self.gaussian_noise, 0.5),
                (self.amplitude_scaling, 0.5),  # 범위 축소됨
                (self.spectral_augmentation, 0.4),  # ❌ 제거: 주파수 의미 파괴
            ]

    def _get_default_augmentations(self):
        """기본 증강 - 생리학적으로 안전한 증강만 유지"""
        return [
            (self.time_shift, 0.5),
            (self.gaussian_noise, 0.5),
            (self.amplitude_scaling, 0.4),  # 확률과 범위 모두 축소
            (self.channel_shuffle, 0.3),        # ❌ 제거: 공간정보 파괴
            (self.spectral_augmentation, 0.4),  # ❌ 제거: 주파수 의미 파괴
        ]

    def time_shift(self, data, max_shift_sec=0.1):  # 0.2초 → 0.1초로 축소
        """
        시간축 이동 - 피로도 연구에서 상대적으로 안전
        범위를 축소하여 더욱 보수적으로 적용
        
        data: (channels, time_steps)
        """
        time_length = data.shape[1]
        max_shift = int(self.fs * max_shift_sec)  # 더 작은 시프트 범위
        shift = np.random.randint(-max_shift, max_shift)
        
        if shift > 0:
            shifted = np.concatenate((data[:, shift:], np.zeros((data.shape[0], shift))), axis=1)
        elif shift < 0:
            shifted = np.concatenate((np.zeros((data.shape[0], -shift)), data[:, :shift]), axis=1)
        else:
            shifted = data
        return shifted

    def gaussian_noise(self, data, mean=0.0, std=0.005):  # std 0.01 → 0.005로 축소
        """
        작은 가우시안 노이즈 - 실제 EEG 측정 노이즈 모사
        더욱 보수적인 노이즈 수준 적용
        """
        noise = np.random.normal(mean, std, size=data.shape)
        return data + noise

    def amplitude_scaling(self, data, scale_lim=0.05):  # 0.1 → 0.05로 축소 (±5%)
        """
        보수적 진폭 스케일링 - 생리학적 범위 내에서만
        ±5% 범위로 제한하여 자연스러운 개인차 수준만 모사
        """
        scaling = np.random.uniform(1 - scale_lim, 1 + scale_lim)
        return data * scaling

    def time_warping(self, data, warp_strength=0.2):
        """
        선형 인덱스 왜곡 기법으로 time warping 적용
        """
        time_length = data.shape[1]
        original_indices = np.arange(time_length)
        warp_factor = 1 + np.random.uniform(-warp_strength, warp_strength)
        warped_indices = original_indices * warp_factor
        warped_indices = np.clip(warped_indices, 0, time_length - 1)

        warped_data = np.zeros_like(data)
        for ch in range(data.shape[0]):
            warped_data[ch] = np.interp(original_indices, warped_indices, data[ch])

        return warped_data

    def channel_shuffle(self, data, shuffle_prob=0.2):
        """
        채널 순서를 shuffle_prob 확률로 섞음
        """
        n_channels = data.shape[0]
        if np.random.rand() < shuffle_prob:
            idx = np.arange(n_channels)
            np.random.shuffle(idx)
            return data[idx, :]
        return data

    def spectral_augmentation(self, data, freq_mask_max_width=10, num_masks=1):
        """
        주파수 축에서 마스크를 씌워 스펙트럼 증강
        data: (channels, time_steps)
        """
        augmented = data.copy()
        for _ in range(num_masks):
            ch = np.random.randint(0, augmented.shape[0])
            freq_width = np.random.randint(1, freq_mask_max_width)
            freq_start = np.random.randint(0, augmented.shape[1] - freq_width)
            augmented[ch, freq_start:freq_start + freq_width] = 0
        return augmented
