# EBT: EEG Branch Transformer

EEG 신호 기반 Fatigue 및 Mental Workload 분류 모델

## 연구 개요
EEG-Deformer(IEEE J-BHI 2024)의 벤치마크 데이터셋에서 
Fatigue 및 Mental Workload 분류 성능 개선을 목표로 설계한 
Branch Transformer 아키텍처

## 주요 특징
- EEG 시계열 신호의 분기(Branch) 구조 처리
- Ablation Study를 통한 모델 구조 검증
- 데이터 증강(augmentation) 및 커스텀 손실 함수 적용

## 기술 스택
Python, PyTorch, NumPy

## 파일 구조
- model_components.py : 메인 Branch Transformer 아키텍처
- model_components_ablation.py : Ablation 실험용 모델 변형
- dataset.py : EEG 데이터 로딩 및 전처리
- augmentations.py : 데이터 증강
- losses.py : 커스텀 손실 함수
- train_utils_ablation.py : 학습 유틸리티
