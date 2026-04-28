# losses.py

import torch
import torch.nn.functional as F


def contrastive_loss(features, labels, temperature=0.5):
    """
    SimCLR 스타일의 Contrastive Loss
    - features: (B, D), L2 정규화된 특징 벡터들
    - labels: (B,)
    """
    batch_size = features.size(0)
    labels = labels.contiguous().view(-1, 1)
    mask = torch.eq(labels, labels.T).float().to(features.device)

    features = F.normalize(features, dim=1)
    similarity = torch.matmul(features, features.T) / temperature

    logits_mask = torch.ones_like(mask) - torch.eye(batch_size).to(features.device)
    mask = mask * logits_mask

    exp_sim = torch.exp(similarity) * logits_mask
    log_prob = similarity - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)

    loss = -(mask * log_prob).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-8)
    return loss.mean()


def focal_loss(logits, targets, gamma=2.0, alpha=0.25):
    """
    Focal Loss for 클래스 불균형 문제
    - logits: (B, C)
    - targets: (B,)
    """
    CE_loss = F.cross_entropy(logits, targets, reduction='none')
    pt = torch.exp(-CE_loss)
    loss = alpha * (1 - pt) ** gamma * CE_loss
    return loss.mean()
