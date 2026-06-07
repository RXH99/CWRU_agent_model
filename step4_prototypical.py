"""
Step 4: 原型网络（Prototypical Networks）— 小样本故障诊断
在 W2 训练好的 1D-CNN 基础上，改造成小样本分类器。

核心思路：
  全量训练 → 模型知道"信号长什么样"
  原型网络 → 模型学会"比较信号之间的相似度"

运行: python step4_prototypical.py
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os, sys, random

# ──────── 配置 ────────
EPISODES = 2000          # 总训练 episode 数
VAL_EPISODES = 200       # 验证 episode 数
WAYS = 4                 # 每 episode 类别数（CWRU 有 4 类）
SHOT = 5                 # 每类支持集样本数（小样本：5-shot）
QUERY = 5                # 每类查询集样本数
LR = 0.0001              # 微调学习率（比全量训练小）
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
INPUT_FILE = "preprocessed_data.npz"

# 类别名称
CLASS_NAMES = ["Normal", "Inner Race", "Ball", "Outer Race"]

# ──────── 1. 数据准备 ────────

class CWRUDataset(Dataset):
    """读取预处理的 .npz 文件，按分割键加载"""
    def __init__(self, npz_path, split="train"):
        data = np.load(npz_path)
        if split == "train":
            self.X = torch.tensor(data["X_train"]).unsqueeze(1)
            self.y = torch.tensor(data["y_train"])
        else:
            self.X = torch.tensor(data["X_test"]).unsqueeze(1)
            self.y = torch.tensor(data["y_test"])
        print(f"加载数据 [{split}]: {len(self.y)} 个样本, 形状 {self.X.shape}")

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def get_all_indices(dataset):
    """返回数据集所有索引（数据已在 step2 中按时间顺序预分割，无需再 split）"""
    indices = list(range(len(dataset.y)))
    indices_by_class = {}
    for i, label in enumerate(dataset.y):
        label = label.item()
        indices_by_class.setdefault(label, []).append(i)
    return indices, indices_by_class


# ──────── 2. 特征提取器（复用 W2 的 CNN1D.features） ────────

class Encoder(nn.Module):
    """
    去掉分类头的 1D-CNN，只保留特征提取部分。
    输入: (B, 1, 1024) → 输出: (B, 64) 特征向量
    """
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=15, stride=2, padding=7),
            nn.BatchNorm1d(16), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )

    def forward(self, x):
        return self.features(x).squeeze(-1)  # (B, 64)


def load_pretrained_encoder(encoder, pretrained_path="cnn1d_model.pth"):
    """
    加载 W2 训练好的全量模型权重，只取 features 部分。
    """
    if not os.path.exists(pretrained_path):
        print(f"⚠️ 找不到 {pretrained_path}，将从头训练 Encoder")
        return encoder

    state_dict = torch.load(pretrained_path, map_location="cpu")

    # 只加载 features 模块的权重（匹配键名）
    encoder_dict = encoder.state_dict()
    matched = {}
    for k, v in state_dict.items():
        if k in encoder_dict:
            matched[k] = v
        elif k.startswith("features.") and k in encoder_dict:
            matched[k] = v
        elif k.startswith("classifier.") or k.startswith("cls."):
            continue  # 跳过分类头

    if matched:
        encoder.load_state_dict(matched, strict=False)
        print(f"✅ 加载预训练权重: {len(matched)}/{len(encoder_dict)} 层匹配")
    else:
        print("⚠️ 没有匹配的预训练权重，将从头训练")
    return encoder


# ──────── 3. Episodic 采样器 ────────

class EpisodicSampler:
    """
    从数据集中采样训练 episode。
    每个 episode: 从全部类别中随机选 WAYS 类，
    每类随机选 SHOT+QUERY 个样本。
    """
    def __init__(self, dataset, indices, ways=4, shot=5, query=5):
        self.X = dataset.X
        self.y = dataset.y
        self.indices = indices
        self.ways = ways
        self.shot = shot
        self.query = query

        # 按类别分组索引
        self.class_to_indices = {}
        for idx in indices:
            label = self.y[idx].item()
            self.class_to_indices.setdefault(label, []).append(idx)

        self.available_classes = list(self.class_to_indices.keys())

    def sample_episode(self):
        """
        返回: support_x, support_y, query_x, query_y
              每个都是 torch tensor
        """
        # 随机选 ways 个类别
        classes = random.sample(self.available_classes, self.ways)

        support_x, support_y = [], []
        query_x, query_y = [], []

        for cls in classes:
            idxs = self.class_to_indices[cls]
            selected = random.sample(idxs, self.shot + self.query)

            # 前 shot 个做 support
            for i in selected[:self.shot]:
                support_x.append(self.X[i])
                support_y.append(cls)
            # 后 query 个做 query
            for i in selected[self.shot:]:
                query_x.append(self.X[i])
                query_y.append(cls)

        support_x = torch.stack(support_x)
        support_y = torch.tensor(support_y)
        query_x = torch.stack(query_x)
        query_y = torch.tensor(query_y)

        return support_x, support_y, query_x, query_y


# ──────── 4. 原型网络核心 ────────

def prototypical_loss(encoder, support_x, support_y, query_x, query_y, device):
    """
    原型网络前向 + 损失计算。

    1. 用 encoder 将 support 和 query 映射到特征空间
    2. 每类的 support embedding 取均值 → 类原型
    3. query embedding 与每个原型的欧氏距离 → Softmax → CrossEntropy
    """
    # 特征提取
    support_emb = encoder(support_x.to(device))  # (WAYS*SHOT, 64)
    query_emb = encoder(query_x.to(device))      # (WAYS*QUERY, 64)

    ways = len(torch.unique(support_y))
    shot = support_x.size(0) // ways

    # 计算类原型
    prototypes = []
    for cls in range(ways):
        cls_mask = (support_y == cls)
        proto = support_emb[cls_mask].mean(dim=0)
        prototypes.append(proto)
    prototypes = torch.stack(prototypes)  # (WAYS, 64)

    # 计算 query → 各原型的欧氏距离
    # query_emb: (WAYS*QUERY, 64) → (WAYS*QUERY, 1, 64)
    # prototypes: (WAYS, 64) → (1, WAYS, 64)
    # 差: (WAYS*QUERY, WAYS, 64) → 平方和 → (WAYS*QUERY, WAYS)
    dists = torch.cdist(query_emb.unsqueeze(1), prototypes.unsqueeze(0)).squeeze(1)  # (Nq, WAYS)

    # 负距离做 Softmax → CrossEntropy
    loss = nn.functional.cross_entropy(-dists, query_y.to(device))

    # 预测准确率
    _, preds = torch.min(dists, dim=1)
    acc = (preds == query_y.to(device)).float().mean().item()

    return loss, acc


# ──────── 5. 训练 ────────

def train():
    print(f"设备: {DEVICE}")
    print("=" * 50)
    print("原型网络 Prototypical Networks — 小样本故障诊断")
    print(f"配置: {WAYS}-way {SHOT}-shot, {EPISODES} episodes")
    print("=" * 50)

    # 数据（step2 已按时间顺序预分割，无泄漏）
    train_dataset = CWRUDataset(INPUT_FILE, split="train")
    val_dataset = CWRUDataset(INPUT_FILE, split="test")
    _, train_class_map = get_all_indices(train_dataset)
    _, val_class_map = get_all_indices(val_dataset)

    train_idx = sum(train_class_map.values(), [])
    val_idx = sum(val_class_map.values(), [])
    print(f"训练样本: {len(train_idx)} | 验证样本: {len(val_idx)}")

    # 模型
    encoder = Encoder().to(DEVICE)
    encoder = load_pretrained_encoder(encoder, "cnn1d_model.pth")
    optimizer = optim.Adam(encoder.parameters(), lr=LR)

    # 采样器
    train_sampler = EpisodicSampler(train_dataset, train_idx, WAYS, SHOT, QUERY)
    val_sampler = EpisodicSampler(val_dataset, val_idx, WAYS, SHOT, QUERY)

    # ──── 训练循环 ────
    print("\n开始训练...")
    best_val_acc = 0.0
    log_interval = 100

    for ep in range(1, EPISODES + 1):
        encoder.train()
        s_x, s_y, q_x, q_y = train_sampler.sample_episode()
        loss, train_acc = prototypical_loss(encoder, s_x, s_y, q_x, q_y, DEVICE)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 验证
        if ep % log_interval == 0 or ep == 1:
            encoder.eval()
            val_accs = []
            with torch.no_grad():
                for _ in range(VAL_EPISODES):
                    s_x, s_y, q_x, q_y = val_sampler.sample_episode()
                    _, acc = prototypical_loss(encoder, s_x, s_y, q_x, q_y, DEVICE)
                    val_accs.append(acc)

            mean_val_acc = np.mean(val_accs) * 100
            best_val_acc = max(best_val_acc, mean_val_acc)

            print(f"Episode {ep:4d}/{EPISODES} | "
                  f"Train Loss: {loss.item():.4f} | "
                  f"Train Acc: {train_acc*100:.1f}% | "
                  f"Val Acc: {mean_val_acc:.1f}% (best: {best_val_acc:.1f}%)")

    # ──── 最终评估 ────
    print(f"\n{'='*50}")
    print(f"✅ 训练完成！最高验证准确率: {best_val_acc:.1f}%")
    print(f"{'='*50}")

    # ──── 保存小样本模型 ────
    torch.save(encoder.state_dict(), "prototypical_encoder.pth")
    print("✅ 小样本编码器已保存到 prototypical_encoder.pth")

    # ──── 小样本总结 ────
    print(f"\n{'='*50}")
    print("📊 结果解读")
    print(f"  场景: {WAYS}-way {SHOT}-shot 小样本分类")
    print(f"  每类训练样本数: {SHOT}")
    print(f"  验证准确率: {best_val_acc:.1f}%")
    if best_val_acc >= 80:
        print("  ✅ 达标！超过 80% 目标")
    else:
        print("  ⚠️ 未达 80% 目标，可尝试:")
        print("     - 增加训练 episode 数")
        print("     - 调节学习率")
        print("     - 增加数据增强（加噪声/平移）")
    print(f"{'='*50}")


if __name__ == "__main__":
    train()
