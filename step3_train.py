"""
Step 3/3: 训练模型
运行: python step3_train.py
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import sys, os

# ──────── 配置 ────────
BATCH_SIZE = 64
EPOCHS = 30
LR = 0.001
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
INPUT_FILE = "preprocessed_data.npz"

# ──────── 数据加载 ────────
if not os.path.exists(INPUT_FILE):
    print(f"❌ 找不到 {INPUT_FILE}，请先运行 step2_preprocess.py")
    sys.exit(1)

data = np.load(INPUT_FILE)
X, y = data["X"], data["y"]

# 打乱 + 8:2 分割
idx = np.random.permutation(len(X))
X, y = X[idx], y[idx]
split = int(len(X) * 0.8)

class FaultDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X).unsqueeze(1)  # (N, 1, 1024)
        self.y = torch.tensor(y)
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_loader = DataLoader(FaultDataset(X[:split], y[:split]),
                          batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(FaultDataset(X[split:], y[split:]),
                        batch_size=BATCH_SIZE)

print(f"设备: {DEVICE}")
print(f"训练: {split} 样本 | 验证: {len(X)-split} 样本")

# ──────── 模型 ────────
class CNN1D(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 16, 15, 2, 7), nn.BatchNorm1d(16), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(16, 32, 7, 1, 3), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 3, 1, 1), nn.BatchNorm1d(64), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.cls = nn.Sequential(
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(32, num_classes),
        )
    def forward(self, x):
        return self.cls(self.features(x).squeeze(-1))

model = CNN1D().to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# ──────── 训练 ────────
best_acc = 0.0
for epoch in range(1, EPOCHS + 1):
    model.train()
    loss_sum = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(inputs), labels)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item()

    model.eval()
    correct = total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            _, preds = torch.max(model(inputs), 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

    acc = 100.0 * correct / total
    best_acc = max(best_acc, acc)
    print(f"Epoch {epoch:2d}/{EPOCHS} | Loss: {loss_sum/len(train_loader):.4f} | Val Acc: {acc:.2f}%")

print(f"\n✅ 训练完成！最高准确率: {best_acc:.2f}%")

# ──────── 保存模型 ────────
torch.save(model.state_dict(), "cnn1d_model.pth")
print(f"模型已保存到 cnn1d_model.pth")
