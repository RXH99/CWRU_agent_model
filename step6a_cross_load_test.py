"""
Step 6a: CWRU 跨负载泛化测试

用 0hp 训练的模型，直接预测 1hp/2hp/3hp 的数据。
如果准确率掉得厉害，说明模型记住了数据集特征而不是故障物理特征。

CWRU 文件命名规则:
  105~108: Normal        (0hp, 1hp, 2hp, 3hp)
  118~121: Inner Race    (0hp, 1hp, 2hp, 3hp)
  130~133: Ball          (0hp, 1hp, 2hp, 3hp)
  144~147: Outer Race    (0hp, 1hp, 2hp, 3hp)

运行: python step6a_cross_load_test.py
"""

import numpy as np
import torch
import torch.nn as nn
from scipy.io import loadmat
import os

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "cnn1d_model.pth"
WINDOW_SIZE = 1024
CLASS_NAMES = ["Normal", "Inner Race", "Ball", "Outer Race"]

# ──────── 新数据结构：路径基址 ────────
BASE_DIR = "data/CWRU data/12k Drive End Bearing Fault Data"
NORMAL_DIR = "data/CWRU data/Normal Baseline"

#  {class_name: {load_hp: 文件名}}
CWRU_FILES = {
    "Normal":     {0: "normal_0.mat", 1: "normal_1.mat", 2: "normal_2.mat", 3: "normal_3.mat"},
    "Inner Race": {0: "IR021_0.mat", 1: "IR021_1.mat", 2: "IR021_2.mat", 3: "IR021_3.mat"},
    "Ball":       {0: "B021_0.mat",  1: "B021_1.mat",  2: "B021_2.mat",  3: "B021_3.mat"},
    "Outer Race": {0: "OR021@6_0.mat", 1: "OR021@6_1.mat", 2: "OR021@6_2.mat", 3: "OR021@6_3.mat"},
}

# 每个类别对应的子目录
SUBDIRS = {
    "Normal":     NORMAL_DIR,
    "Inner Race": f"{BASE_DIR}/Inner Race/0021",
    "Ball":       f"{BASE_DIR}/Ball/0021",
    "Outer Race": f"{BASE_DIR}/Outer Race/Centered/0021",
}


# ──────── 1. 模型 ────────

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
        feat = self.features(x).squeeze(-1)
        return self.cls(feat)


# ──────── 2. 下载 + 读取 ────────

def ensure_data(_=None, test_hp=0):
    """检查指定负载的 4 个文件是否存在"""
    for cls_name in CLASS_NAMES:
        fname = CWRU_FILES[cls_name][test_hp]
        path = os.path.join(SUBDIRS[cls_name], fname)
        if not os.path.exists(path):
            print(f"  ⚠️ 缺失: {cls_name}/{fname}")
            return False
    print(f"  ✅ {test_hp}hp 全部 {len(CLASS_NAMES)} 个文件就绪")
    return True


def load_and_preprocess(cls_name, hp):
    """从新数据结构读取单个 .mat 文件，滑窗 + 归一化"""
    fname = CWRU_FILES[cls_name][hp]
    path = os.path.join(SUBDIRS[cls_name], fname)

    if not os.path.exists(path):
        return None

    mat = loadmat(path)
    keys = [k for k in mat.keys() if not k.startswith("__")]
    # 选 Drive End 变量（DE_time），否则选第一个大数组
    de_key = next((k for k in keys if "DE" in k.upper()), keys[0])
    signal = mat[de_key].flatten().astype(np.float32)

    segments = []
    step = WINDOW_SIZE // 2
    for i in range(0, len(signal) - WINDOW_SIZE + 1, step):
        seg = signal[i:i + WINDOW_SIZE]
        # 单段归一化（确保 float32）
        seg = ((seg - seg.mean()) / (seg.std() + 1e-8)).astype(np.float32)
        segments.append(seg)

    return np.array(segments)


# ──────── 3. 测试 ────────

def test_load(model, test_hp):
    """测试模型在指定负载上的表现。"""
    all_preds, all_labels = [], []
    per_class = {c: {"correct": 0, "total": 0} for c in CLASS_NAMES}

    for cls_idx, cls_name in enumerate(CLASS_NAMES):
        X = load_and_preprocess(cls_name, test_hp)
        if X is None:
            continue

        total = len(X)
        x_tensor = torch.tensor(X).unsqueeze(1).to(DEVICE)

        with torch.no_grad():
            outputs = model(x_tensor)
            _, preds = torch.max(outputs, 1)
            preds = preds.cpu().numpy()

        correct = (preds == cls_idx).sum()
        per_class[cls_name]["total"] = total
        per_class[cls_name]["correct"] = int(correct)
        all_preds.extend(preds.tolist())
        all_labels.extend([cls_idx] * total)

    total_all = sum(v["total"] for v in per_class.values())
    correct_all = sum(v["correct"] for v in per_class.values())
    overall_acc = correct_all / total_all * 100 if total_all else 0

    return overall_acc, per_class


# ──────── 4. 结果展示 ────────

def print_results(results):
    loads = sorted(results.keys())

    print(f"\n{'='*65}")
    print("CWRU 跨负载泛化测试结果")
    print(f"{'='*65}")
    print(f"{'负载':>6} | {'总体准确率':>10} | ", end="")
    for c in CLASS_NAMES:
        print(f"{c:>12}", end=" | ")
    print()
    print(f"{'─'*6}─┼─{'─'*10}─┼─", end="")
    for _ in CLASS_NAMES:
        print(f"{'─'*12}─┼", end="")
    print()

    for hp in loads:
        acc, per_class = results[hp]
        print(f"  {hp}hp  | {acc:>7.1f}%   | ", end="")
        for c in CLASS_NAMES:
            info = per_class[c]
            if info["total"] > 0:
                ca = info["correct"] / info["total"] * 100
                print(f"{ca:>5.1f}%({info['correct']}/{info['total']})", end="  | ")
            else:
                print(f"{' N/A':>12}", end=" | ")
        print()

    print(f"{'─'*6}─┴─{'─'*10}─┴─", end="")
    for _ in CLASS_NAMES:
        print(f"{'─'*12}─┴", end="")
    print()

    # 结论
    train_acc = results[0][0]
    test_accs = {hp: results[hp][0] for hp in loads if hp != 0}

    print(f"\n📊 泛化分析:")
    print(f"  训练集 (0hp):   {train_acc:.1f}%")
    for hp, acc in test_accs.items():
        drop = train_acc - acc
        if drop < 2:
            print(f"  {hp}hp (跨域):    {acc:.1f}%  (↓{drop:.1f}% ✅ 泛化能力强)")
        elif drop < 10:
            print(f"  {hp}hp (跨域):    {acc:.1f}%  (↓{drop:.1f}% ⚠️ 轻微下降)")
        else:
            print(f"  {hp}hp (跨域):    {acc:.1f}%  (↓{drop:.1f}% ❌ 显著下降，需要域自适应)")


# ──────── 主流程 ────────

def run():
    print("=" * 65)
    print("CWRU 跨负载泛化测试")
    print("模型: cnn1d_model.pth (0hp 训练)")
    print("测试: 0hp(训练集) / 1hp / 2hp / 3hp")
    print("=" * 65)

    # 检查模型
    if not os.path.exists(MODEL_PATH):
        print(f"❌ 找不到 {MODEL_PATH}")
        return

    model = CNN1D().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    print(f"✅ 加载模型 ({DEVICE})\n")

    # 检查 0hp 和 1hp
    print("\n检查数据文件...")
    results = {}
    for hp in [0, 1, 2, 3]:
        if not ensure_data(test_hp=hp):
            print(f"  {hp}hp: 跳过（文件不全）")
            continue
        print(f"\n测试 {hp}hp ...")
        acc, per_class = test_load(model, hp)
        results[hp] = (acc, per_class)

    if not results:
        print("\n❌ 没有可用的测试数据")
        return

    # 展示结果
    print_results(results)


if __name__ == "__main__":
    run()
