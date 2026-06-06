"""
Step 5: Integrated Gradients — 模型可解释性

为什么不用 Grad-CAM:
  CNN1D 结构含 AdaptiveAvgPool1d(1)，所有时间步被压平为单一值，
  梯度均分导致 Grad-CAM 热力图无法定位。换用 Integrated Gradients，
  直接在输入空间做归因，不受 GAP 限制。

Integrated Gradients 原理:
  1. 选择基线（全零信号）
  2. 在基线和输入之间线性插 N 步
  3. 对每一步计算梯度
  4. 累加梯度 × (输入 - 基线) / N

用法:
  python step5_integrated_gradients.py
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import os, argparse, sys

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
INPUT_FILE = "preprocessed_data.npz"
MODEL_PATH = "cnn1d_model.pth"
OUTPUT_DIR = "gradcam_outputs"
CLASS_NAMES = ["Normal", "Inner Race", "Ball", "Outer Race"]

os.makedirs(OUTPUT_DIR, exist_ok=True)

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


# ──────── 2. Integrated Gradients ────────

def integrated_gradients(model, x, target_class=None, steps=30):
    """
    Integrated Gradients 归因。

    参数:
      model: CNN1D
      x: (1, 1, 1024) 输入信号
      target_class: 目标类别（None=用预测类别）
      steps: 插值步数（越多越精确，50 是速度和精度的平衡）

    返回:
      attr: (1024,) 归因分数，正值表示该位置促进预测该类
      pred_class, pred_conf
    """
    model.eval()
    x = x.to(DEVICE)
    baseline = torch.zeros_like(x)

    # 预测类别
    with torch.no_grad():
        output = model(x)
        probs = torch.softmax(output, dim=1)
        pred_class = output.argmax(dim=1).item() if target_class is None else target_class
        pred_conf = probs[0, pred_class].item()

    # 线性插值: 从 baseline 到 x，均匀取 steps 个点
    alphas = torch.linspace(0, 1, steps, device=DEVICE)
    # alphas: (steps,) → (steps, 1, 1)  ×  x: (1, 1, 1024)  →  (steps, 1, 1024)
    interpolated = baseline + alphas.view(-1, 1, 1) * (x - baseline)
    interpolated.requires_grad_(True)

    batch_size = min(10, steps)
    grad_sum = torch.zeros_like(x)

    for i in range(0, steps, batch_size):
        batch = interpolated[i:i + batch_size]
        out = model(batch)

        # 对目标类别的得分求梯度
        grad_outputs = torch.zeros_like(out)
        grad_outputs[:, pred_class] = 1.0

        grads = torch.autograd.grad(
            outputs=out,
            inputs=batch,
            grad_outputs=grad_outputs,
            create_graph=False,
            retain_graph=False,
        )[0]  # (batch, 1, 1024)

        grad_sum += grads.sum(dim=0, keepdim=True)

    # 归因 = (x - baseline) × 平均梯度
    attr = (x - baseline) * (grad_sum / steps)
    attr = attr.squeeze().cpu().numpy()  # (1024,)

    return attr, pred_class, pred_conf


# ──────── 3. 可视化 ────────

def plot_single(signal, attribution, pred_class, conf,
                true_class=None, save_path=None):
    """
    上：原始信号
    下：原始信号 + 归因热力图（正值红色，负值蓝色）
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 6), sharex=True)
    x_vals = np.arange(len(signal))

    # ── 上：原始信号 ──
    ax1.plot(x_vals, signal, color="#2c3e50", linewidth=0.8)
    ax1.set_ylabel("Amplitude")
    label = CLASS_NAMES[true_class] if true_class is not None else "?"
    ax1.set_title(f"True: {label}")
    ax1.grid(alpha=0.25)

    # ── 下：信号 + 归因叠加 ──
    # 归因绝对值归一化，方便着色
    max_abs = np.abs(attribution).max()
    if max_abs > 1e-6:
        norm_attr = attribution / max_abs  # [-1, 1]
    else:
        norm_attr = np.zeros_like(attribution)

    # 用 imshow 铺归因图
    attr_2d = norm_attr[np.newaxis, :]  # (1, 1024)
    ax2.imshow(attr_2d, aspect="auto",
               extent=[0, len(signal), signal.min(), signal.max()],
               cmap="RdBu_r", alpha=0.55,
               vmin=-1, vmax=1)
    # 信号线
    ax2.plot(x_vals, signal, color="#2c3e50", linewidth=0.6, alpha=0.6)

    pred_label = CLASS_NAMES[pred_class] if pred_class < len(CLASS_NAMES) else "?"
    ax2.set_title(f"Pred: {pred_label}  (conf={conf:.1%})")
    ax2.set_xlabel("Time Step")
    ax2.set_ylabel("Amplitude")
    ax2.grid(alpha=0.25)

    fig.suptitle("Integrated Gradients - Input Attribution",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  -> saved: {save_path}")
    plt.close()


def plot_comparison(signals, attributions, preds, confs, save_path=None):
    n = len(signals)
    fig, axes = plt.subplots(n, 2, figsize=(18, 3 * n), sharex="col")

    for i in range(n):
        x_vals = np.arange(len(signals[i]))

        # 左：原始信号
        axes[i, 0].plot(signals[i], color="#2c3e50", linewidth=0.7)
        axes[i, 0].set_ylabel("Amplitude")
        axes[i, 0].set_title(f"{CLASS_NAMES[preds[i]]}")
        axes[i, 0].grid(alpha=0.25)

        # 右：归因图
        max_abs = np.abs(attributions[i]).max()
        norm = attributions[i] / max_abs if max_abs > 1e-6 else np.zeros_like(attributions[i])
        attr_2d = norm[np.newaxis, :]
        axes[i, 1].imshow(attr_2d, aspect="auto",
                          extent=[0, len(signals[i]), signals[i].min(), signals[i].max()],
                          cmap="RdBu_r", alpha=0.55, vmin=-1, vmax=1)
        axes[i, 1].plot(x_vals, signals[i], color="#2c3e50", linewidth=0.5, alpha=0.5)
        axes[i, 1].set_title(f"Attribution  conf={confs[i]:.1%}")
        axes[i, 1].grid(alpha=0.25)

    fig.suptitle("Integrated Gradients - Comparison",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  -> comparison saved: {save_path}")
    plt.close()


# ──────── 4. 主流程 ────────

def run():
    import time
    start_time = time.time()
    if not os.path.exists(MODEL_PATH):
        print(f"Model not found: {MODEL_PATH}")
        return
    if not os.path.exists(INPUT_FILE):
        print(f"Data not found: {INPUT_FILE}")
        return

    model = CNN1D().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    data = np.load(INPUT_FILE)
    X, y = data["X"], data["y"]
    print(f"Data: {len(X)} samples")

    signals, attrs, preds, confs, trues = [], [], [], [], []

    for cls in range(4):
        indices = np.where(y == cls)[0]
        selected = np.random.choice(indices, size=min(3, len(indices)), replace=False)

        for idx in selected:
            x_tensor = torch.tensor(X[idx]).float().unsqueeze(0).unsqueeze(0)

            print(f"  Computing IG for {CLASS_NAMES[cls]} sample {idx}...", end=" ")
            sys.stdout.flush()
            attr, pred, conf = integrated_gradients(model, x_tensor, steps=30)
            elapsed = time.time() - start_time
            print(f"pred={CLASS_NAMES[pred]} conf={conf:.1%}  ({elapsed:.0f}s elapsed)")

            signals.append(X[idx])
            attrs.append(attr)
            preds.append(pred)
            confs.append(conf)
            trues.append(cls)

            fname = f"IG_{CLASS_NAMES[cls]}_to_{CLASS_NAMES[pred]}_{idx}.png"
            plot_single(X[idx], attr, pred, conf, true_class=cls,
                        save_path=os.path.join(OUTPUT_DIR, fname))

    # 对比图
    cls_seen = set()
    sel = []
    for i, t in enumerate(trues):
        if t not in cls_seen:
            cls_seen.add(t); sel.append(i)
        if len(cls_seen) == 4:
            break
    plot_comparison([signals[i] for i in sel],
                    [attrs[i] for i in sel],
                    [preds[i] for i in sel],
                    [confs[i] for i in sel],
                    save_path=os.path.join(OUTPUT_DIR, "IG_comparison.png"))

    print(f"\nDone. Results in {OUTPUT_DIR}/")


if __name__ == "__main__":
    print("=" * 55)
    print("Integrated Gradients - Input Attribution")
    print("=" * 55)
    run()
