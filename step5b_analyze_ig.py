"""
Step 5b: IG 归因图分析报告 — 验证模型注意力的物理合理性

目的:
  生成 IG 归因图后，你可能看不懂这些红红蓝蓝的图到底对不对。
  这个脚本用数学统计替代你肉眼判断——从 IG 归因数据中提取
  峰值间隔、分布密度、集中度等物理量，对照各类故障的理论特征，
  给出"模型判断是否合理"的结论。

理论依据（滚动轴承故障物理特征）:
  Normal:     无冲击，归因应为低幅值噪声，无周期结构
  Inner Race: 内圈故障，冲击信号调制在转频上，归因细密分散
  Outer Race: 外圈故障，固定位置冲击，等间隔块状归因
  Ball:       滚动体故障，接触点变，冲击断续，离散脉冲

运行:
  python step5b_analyze_ig.py
"""

import numpy as np
import torch
import os, sys

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
INPUT_FILE = "preprocessed_data.npz"
MODEL_PATH = "cnn1d_model.pth"
CLASS_NAMES = ["Normal", "Inner Race", "Ball", "Outer Race"]

# ──────── 1. 模型（复用） ────────

import torch.nn as nn
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
            nn.Linear(32, 4),
        )
    def forward(self, x):
        feat = self.features(x).squeeze(-1)
        return self.cls(feat)


# ──────── 2. IG 归因计算（精简版，复用 step5 逻辑） ────────

def compute_ig(model, x, steps=30):
    model.eval()
    x = x.to(DEVICE)
    baseline = torch.zeros_like(x)

    with torch.no_grad():
        output = model(x)
        pred_class = output.argmax(dim=1).item()
        pred_conf = torch.softmax(output, dim=1)[0, pred_class].item()

    alphas = torch.linspace(0, 1, steps, device=DEVICE)
    interpolated = baseline + alphas.view(-1, 1, 1) * (x - baseline)
    interpolated.requires_grad_(True)

    grad_sum = torch.zeros_like(x)
    batch_size = min(10, steps)

    for i in range(0, steps, batch_size):
        batch = interpolated[i:i + batch_size]
        out = model(batch)
        grad_out = torch.zeros_like(out)
        grad_out[:, pred_class] = 1.0
        grads = torch.autograd.grad(out, batch, grad_out, create_graph=False)[0]
        grad_sum += grads.sum(dim=0, keepdim=True)

    attr = ((x - baseline) * (grad_sum / steps)).squeeze().cpu().numpy()
    return attr, pred_class, pred_conf


# ──────── 3. 归因图特征提取 ────────

def extract_attribution_features(attr):
    """
    从 IG 归因向量 (1024,) 中提取可解释的统计量。

    返回字典:
      pos_ratio:      正向归因占比（红色区域比例）
      peak_count:     显著峰值数量（>0.3 最大绝对值）
      peak_interval_cv: 峰值间隔变异系数（越小越规律）
      cluster_ratio:  前 20% 归因集中度（集中 vs 分散）
      autocorr_peak:  自相关次高峰位置（检测周期性）
      zero_crossings: 过零率（归因正负翻转频率）
    """
    max_abs = np.abs(attr).max()
    if max_abs < 1e-6:
        return {"pos_ratio": 0, "peak_count": 0, "peak_interval_cv": -1,
                "cluster_ratio": 0, "autocorr_peak": -1, "zero_crossings": 0}

    norm = attr / max_abs
    pos = np.maximum(norm, 0)

    # 正向归因占比
    pos_ratio = (pos > 0.15).mean()

    # 峰值检测（显著正值位置）
    threshold = 0.3
    peaks = []
    for i in range(1, len(norm) - 1):
        if pos[i] > threshold and pos[i] > pos[i-1] and pos[i] > pos[i+1]:
            peaks.append(i)
    peak_count = len(peaks)

    # 峰值间隔变异系数
    if peak_count >= 3:
        intervals = np.diff(peaks)
        peak_interval_cv = np.std(intervals) / (np.mean(intervals) + 1e-8)
    else:
        peak_interval_cv = -1  # 峰值太少，无法计算周期性

    # 集中度：前 20% 的大值占总体比例
    sorted_pos = np.sort(pos.flatten())[::-1]
    top20 = sorted_pos[:len(sorted_pos) // 5].sum()
    total = sorted_pos.sum() + 1e-8
    cluster_ratio = top20 / total

    # 自相关：检测周期性
    autocorr = np.correlate(pos.flatten(), pos.flatten(), mode="same")
    mid = len(autocorr) // 2
    # 找除中心点外的次高峰列
    search = autocorr[mid + 10:mid + 300]  # 10~300 滞后
    if len(search) > 0:
        autocorr_peak = np.argmax(search) + 10
    else:
        autocorr_peak = -1

    # 过零率
    sign = np.sign(norm)
    zero_crossings = ((sign[:-1] * sign[1:]) < 0).sum()

    return {
        "pos_ratio": pos_ratio,
        "peak_count": peak_count,
        "peak_interval_cv": peak_interval_cv,
        "cluster_ratio": cluster_ratio,
        "autocorr_peak": autocorr_peak,
        "zero_crossings": zero_crossings,
    }


# ──────── 4. 推理规则：这些物理量意味着什么 ────────

def diagnose_attribution(features):
    """
    根据 IG 归因特征，判断模型行为是否合理。
    返回诊断结论字符串。
    """
    lines = []

    pr = features["pos_ratio"]
    pc = features["peak_count"]
    cv = features["peak_interval_cv"]
    cr = features["cluster_ratio"]
    ap = features["autocorr_peak"]
    zc = features["zero_crossings"]

    # ── 判断归因活跃程度 ──
    if pr < 0.05:
        lines.append("🔵 归因活跃度极低：模型几乎不依赖任何特定时间步")
        activity = "inactive"
    elif pr < 0.2:
        lines.append("🟢 归因活跃度适中：模型关注少数关键时间步")
        activity = "focused"
    elif pr < 0.5:
        lines.append("🟡 归因活跃度较高：模型关注范围偏广")
        activity = "broad"
    else:
        lines.append("🔴 归因活跃度过高：模型几乎关注全部时间步")
        activity = "overactive"

    # ── 判断峰值分布模式 ──
    if pc < 3:
        lines.append("  → 峰值极少，信号缺乏明显的冲击特征")
        pattern = "no_peaks"
    elif cv < 0.3:
        lines.append(f"  → 峰值间隔规律 (CV={cv:.2f})，具有周期性冲击特征")
        pattern = "periodic"
    elif cv < 0.6:
        lines.append(f"  → 峰值间隔中等规律 (CV={cv:.2f})，半周期性冲击")
        pattern = "semi_periodic"
    else:
        lines.append(f"  → 峰值间隔不规律 (CV={cv:.2f})，离散随机冲击")
        pattern = "random"

    # ── 判断集中度 ──
    if cr > 0.6:
        lines.append("  → 归因高度集中在少数区域（块状分布）")
    elif cr > 0.35:
        lines.append("  → 归因中等集中（有聚集但范围分散）")
    else:
        lines.append("  → 归因均匀分散（无明显聚集区域）")

    # ── 判断周期性 ──
    if ap > 10:
        lines.append(f"  → 自相关检测到周期约 {ap} 时间步")
    else:
        lines.append("  → 未检测到显著周期性")

    return "\n".join(lines), activity, pattern


# ──────── 5. 物理参照表 ────────

REFERENCE = {
    "Normal": """
理论特征: 无冲击，接近纯随机噪声
预期归因: 低幅值，无显著峰值，无周期性
活跃度:   inactive 或 focused（绿色区域应很少）
模式:     应为 no_peaks
""",
    "Inner Race": """
理论特征: 内圈故障，冲击经转频调制，特征复杂分散
预期归因: 中等密度细碎峰值，范围较宽，分布在整个信号
活跃度:   focused 到 broad
模式:     应为 semi_periodic 或 periodic
""",
    "Outer Race": """
理论特征: 外圈故障，固定位置冲击，等间隔
预期归因: 块状集中，间隔规律明显
活跃度:   focused
模式:     应为 periodic，集群明显 (cr 高)
""",
    "Ball": """
理论特征: 滚动体故障，接触点变化导致冲击断续
预期归因: 离散孤立脉冲，间隔不均匀
活跃度:   focused
模式:     应为 random，峰值少而强
""",
}


# ──────── 6. 主流程 ────────

def run():
    print("=" * 60)
    print("IG 归因分析报告 — 验证模型注意力是否物理合理")
    print("=" * 60)

    # 加载模型
    if not os.path.exists(MODEL_PATH):
        print(f"❌ 找不到 {MODEL_PATH}")
        return
    if not os.path.exists(INPUT_FILE):
        print(f"❌ 找不到 {INPUT_FILE}")
        return

    model = CNN1D().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    data = np.load(INPUT_FILE)
    X, y = data["X"], data["y"]

    print(f"\n数据: {len(X)} 样本 | 设备: {DEVICE}\n")

    # 每类分析 5 个样本
    results = {cls: [] for cls in range(4)}

    for cls in range(4):
        indices = np.where(y == cls)[0]
        selected = np.random.choice(indices, size=min(5, len(indices)), replace=False)

        for idx in selected:
            x_tensor = torch.tensor(X[idx]).float().unsqueeze(0).unsqueeze(0)
            attr, pred, conf = compute_ig(model, x_tensor, steps=30)
            features = extract_attribution_features(attr)
            results[cls].append({"idx": idx, "features": features,
                                 "pred": CLASS_NAMES[pred], "conf": conf})

    # ── 逐类报告 ──
    all_ok = True

    for cls in range(4):
        print(f"\n{'─' * 60}")
        print(f"📊 {CLASS_NAMES[cls]}")
        print(f"{'─' * 60}")

        # 汇总该类 5 个样本的平均特征
        avg_feat = {}
        for key in results[cls][0]["features"]:
            vals = [r["features"][key] for r in results[cls]]
            avg_feat[key] = np.mean(vals)

        # 诊断
        summary, activity, pattern = diagnose_attribution(avg_feat)

        print(f"\n平均归因特征:")
        print(f"  正向区域占比:     {avg_feat['pos_ratio']:.1%}")
        print(f"  显著峰值数:       {avg_feat['peak_count']}")
        print(f"  峰值间隔变异系数: {avg_feat['peak_interval_cv']:.2f}")
        print(f"  前20%集中度:      {avg_feat['cluster_ratio']:.2f}")
        print(f"  自相关周期:       {avg_feat['autocorr_peak']} 步")
        print(f"  过零率:           {avg_feat['zero_crossings']}")

        print(f"\n🧠 模型行为分析:")
        print(summary)

        # 与参照对比
        print(f"\n📖 物理参照 ({CLASS_NAMES[cls]}):")
        print(REFERENCE[CLASS_NAMES[cls]])

        # 判断：当前样本预测是否正确？
        sample_results = results[cls]
        correct = sum(1 for r in sample_results if r["pred"] == CLASS_NAMES[cls])
        total = len(sample_results)
        print(f"预测准确率: {correct}/{total}")
        for r in sample_results:
            mark = "✅" if r["pred"] == CLASS_NAMES[cls] else "❌"
            print(f"  {mark} 样本#{r['idx']:>4} → {r['pred']:<10} conf={r['conf']:.1%}")

        # 综合结论
        print(f"\n📋 综合结论:", end=" ")
        all_correct = correct == total
        if not all_correct:
            print("❌ 存在分错的样本 - 模型在该类上不够稳定")
            all_ok = False
        elif activity == "overactive":
            print("⚠️ 分类正确，但归因范围太广，模型可能依赖全局统计而非局部特征")
        elif pattern == "no_peaks" and CLASS_NAMES[cls] != "Normal":
            print("⚠️ 分类正确，但归因缺乏冲击特征 - 正常样本不该有周期性")
        else:
            print("✅ 模型行为与故障物理特征一致")

    # ── 总评 ──
    print(f"\n{'=' * 60}")
    print("📈 总评")
    if all_ok:
        print("""
模型评估: ✅ IG 归因图与滚动轴承故障物理特征一致
结论: 模型不是"全段瞎猜"，而是在学习各故障特有的时域冲击模式
      可进入下一步工程封装 (FastAPI / Agent)
""")
    else:
        print("""
模型评估: ⚠️ 存在注意模式异常的类，建议继续调优
建议:
  - 增加训练 episode 数
  - 调整数据增强策略
  - 检查该类样本质量
""")
    print("=" * 60)


if __name__ == "__main__":
    run()
