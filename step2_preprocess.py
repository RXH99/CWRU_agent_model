"""
Step 2/3: 数据预处理（.mat → .npy 存盘，下次直接加载）
运行: python step2_preprocess.py
"""
import os, sys
import numpy as np
from scipy.io import loadmat

OUTPUT = "preprocessed_data.npz"
WINDOW_SIZE = 1024

# ── 新数据结构路径 ──
BASE_DIR = "data/CWRU data/12k Drive End Bearing Fault Data"
NORMAL_DIR = "data/CWRU data/Normal Baseline"

# {文件名: 标签}
FILES = {
    f"{NORMAL_DIR}/normal_0.mat": 0,                                    # Normal, 0hp
    f"{BASE_DIR}/Inner Race/0021/IR021_0.mat": 1,                        # Inner, 0hp
    f"{BASE_DIR}/Ball/0021/B021_0.mat": 2,                               # Ball, 0hp
    f"{BASE_DIR}/Outer Race/Centered/0021/OR021@6_0.mat": 3,             # Outer, 0hp
}
CLASS_NAMES = ["Normal", "Inner Race", "Ball", "Outer Race"]

def run():
    # 检查文件是否存在
    for path in FILES:
        if not os.path.exists(path):
            print(f"❌ 缺少 {os.path.basename(path)}")
            print(f"   预期路径: {path}")
            sys.exit(1)

    all_signals, all_labels = [], []
    test_signals, test_labels = [], []

    for path, label in FILES.items():
        mat = loadmat(path)
        # 选择 Drive End 变量（DE_time），没有则选第一个大数组
        keys = [k for k in mat.keys() if not k.startswith("__")]
        de_key = next((k for k in keys if "DE" in k.upper()), keys[0])
        signal = mat[de_key].flatten().astype(np.float32)
        print(f"  {os.path.basename(path)}: var={de_key}, 信号长度={len(signal)}")

        step = WINDOW_SIZE // 2
        windows = []
        for i in range(0, len(signal) - WINDOW_SIZE + 1, step):
            windows.append(signal[i : i + WINDOW_SIZE])

        # 按时间顺序分割：前 80% 训练，后 20% 测试
        # 防止相邻（重叠）窗口同时出现在训练和测试中
        split = int(len(windows) * 0.8)
        for w in windows[:split]:
            all_signals.append(w)
            all_labels.append(label)
        for w in windows[split:]:
            test_signals.append(w)
            test_labels.append(label)

    X_train = np.array(all_signals)
    y_train = np.array(all_labels, dtype=np.int64)
    X_test = np.array(test_signals)
    y_test = np.array(test_labels, dtype=np.int64)

    # 归一化（每段独立归一化）
    def normalize(X):
        mean = X.mean(axis=1, keepdims=True)
        std = X.std(axis=1, keepdims=True) + 1e-8
        return (X - mean) / std

    X_train = normalize(X_train)
    X_test = normalize(X_test)

    np.savez(OUTPUT, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
    print(f"\n✅ 预处理完成！")
    print(f"   训练: {len(X_train)} 样本 | 测试: {len(X_test)} 样本")
    print(f"   已保存到 {OUTPUT}")

if __name__ == "__main__":
    run()
