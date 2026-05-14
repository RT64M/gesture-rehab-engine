"""
端到端 Demo（无需真实数据集）

用合成的 5 种十类范围内手势关键点，演示整个 pipeline:
    生成合成关键点 -> 特征提取 -> 拟合 (μ, Σ) -> 马氏距离分类 -> 评分

跑通这个，说明代码 OK。下载真实 HaGRIDv2 后只需替换数据加载部分。
"""
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from src.features import batch_extract_geometric, feature_names, extract_features
from src.descriptors import build_descriptor, classify


def synth_hand(gesture: str, seed: int) -> np.ndarray:
    """根据手势类型生成合成的 21 个关键点（粗糙但拓扑正确）。"""
    rng = np.random.default_rng(seed)
    # 共用的手腕和手掌
    pts = np.zeros((21, 2))
    pts[0] = [0.5, 0.95]  # WRIST

    if gesture == "fist":
        # 所有手指都弯曲到手掌内
        pts[1:5]   = [[0.45, 0.88], [0.42, 0.82], [0.45, 0.80], [0.48, 0.82]]   # 拇指弯曲
        pts[5:9]   = [[0.46, 0.78], [0.45, 0.83], [0.46, 0.85], [0.47, 0.84]]   # 食指弯
        pts[9:13]  = [[0.50, 0.78], [0.50, 0.84], [0.51, 0.86], [0.51, 0.85]]   # 中指弯
        pts[13:17] = [[0.54, 0.78], [0.55, 0.84], [0.55, 0.86], [0.54, 0.85]]   # 无名指弯
        pts[17:21] = [[0.58, 0.78], [0.58, 0.83], [0.57, 0.85], [0.56, 0.84]]   # 小指弯
    elif gesture == "palm":
        # 所有手指都伸直
        pts[1:5]   = [[0.40, 0.85], [0.34, 0.78], [0.30, 0.72], [0.27, 0.66]]
        pts[5:9]   = [[0.45, 0.78], [0.43, 0.62], [0.42, 0.50], [0.41, 0.40]]
        pts[9:13]  = [[0.50, 0.77], [0.50, 0.58], [0.50, 0.45], [0.50, 0.34]]
        pts[13:17] = [[0.55, 0.78], [0.57, 0.60], [0.58, 0.48], [0.59, 0.38]]
        pts[17:21] = [[0.60, 0.80], [0.63, 0.66], [0.65, 0.56], [0.67, 0.48]]
    elif gesture == "ok":
        # 食指 + 拇指尖相碰，其他手指伸直
        pts[1:5]   = [[0.42, 0.80], [0.40, 0.70], [0.42, 0.60], [0.46, 0.55]]   # 拇指弯向食指
        pts[5:9]   = [[0.46, 0.78], [0.46, 0.65], [0.48, 0.58], [0.47, 0.55]]   # 食指弯回
        pts[9:13]  = [[0.50, 0.77], [0.50, 0.58], [0.50, 0.45], [0.50, 0.34]]   # 中无小伸直
        pts[13:17] = [[0.55, 0.78], [0.57, 0.60], [0.58, 0.48], [0.59, 0.38]]
        pts[17:21] = [[0.60, 0.80], [0.63, 0.66], [0.65, 0.56], [0.67, 0.48]]
    elif gesture == "peace":
        # 食指、中指伸直且分开；拇指、无名指、小指弯曲
        pts[1:5]   = [[0.45, 0.88], [0.42, 0.82], [0.45, 0.80], [0.48, 0.82]]   # 拇指弯
        pts[5:9]   = [[0.45, 0.78], [0.42, 0.62], [0.40, 0.50], [0.38, 0.40]]   # 食指伸直左偏
        pts[9:13]  = [[0.50, 0.77], [0.52, 0.58], [0.54, 0.45], [0.56, 0.34]]   # 中指伸直右偏
        pts[13:17] = [[0.55, 0.78], [0.55, 0.84], [0.55, 0.86], [0.54, 0.85]]   # 无名指弯
        pts[17:21] = [[0.60, 0.80], [0.58, 0.83], [0.57, 0.85], [0.56, 0.84]]   # 小指弯
    elif gesture == "rock":
        # 食指和小指伸直；中指、无名指弯曲
        pts[1:5]   = [[0.44, 0.86], [0.40, 0.80], [0.38, 0.74], [0.36, 0.68]]
        pts[5:9]   = [[0.45, 0.78], [0.42, 0.62], [0.40, 0.50], [0.38, 0.39]]
        pts[9:13]  = [[0.50, 0.78], [0.50, 0.84], [0.51, 0.86], [0.51, 0.85]]
        pts[13:17] = [[0.54, 0.78], [0.55, 0.84], [0.55, 0.86], [0.54, 0.85]]
        pts[17:21] = [[0.60, 0.80], [0.64, 0.66], [0.67, 0.56], [0.70, 0.47]]
    else:
        raise ValueError(gesture)

    # 加噪声（模拟同一手势的人际/同人差异）
    noise = rng.normal(0, 0.008, pts.shape)
    return pts + noise


def main():
    gestures = ["fist", "palm", "ok", "peace", "rock"]
    cn = {"fist": "握拳", "palm": "张开", "ok": "OK", "peace": "剪刀", "rock": "摇滚"}

    print("=" * 60)
    print("端到端 Demo（合成数据）")
    print("=" * 60)

    # 每个手势生成 200 个样本
    n_per_class = 200
    n_test = 40

    fnames = feature_names()
    print(f"特征维度: {len(fnames)}\n")

    train_feats, test_feats = {}, {}
    for g in gestures:
        landmarks = np.stack([synth_hand(g, seed=i) for i in range(n_per_class)])
        feats = batch_extract_geometric(landmarks, apply_pca_alignment=True)
        train_feats[g] = feats[n_test:]
        test_feats[g] = feats[:n_test]

    # 拟合描述符
    print("--- 拟合描述符 ---")
    descs = {}
    for g in gestures:
        d = build_descriptor(g, train_feats[g], fnames)
        descs[g] = d
        print(f"  {cn[g]:>3} (N={d.n_samples}): μ 范数={np.linalg.norm(d.mu):.2f}, "
              f"Σ 平均对角={np.diag(d.sigma).mean():.4f}")

    # 测试分类
    print("\n--- 测试集分类（马氏距离最近邻）---")
    confusion = np.zeros((5, 5), dtype=int)
    correct = 0
    total = 0
    for true_g in gestures:
        i = gestures.index(true_g)
        for x in test_feats[true_g]:
            pred_g, _ = classify(x, descs)
            j = gestures.index(pred_g)
            confusion[i, j] += 1
            if pred_g == true_g:
                correct += 1
            total += 1

    header = "真\\预  " + " ".join(f"{cn[g]:>5}" for g in gestures)
    print(f"\n  {header}")
    for i, g in enumerate(gestures):
        row = " ".join(f"{confusion[i, j]:>5d}" for j in range(5))
        print(f"  {cn[g]:>4} {row}")
    print(f"\n  整体准确率: {correct}/{total} = {correct/total*100:.1f}%")

    # 打分 demo: 一个完美的 fist 和一个变形的 fist
    print("\n--- 评分演示 ---")
    perfect_fist = synth_hand("fist", seed=999)
    distorted_fist = synth_hand("fist", seed=999)
    distorted_fist[8] = [0.40, 0.50]  # 食指尖故意伸出去

    fb_perfect = extract_features(perfect_fist)
    fb_distorted = extract_features(distorted_fist)

    for tau in [1.0, 3.0, 5.0]:
        s_perfect = descs["fist"].score(fb_perfect.geometric_vector, tau=tau)
        s_distorted = descs["fist"].score(fb_distorted.geometric_vector, tau=tau)
        print(f"  τ={tau}: 标准握拳 score={s_perfect:.3f}, 手指伸出的握拳 score={s_distorted:.3f}")

    print("\n  解读：τ 越小，评分越严格；变形的握拳得分明显低于标准握拳。")
    print("  在自适应阶段，每个用户的初始 τ 会从这个全局值开始，")
    print("  然后根据用户表现自动调整（这是动量更新模块的工作）。")


if __name__ == "__main__":
    main()
