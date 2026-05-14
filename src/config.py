"""
项目全局配置：手势类别、MediaPipe 关键点常量、关节连接关系。

所有"魔法数字"都集中在这里，方便后期调整。
"""
from pathlib import Path

# ============================================================
# 路径
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"          # HaGRIDv2 标注 JSON 存放处
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"  # 提取后的特征向量
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
ARTIFACT_FIGURES_DIR = ARTIFACTS_DIR / "figures"
REPORT_DIR = PROJECT_ROOT / "report"
REPORT_FIGURES_DIR = REPORT_DIR / "figures"

# ============================================================
# 目标手势（与 HaGRIDv2 类别名严格对应）
# ============================================================
# HaGRIDv2 中对应的类别名。当前只使用论文范围内的 10 类常见单手手势；
# 不包含 no_gesture、双手类或 v2 新增复杂类。
TARGET_GESTURES = [
    "fist",
    "four",
    "ok",
    "one",
    "palm",
    "peace",
    "three",
    "stop",
    "call",
    "rock",
]

# Display label mapping used by generated figures and reports.
# Project report artifacts are English-only.
GESTURE_CN = {
    "call": "Call",
    "fist": "Fist",
    "four": "Four",
    "ok": "OK",
    "one": "One",
    "palm": "Palm",
    "peace": "Peace",
    "rock": "Rock",
    "stop": "Stop",
    "three": "Three",
}

# ============================================================
# MediaPipe 21 个关键点常量
# 索引参考：https://developers.google.com/mediapipe/solutions/vision/hand_landmarker
# ============================================================
WRIST = 0
# 拇指 (Thumb): CMC->MCP->IP->TIP
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
# 食指 (Index)
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
# 中指 (Middle)
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
# 无名指 (Ring)
RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
# 小指 (Pinky)
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20

# 用于尺度归一化的"参考骨长"：手腕 -> 中指 MCP
# 选这条是因为它最稳定（位于手掌中心，不受手指弯曲影响）
SCALE_REF_FROM = WRIST
SCALE_REF_TO = MIDDLE_MCP

# ============================================================
# 关节角度计算的三元组
# 每个三元组 (a, b, c) 表示：以 b 为顶点，向量 b->a 和 b->c 的夹角
# 这正好对应一个解剖学上的关节屈曲角度
# ============================================================
JOINT_TRIPLETS = {
    # 拇指：CMC, MCP, IP 三个关节
    "thumb_cmc":  (WRIST,      THUMB_CMC,  THUMB_MCP),
    "thumb_mcp":  (THUMB_CMC,  THUMB_MCP,  THUMB_IP),
    "thumb_ip":   (THUMB_MCP,  THUMB_IP,   THUMB_TIP),
    # 食指
    "index_mcp":  (WRIST,      INDEX_MCP,  INDEX_PIP),
    "index_pip":  (INDEX_MCP,  INDEX_PIP,  INDEX_DIP),
    "index_dip":  (INDEX_PIP,  INDEX_DIP,  INDEX_TIP),
    # 中指
    "middle_mcp": (WRIST,      MIDDLE_MCP, MIDDLE_PIP),
    "middle_pip": (MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP),
    "middle_dip": (MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP),
    # 无名指
    "ring_mcp":   (WRIST,      RING_MCP,   RING_PIP),
    "ring_pip":   (RING_MCP,   RING_PIP,   RING_DIP),
    "ring_dip":   (RING_PIP,   RING_DIP,   RING_TIP),
    # 小指
    "pinky_mcp":  (WRIST,      PINKY_MCP,  PINKY_PIP),
    "pinky_pip":  (PINKY_MCP,  PINKY_PIP,  PINKY_DIP),
    "pinky_dip":  (PINKY_PIP,  PINKY_DIP,  PINKY_TIP),
}
JOINT_NAMES = list(JOINT_TRIPLETS.keys())  # 15 个关节角度

# ============================================================
# 第三层特征：指间几何关系
# 指尖之间的归一化距离（已用参考骨长归一化过）
# ============================================================
TIP_PAIRS = {
    "thumb_index_tip":  (THUMB_TIP, INDEX_TIP),    # OK 手势的关键
    "thumb_middle_tip": (THUMB_TIP, MIDDLE_TIP),
    "thumb_ring_tip":   (THUMB_TIP, RING_TIP),
    "thumb_pinky_tip":  (THUMB_TIP, PINKY_TIP),
    "index_middle_tip": (INDEX_TIP, MIDDLE_TIP),   # 剪刀手张开度
    "middle_ring_tip":  (MIDDLE_TIP, RING_TIP),
    "ring_pinky_tip":   (RING_TIP, PINKY_TIP),
}

# 相邻手指的"展开角"：以手腕为顶点，看相邻指尖的张角
SPREAD_TRIPLETS = {
    "spread_thumb_index":  (THUMB_TIP, WRIST, INDEX_TIP),
    "spread_index_middle": (INDEX_TIP, WRIST, MIDDLE_TIP),
    "spread_middle_ring":  (MIDDLE_TIP, WRIST, RING_TIP),
    "spread_ring_pinky":   (RING_TIP, WRIST, PINKY_TIP),
}

# ============================================================
# 特征向量维度自检（运行时验证）
# ============================================================
N_LANDMARKS = 21
N_JOINT_ANGLES = len(JOINT_TRIPLETS)        # 15
N_TIP_DISTANCES = len(TIP_PAIRS)            # 7
N_SPREAD_ANGLES = len(SPREAD_TRIPLETS)      # 4
# 第二层 + 第三层 = 26 维（这是默认的"几何特征向量"维度）
N_GEOMETRIC_FEATURES = N_JOINT_ANGLES + N_TIP_DISTANCES + N_SPREAD_ANGLES

# ============================================================
# 数据过滤阈值
# ============================================================
# HaGRIDv2 中部分样本 landmarks 全为 0（MediaPipe 检测失败），需要过滤
MIN_VALID_LANDMARK_RATIO = 0.95  # 至少 95% 的关键点不能是 (0,0)
