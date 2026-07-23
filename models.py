# models.py
from enum import Enum
from typing import List, Optional
from dataclasses import dataclass, field

class HallucinationType(Enum):
    """幻觉类型枚举（对应 Ground Truth 标签）"""
    NONE = "无幻觉"
    POLICY_FABRICATION = "政策编造"
    PARAM_FABRICATION = "参数编造"
    INFO_FABRICATION = "信息编造"
    CAPABILITY_OVERSTEP = "能力越界"
    SAFETY_MISLEAD = "安全误导"
    INFO_OMISSION = "信息遗漏"
    LOGIC_CONFLICT = "逻辑矛盾"
    OVER_INFERENCE = "过度推断"   # 扩展保留

# 严重程度映射（P0 最高）
SEVERITY_MAP = {
    HallucinationType.POLICY_FABRICATION: "P0",
    HallucinationType.PARAM_FABRICATION: "P0",
    HallucinationType.SAFETY_MISLEAD: "P0",
    HallucinationType.INFO_FABRICATION: "P1",
    HallucinationType.CAPABILITY_OVERSTEP: "P1",
    HallucinationType.LOGIC_CONFLICT: "P1",
    HallucinationType.INFO_OMISSION: "P2",
    HallucinationType.OVER_INFERENCE: "P2",
    HallucinationType.NONE: "P3",
}

@dataclass
class DetectionResult:
    """单条检测结果"""
    id: str
    user_question: str
    reply: str
    is_hallucination: bool
    detected_types: List[HallucinationType] = field(default_factory=list)
    severity: str = "P3"
    reason: str = ""
    ground_truth_types: List[str] = field(default_factory=list)  # 用于评估