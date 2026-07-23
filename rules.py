# rules.py
import re
from typing import List, Dict, Optional
from models import HallucinationType

class RuleDetector:
    """基于关键词、正则、知识库比对的确定性规则检测器"""

    def __init__(self, knowledge_base: Optional[Dict] = None):
        # 敏感词（隐私泄露类）
        self.sensitive_keywords = {
            '身份证': HallucinationType.SAFETY_MISLEAD,
            '银行卡': HallucinationType.SAFETY_MISLEAD,
        }
        # 能力越界关键词（声称做了系统做不到的事）
        self.capability_action_words = ['我帮您查', '已帮您修改', '已升级', '直接发到您账户']
        # 虚构活动正则
        self.fake_promotion_patterns = [
            r'满\d+减\d+',          # 满300减50
            r'限时.*\d+折',
        ]
        # 知识库（传入真实数据）
        self.kb = knowledge_base or {}

    def detect(self, question: str, reply: str, history: List[str] = None) -> List[HallucinationType]:
        """返回规则命中的幻觉类型列表"""
        types = []

        # 1. 隐私/安全误导检测（敏感词）
        for kw, htype in self.sensitive_keywords.items():
            if kw in reply:
                types.append(htype)
                break

        # 2. 优惠/政策编造（匹配虚假活动）
        for pat in self.fake_promotion_patterns:
            if re.search(pat, reply):
                types.append(HallucinationType.POLICY_FABRICATION)
                break

        # 3. 能力越界（声称执行了非能力内动作）
        for word in self.capability_action_words:
            if word in reply:
                types.append(HallucinationType.CAPABILITY_OVERSTEP)
                break

        # 4. 参数编造（比对知识库中的产品参数）
        # 示例逻辑：若知识库有参数且回复不符，则标记
        if self.kb and 'product_params' in self.kb:
            for product, params in self.kb['product_params'].items():
                if product in question:
                    for key, val in params.items():
                        if key in reply and val not in reply:
                            types.append(HallucinationType.PARAM_FABRICATION)
                            break

        # 5. 信息编造 / 地址错误（简单规则：检测是否包含具体地址）
        if re.search(r'\d+号.*收', reply):  
            types.append(HallucinationType.INFO_FABRICATION)

        # 6. 逻辑矛盾（问A答B，关键词零重叠）
        action_words = ['退款', '退货', '换货', '投诉']
        if any(w in question for w in action_words):
            if not any(w in reply for w in action_words):
                types.append(HallucinationType.LOGIC_CONFLICT)

        # 去重返回
        return list(set(types))
