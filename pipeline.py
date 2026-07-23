# pipeline.py
from typing import List, Optional
from models import DetectionResult, HallucinationType, SEVERITY_MAP
from rules import RuleDetector
from llm_judge import LLMJudge
from config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, LLM_MODEL

class DetectorPipeline:
    def __init__(self, use_llm: bool = False, use_mock: bool = True):
        self.rule_detector = RuleDetector()
        if use_llm:
            # 传入 DeepSeek 配置
            self.llm_judge = LLMJudge(
                use_mock=use_mock,
                api_key=DEEPSEEK_API_KEY,
                base_url=DEEPSEEK_BASE_URL,
                model=LLM_MODEL
            )
        else:
            self.llm_judge = None
        self.use_llm = use_llm

class DetectorPipeline:
    """两阶段检测流水线：规则初筛 + LLM 精判"""

    def __init__(self, use_llm: bool = False, use_mock: bool = True, api_key: str = None):
        self.rule_detector = RuleDetector()
        self.llm_judge = LLMJudge(use_mock=use_mock, api_key=api_key) if use_llm else None
        self.use_llm = use_llm

    def detect_single(self, item_id: str, question: str, reply: str,
                      knowledge_base: dict = None, history: List[str] = None) -> DetectionResult:
        # 若传入了知识库，更新规则器
        if knowledge_base:
            self.rule_detector.kb = knowledge_base

        # 第一阶段：规则检测
        rule_types = self.rule_detector.detect(question, reply, history)

        # 若规则命中，直接采纳（避免不必要的 LLM 调用）
        if rule_types:
            final_types = rule_types
        else:
            # 第二阶段：LLM 判定
            if self.use_llm and self.llm_judge:
                final_types = self.llm_judge.judge(question, reply, history)
            else:
                final_types = []

        final_types = list(set(final_types))
        is_hall = len(final_types) > 0
        severity = "P3"
        if is_hall:
            severity = min(SEVERITY_MAP[t] for t in final_types)

        reason = "，".join([t.value for t in final_types]) if final_types else "无幻觉"
        return DetectionResult(
            id=item_id,
            user_question=question,
            reply=reply,
            is_hallucination=is_hall,
            detected_types=final_types,
            severity=severity,
            reason=reason
        )