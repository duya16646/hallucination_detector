# llm_judge.py
import sys
import os
import random
import ast
from typing import List, Optional
from models import HallucinationType
from rules import RuleDetector
import config

# 强制 Python 使用 UTF-8 编码（解决 Windows 下编码问题）
if sys.platform == 'win32':
    sys.stdin.reconfigure(encoding='utf-8')
    sys.stdout.reconfigure(encoding='utf-8')
    os.environ['PYTHONIOENCODING'] = 'utf-8'
# 强制设置标准输出和标准错误的编码为 UTF-8（解决控制台打印乱码）
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
if sys.stderr.encoding != 'utf-8':
    sys.stderr.reconfigure(encoding='utf-8')


class LLMJudge:
    """DeepSeek 大模型判定器（兼容 OpenAI SDK）"""

    def __init__(self, use_mock: bool = True, api_key: str = None,
                 base_url: str = None, model: str = "deepseek-chat"):
        self.use_mock = use_mock
        self.api_key = api_key or config.DEEPSEEK_API_KEY
        self.base_url = base_url or config.DEEPSEEK_BASE_URL
        self.model = model or config.LLM_MODEL
        self.rule_detector = RuleDetector()
        self.client = None

        if not use_mock:
            try:
                import openai
                if not hasattr(openai, 'OpenAI'):
                    raise ImportError("OpenAI 版本过低，请升级到 >=1.0.0 (pip install openai --upgrade)")

                # 设置环境变量，确保 httpx 使用 UTF-8（部分系统需要）
                os.environ["PYTHONIOENCODING"] = "utf-8"

                self.client = openai.OpenAI(
                    api_key=self.api_key,
                    base_url=self.base_url,
                    # 显式指定超时等，可忽略
                )
                print(f"✅ DeepSeek 客户端已初始化，模型: {self.model}")
            except ImportError as e:
                print(f"⚠️ 未安装 openai 库或版本不兼容: {e}")
                print("   请运行: pip install openai --upgrade")
                print("   将自动切换至 Mock 模拟模式")
                self.use_mock = True
            except Exception as e:
                print(f"⚠️ 初始化失败: {e}，切换至 Mock 模式")
                self.use_mock = True

    def judge(self, question: str, reply: str, history: List[str] = None) -> List[HallucinationType]:
        if self.use_mock or self.client is None:
            return self._mock_judge(question, reply, history)
        return self._llm_judge(question, reply, history)

    def _mock_judge(self, question: str, reply: str, history: List[str] = None) -> List[HallucinationType]:
        """模拟模式：基于规则 + 随机补充（用于降级或演示）"""
        rule_types = self.rule_detector.detect(question, reply, history)
        all_types = list(HallucinationType)
        extra = random.sample([t for t in all_types if t not in [HallucinationType.NONE]], k=random.randint(0, 1))
        return list(set(rule_types + extra))

    def _llm_judge(self, question: str, reply: str, history: List[str] = None) -> List[HallucinationType]:
        """调用 DeepSeek API 进行深度判定"""
        # 确保所有字符串都是 UTF-8 编码的（Python 3 默认就是 unicode，但显式转换一下更安全）
        # 但这里实际上是 unicode，无需 encode，只是为了避免可能的隐式转换问题
        question = str(question) if question else ""
        reply = str(reply) if reply else ""
        history_str = str(history) if history else "无"

        prompt = f"""
你是一个严格的智能客服幻觉检测专家。请根据【用户问题】和【客服回复】，判断回复是否包含以下任何一种幻觉类型：

1. 政策编造：编造或歪曲退货、优惠、发票规则（如将7天说成30天）
2. 参数编造：杜撰产品参数（蓝牙版本、材质、接口等）
3. 信息编造：虚构线下门店、品牌关联、具体地址等不存在的事实
4. 能力越界：声称具备系统实际没有的查询/修改能力（如查物流、改地址）
5. 安全误导：对孕妇等特殊人群给出错误的安全指导
6. 信息遗漏：忽略用户评价中关键的统计数据（如尺码偏大）
7. 逻辑矛盾：回复与问题完全无关或自相矛盾

用户问题：{question}
客服回复：{reply}
知识库参考（若有）：{history_str}

请务必只输出一个 JSON 列表，如 ["政策编造", "参数编造"]。如果没有任何幻觉，输出空列表 []。
"""
        try:
            # 调用前打印请求内容（可选），但可能包含中文，确保控制台支持 UTF-8
            # print(f"请求内容: {prompt[:100]}...")  # 可以取消注释调试
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "你是一个严谨的幻觉检测助手，只输出 JSON 列表。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                response_format={"type": "json_object"}  # DeepSeek 支持 JSON 模式
            )

            raw_content = response.choices[0].message.content.strip()
            # 去除可能的 Markdown 标记
            if raw_content.startswith("```json"):
                raw_content = raw_content[7:-3]
            elif raw_content.startswith("```"):
                raw_content = raw_content[3:-3]

            type_names = ast.literal_eval(raw_content)
            if not isinstance(type_names, list):
                type_names = []

            result = []
            for name in type_names:
                for t in HallucinationType:
                    if t.value == name:
                        result.append(t)
                        break
            return result

        except Exception as e:
            print(f"❌ DeepSeek API 调用失败: {e}")
            # 为了调试，打印错误详情
            import traceback
            traceback.print_exc()
            print("🔄 回退至 Mock 模式")
            return self._mock_judge(question, reply, history)