# config.py
import os
from dotenv import load_dotenv   # 需要安装 python-dotenv

# 加载 .env 文件（本地开发用，不提交到 Git）
load_dotenv()

# ---- 数据路径 ----
DATA_DIR = "data"
REPLIES_FILE = os.path.join(DATA_DIR, "task4_replies.json")
GROUND_TRUTH_FILE = os.path.join(DATA_DIR, "task4_ground_truth.json")

# ---- LLM 配置（从环境变量读取） ----
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "").strip()
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1").strip()
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-chat")

# ---- 运行模式 ----
USE_LLM = os.getenv("USE_LLM", "True").lower() == "true"
USE_MOCK = os.getenv("USE_MOCK", "False").lower() == "true"

# ---- 输出 ----
OUTPUT_REPORT = "report.json"

# 调试：检查 Key 是否已读取（可删除）
if DEEPSEEK_API_KEY:
    print(f"✅ 已读取 DEEPSEEK_API_KEY（前6位: {DEEPSEEK_API_KEY[:6]}...）")
else:
    print("⚠️ 未设置 DEEPSEEK_API_KEY 环境变量，将无法调用真实 API")