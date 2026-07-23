# main.py
import json
import sys
from config import REPLIES_FILE, GROUND_TRUTH_FILE, USE_LLM, USE_MOCK, OUTPUT_REPORT
from pipeline import DetectorPipeline
from evaluator import Evaluator

def load_data():
    """加载 JSON 数据"""
    try:
        with open(REPLIES_FILE, 'r', encoding='utf-8') as f:
            replies = json.load(f)
        with open(GROUND_TRUTH_FILE, 'r', encoding='utf-8') as f:
            ground_truth = json.load(f)
        return replies, ground_truth
    except FileNotFoundError as e:
        print(f"❌ 文件未找到: {e}")
        print("请确保 data/ 目录下存在 task4_replies.json 和 task4_ground_truth.json")
        sys.exit(1)

def main():
    print(" 启动智能客服幻觉检测工具...")
    replies, ground_truth = load_data()
    print(f"✅ 加载 {len(replies)} 条回复，{len(ground_truth)} 条标注")

    # 初始化检测流水线
    detector = DetectorPipeline(use_llm=USE_LLM, use_mock=USE_MOCK)

    # 逐条检测
    results = []
    for item in replies:
        # 将 knowledge_base 作为上下文传入（可选）
        kb = item.get('knowledge_base', None)
        # 若知识库是字符串，转为字典（简化处理，也可直接透传）
        if isinstance(kb, str):
            kb = {"raw": kb}
        res = detector.detect_single(
            item_id=item['id'],
            question=item['user_question'],
            reply=item['system_reply'],
            knowledge_base=kb,
            history=[]  # 本数据集无历史
        )
        results.append(res)

    # 评估
    evaluator = Evaluator()
    report = evaluator.evaluate(results, ground_truth)

    # 打印结果
    print("\n" + "=" * 40)
    print("📊 检测结果汇总")
    print("=" * 40)
    for res in results:
        print(f"ID {res.id}: {'🔴 幻觉' if res.is_hallucination else '✅ 正常'} | "
              f"类型: {[t.value for t in res.detected_types]} | 严重: {res.severity}")

    print("\n" + "=" * 40)
    print("📈 评估指标")
    print("=" * 40)
    print(f"TP: {report['TP']}, FP: {report['FP']}, FN: {report['FN']}, TN: {report['TN']}")
    print(f"召回率 (Recall): {report['recall']:.2%}")
    print(f"精确率 (Precision): {report['precision']:.2%}")
    print(f"F1 分数: {report['f1']:.2%}")

    print("\n" + "=" * 40)
    print("⚠️ 误判详情 (漏检/误报)")
    print("=" * 40)
    for d in report['detail']:
        if d['status'] in ['FN', 'FP']:
            print(f"ID {d['id']} [{d['status']}]: GT={d['ground_truth']}, Pred={d['predicted']}")

    # 保存报告
    with open(OUTPUT_REPORT, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n💾 详细报告已保存至 {OUTPUT_REPORT}")

if __name__ == "__main__":
    main()
