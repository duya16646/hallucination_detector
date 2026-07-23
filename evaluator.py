# evaluator.py
from typing import List, Dict
from models import DetectionResult, HallucinationType

class Evaluator:
    @staticmethod
    def evaluate(results: List[DetectionResult], ground_truth: List[Dict]) -> Dict:
        """对比检测结果与人工标注，输出混淆矩阵及指标"""
        gt_map = {item['id']: item for item in ground_truth}
        tp = fp = fn = tn = 0
        detail = []

        for res in results:
            gt_item = gt_map.get(res.id)
            if not gt_item:
                continue

            gt_has = gt_item.get('is_hallucination', False)
            pred_has = res.is_hallucination

            if pred_has and gt_has:
                tp += 1
                status = "TP"
            elif pred_has and not gt_has:
                fp += 1
                status = "FP"
            elif not pred_has and gt_has:
                fn += 1
                status = "FN"
            else:
                tn += 1
                status = "TN"

            detail.append({
                'id': res.id,
                'ground_truth': gt_item.get('hallucination_type', '无'),
                'predicted': [t.value for t in res.detected_types],
                'status': status,
                'reason': res.reason
            })

        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn,
            'recall': recall, 'precision': precision, 'f1': f1,
            'detail': detail
        }