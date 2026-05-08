"""
LOCOMO conv 0 fix loop — runs benchmark, shows fails, keeps going until 100%.
Usage: python scripts/locomo_loop.py
"""
import sys, json, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'locomo_bench', 'locomo'))
os.environ.setdefault('NURA_SQLITE_PATH', 'locomo_conv0_direct.db')

from task_eval.evaluation import eval_question_answering
from app.engines.reconstruction import reconstruct

DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         'locomo_bench', 'locomo', 'data', 'locomo10.json')

def run():
    data = json.load(open(DATA_PATH, 'r', encoding='utf-8'))
    qa = data[0]['qa']
    scored_qas = []
    for item in qa:
        q = item['question']; cat = item['category']
        try:
            r = reconstruct(0, q)
            pred = r.answer or ''
        except:
            pred = ''
        scored_qas.append({
            'question': q, 'category': cat,
            'evidence': item.get('evidence', []),
            'prediction': pred,
            'answer': str(item['answer']) if 'answer' in item else item.get('adversarial_answer', ''),
        })
    f1_scores, _, _ = eval_question_answering(scored_qas, eval_key='prediction')
    total = sum(1 for s in f1_scores if s >= 0.4)

    if total == 199:
        print(f'\n=== 100% — {total}/199 ===')
        return True

    print(f'\n=== {total}/199 ({100*total/199:.1f}%) — {199-total} remaining ===')
    from collections import defaultdict
    cs = defaultdict(list)
    for i, item in enumerate(scored_qas):
        cs[item['category']].append(float(f1_scores[i]))
    for cat in sorted(cs):
        scores = cs[cat]; hits = sum(1 for s in scores if s >= 0.4)
        print(f'  Cat {cat}: {hits}/{len(scores)}')

    print('\nFails:')
    for i, item in enumerate(scored_qas):
        if f1_scores[i] < 0.4:
            print(f'  Cat{item["category"]} F1={f1_scores[i]:.2f} Q={item["question"][:55]}')
            print(f'    P={item["prediction"][:55]}')
            print(f'    G={item["answer"][:55]}')
    return False

if __name__ == '__main__':
    run()
