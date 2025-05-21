import json
import os
from tqdm import tqdm


task_root = 'Experiment/task-8'
result_path_list = []
for root, dirs, files in os.walk(task_root):
    for file in files:
        if file.endswith('.json') and not file.endswith('eval.json'):
            result_path_list.append(os.path.join(root, file))


for path in tqdm(result_path_list):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    no_math = []
    yes_count = 0

    for x in data:
        gt = x['ground-truth']
        ans = x['response']
        
        if len(gt) == 2:
            if gt[0] in ans:
                yes_count += 0.5
            if gt[1] in ans:
                yes_count += 0.5
        else:
            if gt[0] in ans:
                yes_count += 1

    
    save_list = [
        {
            "yes_count": yes_count
        }
    ]
    save_list = save_list + no_math
    save_path = path.replace('.json', '-eval.json')
    save_path = save_path.replace('/result-', '/eval/result-')
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(save_list, f, ensure_ascii=False, indent=2)

