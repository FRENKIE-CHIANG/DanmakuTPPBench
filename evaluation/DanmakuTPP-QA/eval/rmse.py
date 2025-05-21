import json
import os
from tqdm import tqdm
from sklearn.metrics import mean_squared_error
import numpy as np


task_root = 'Experiment/task-2'
result_path_list = []
for root, dirs, files in os.walk(task_root):
    for file in files:
        if file.endswith('.json') and not file.endswith('eval.json'):
            result_path_list.append(os.path.join(root, file))


for path in tqdm(result_path_list):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    no_math = []

    match_count = 0

    sum_rmse = 0
    for x in data:
        gt = x['ground-truth']
        ans = x['response']
        ans = ans.replace('*', '').strip()
        if len(ans) > 10:
            no_math.append(x)
        
        else:
            gt = float(gt)
            ans = float(ans)
            rmse = np.sqrt(np.mean((gt - ans) ** 2))
            sum_rmse += rmse
            match_count += 1

    ave_rmse = sum_rmse / match_count

    
    save_list = [
        {
            "match_count": match_count,
            "ave_rmse": ave_rmse
        }
    ]
    save_list = save_list + no_math
    save_path = path.replace('.json', '-eval.json')
    save_path = save_path.replace('/result-', '/eval/result-')
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(save_list, f, ensure_ascii=False, indent=2)

