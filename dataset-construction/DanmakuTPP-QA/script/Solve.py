 
from openai import OpenAI
import os
import base64
import json
import ast
from tqdm import tqdm

client = OpenAI(
    api_key="", # complete with your api
    base_url=""  # complete with your api
)


tpp_file_list = []
tpp_root = 'path to your file dir'
for root, dirs, files in os.walk(tpp_root):
    for file in files:
        if file.endswith('.json'):
            tpp_file_list.append(os.path.join(root, file))

result_save_list = []
out_of_len = []
for file_path in tqdm(tpp_file_list):
    with open(file_path, 'r', encoding='utf-8') as f:
        tpp_data = json.load(f)
    f.close()
    time_list = []
    for x in tpp_data:
        time_list.append(str(x['time']))
    if len(time_list) > 2000:
        n = len(time_list)
        out_of_len.append(file_name)

    time_sequence = ", ".join(time_list)

    """
        Prompt Template for Task-Solve Agent
        use after replacing the defined task.

    """
    prompt = ""

    completion = client.chat.completions.create(
        model="qwen3-235b-a22b",
        messages=[
            {'role': 'user', 'content': prompt}
        ],
        stream=False,
        max_tokens=512,
        temperature=0.9,
    )
    response = completion.choices[0].message.content
    response = response.strip()
    # print(completion.choices[0].message.content)
    parts = response.split('peak_timestamps:')
    num_peaks = parts[0].replace('num_peaks:', '').strip()
    peak_time = parts[1].strip()
    try:
        peak_time = ast.literal_eval(peak_time)
    except:
        peak_time = parts[1].strip()

    result_save_list.append(
        {
            "video": file_name,
            "video_path": file_path,
            "peaks_num_time_response": response,
            "num_peaks": num_peaks,
            "peak_timestamps": peak_time
        }
    )

result_save_path = 'result_qwen-v3-235b.json'
out_of_len_path = 'out_of_len_files.txt'

with open(result_save_path, 'w', encoding='utf-8') as f2:
    json.dump(result_save_list, f2, ensure_ascii=False, indent=2)

with open(out_of_len_path, 'w', encoding='utf-8') as f3:
    for x in out_of_len:
        f3.writelines(x + '\n')