from openai import OpenAI
import os
import json
from tqdm import tqdm

client = OpenAI(
    api_key="",
    base_url="" 
)

file_path = []
data_300_path = 'path to your file'
with open(data_300_path, 'r', encoding='utf-8') as f1:
    for line in f1:
        file_path.append(line.strip())

test_path = []
data_test_path = 'path to your file'
with open(data_test_path, 'r', encoding='utf-8') as f1:
    for line in f1:
        test_path.append(line.strip())

temp = []
for x in file_path:
    if x not in test_path:
        temp.append(x)
file_path = temp

video_count = 0
for file in tqdm(file_path):
    file_name = file.split('id-')[-1].replace('.json', '')
    file_name = 'id-' + file_name

    with open(file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    batches = [data[i:i+10] for i in range(0, len(data), 10)]
    
    video_save = []
    i = 0
    for batch in batches:
        i+=1
        """ Prompt Template for Annotation Agent """
        prompt = ""

        completion = client.chat.completions.create(
            model="your API model name",
            messages=[{'role': 'system', 'content': 'You are an expert in the domain of temporal point process (TPP).'},
                    {'role': 'user', 'content': prompt2}],
            stream=False,
            max_tokens=500,
            temperature=0.9
        )
        video_save.append(
            {
                "video_name": file_name,
                "batch": i,
                "annotations": completion.choices[0].message.content
            }
        )
        batch_nums = len(batches)
        print(f"video: {video_count}  &&&  batch: {i}/{batch_nums}  &&&  {file_name}")

    save_path = f'' # need to complete

    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(video_save, f, ensure_ascii=False, indent=2)
