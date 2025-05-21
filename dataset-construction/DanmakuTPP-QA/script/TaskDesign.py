# pip install transformers pillow imagehash faiss-cpu

import json
import imagehash
from PIL import Image
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from collections import defaultdict
import faiss  # 向量数据库
from openai import OpenAI


def extract_key_frames(data, interval=30):
    key_frames = []
    prev_hash = None
    for entry in data:
        if entry['time'] % interval == 0:
            try:
                img = Image.open(entry['image_path'])
                curr_hash = str(imagehash.phash(img))
                if curr_hash != prev_hash:
                    key_frames.append(entry['image_path'])
                    prev_hash = curr_hash
            except FileNotFoundError:
                continue
    return key_frames[:10]

def generate_intensity_curve(segment_data):
    times = [d['time'] for d in segment_data]
    hist, _ = np.histogram(times, bins=10, density=True)
    return hist.tolist() 

def preprocess_video(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    global_stats = {
        "duration": max(d['time'] for d in data),
        "danmu_count": len(data),
        "key_scenes": extract_key_frames(data),
    }
    
    segment_data = data[:100]
    segment_sample = {
        "time_range": f"0-{segment_data[-1]['time']}s",
        "danmu_sample": [{"time": d['time'], "text": d['text']} for d in segment_data],
        "image_path": segment_data[len(segment_data)//2]['image_path'],
        "intensity_curve": generate_intensity_curve(segment_data)
    }
    return_dict = {"global_stats": global_stats, "segment_sample": segment_sample}

    with open('/data01/jy/work-dir/TPP/Agents/demo.json', 'w', encoding='utf-8') as f:
        json.dump(return_dict, f, ensure_ascii=False, indent=2)

    return return_dict

class VectorDB:
    def __init__(self):
        self.dimension = 512 
        self.index = faiss.IndexFlatL2(self.dimension)
        self.metadata = [] 
    
    def add_embeddings(self, embeddings, labels):
        self.index.add(embeddings)
        self.metadata.extend(labels)
    
    def search(self, query_embedding, k=5):
        distances, indices = self.index.search(np.array([query_embedding]), k)
        return [self.metadata[i] for i in indices[0]]

db = VectorDB()
db.add_embeddings(np.random.rand(100, 512), ["scene_{}".format(i) for i in range(100)])

class TaskDefineAgent:
    def __init__(self, model_name="/data01/jy/work-dir/model/Qwen2.5-7B-Instruct"):

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
    
    def generate_tasks(self, processed_data):
        """Prompt Template for Task-Design Agent"""
        prompt = ""
        
        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.model.generate(**inputs, max_new_tokens=256)
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        tasks = []
        for line in response.split('\n'):
            if line.startswith('1.') or line.startswith('2.') or line.startswith('3.'):
                tasks.append(line.split('. ', 1)[1])
        return tasks

def main(json_path):
    processed_data = preprocess_video(json_path)
    
    agent = TaskDefineAgent()
    
    tasks = agent.generate_tasks(processed_data)
    print("Generated Tasks: ", tasks)
    


if __name__ == "__main__":
    file_path = "path to your JSON file"
    main(file_path)  