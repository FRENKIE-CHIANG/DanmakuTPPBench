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


test_file = 'path to your file'


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

with open(test_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

result_save = []
i = 0

for item in tqdm(data):
    i += 1
    video = item['video']
    question = item['question']
    gt = item['num_peaks']
    time_window = item['time_window']
    time_sequence = item['comment_time_sequence']
    comment_sequence = item['comment_sequence']

    # prompt template for evaluation
    question_llm = f"""
    # Task:
    Given the timestamp sequence of bullet comments in a video, count the burst peaks in the bullet comments.

    # Timestamp sequence:
    {time_sequence}

    # Output Format:
    **ONLY** output the number of peaks **Directly**.
    **DO NOT** print any additional information.
    """

    # task-1
    # question_llm = f"""
    #     # Task:
    #     Given the timestamp sequence of bullet comments in a video, count the burst peaks in the bullet comments.

    #     # Timestamp sequence:
    #     {time_sequence}

    #     # Output Format:
    #     **ONLY** output the number of peaks **Directly**.
    #     **DO NOT** print any additional information.
    #     """

    # task-2
    # question_llm = f"""
    #     # Task:
    #     Given the timestamp sequence of bullet comments in a video, please predict the time of the next bullet comment after the time window: {time_window}

    #     # Timestamp sequence:
    #     {time_sequence}

    #     # Output Format:
    #     **ONLY** output the time.
    #     **DO NOT** print any additional information.
    #     """

    # task-3
    # question_llm = f"""
    #     # Task:
    #     Given the timestamp sequence of bullet comments in a video, please predict the time of the next burst peak of bullet comment after the time window: {time_window}

    #     # Timestamp sequence:
    #     {time_sequence}

    #     # Output Format:
    #     **ONLY** output the time.
    #     **DO NOT** print any additional information.
    #     """

    # task-4
    # question_llm = f"""
    #     # Task:
    #     You are provided with a timestamp sequence of bullet comments in a video and the comments sequence. Please calculate the average sentiment polarity (-1 to 1) within the time window: {time_window}

    #     # Timestamp sequence:
    #     {time_sequence}

    #     # Comment Sequence:
    #     {comment_sequence}

    #     # Output Format:
    #     **ONLY** output the sentiment polarity.
    #     **DO NOT** print any additional information.
    #     """


    # task-5
    # question_llm = f"""
    #     # Task:
    #     You are provided with a timestamp sequence of bullet comments in a video and the comments sequence. Please predict the sentiment polarity (-1 to 1) of the next bullet comment after the time window: {time_window}

    #     # Timestamp sequence:
    #     {time_sequence}

    #     # Comment Sequence:
    #     {comment_sequence}

    #     # Output Format:
    #     **ONLY** output the sentiment polarity.
    #     **DO NOT** print any additional information.
    #     """


    # task-6
    # question_llm = f"""
    #     # Task:
    #     You are provided with a timestamp sequence of bullet comments in a video and the comments sequence. Please predict the sentiment polarity (-1 to 1) of the next burst peak of bullet comment after the time window:  {time_window}

    #     # Timestamp sequence:
    #     {time_sequence}

    #     # Comment Sequence:
    #     {comment_sequence}

    #     # Output Format:
    #     **ONLY** output the sentiment polarity.
    #     **DO NOT** print any additional information.
    #     """

    # task-7
    # question_llm = f"""
    #     # Task:
    #     You are provided with a timestamp sequence of bullet comments in a video and the comments sequence. Predict the event type of the next bullet comment after the time window: {time_window}\nCandidate: [\"question\", \"humor/meme\", \"critical_comment\", \"personal_experience\", \"off_topic\", \"quote/reference\", \"nonsense_text\", \"social_interaction\", \"emotional_reaction\"]

    #     # Timestamp sequence:
    #     {time_sequence}

    #     # Comment Sequence:
    #     {comment_sequence}

    #     # Output Format:
    #     **ONLY** output the event type.
    #     **DO NOT** print any additional information.
    #     """

    # task-8
    # question_llm = f"""
    #     # Task:
    #     You are provided with a timestamp sequence of bullet comments in a video and the comments sequence. Please predict the **TWO** most likely types of events that will trigger the next burst peak of bullet comments after the time window: {time_window}\nCandidate: [\"question\", \"humor/meme\", \"critical_comment\", \"personal_experience\", \"off_topic\", \"quote/reference\", \"nonsense_text\", \"social_interaction\", \"emotional_reaction\"]

    #     # Timestamp sequence:
    #     {time_sequence}

    #     # Comment Sequence:
    #     {comment_sequence}

    #     # Output Format:
    #     **ONLY** output the two event type.
    #     **DO NOT** print any additional information.
    #     """

    # task-9
    # question_llm = f"""
    #     # Task:
    #     You are provided with a timestamp sequence of bullet comments in a video and the comments sequence. Please focus on the sentiment variation trend to provide a detailed analysis report of the sentiment trend in the bullet comments over time.

    #     # Timestamp sequence:
    #     {time_sequence}

    #     # Comment Sequence:
    #     {comment_sequence}


    #     # Output Format:
    #     **ONLY** output the analysis report.
    #     **DO NOT** print any additional information.
    #     **DO NOT** exceed 120 words.
    #     Answer in **English**.
    #     """

    # task-10
    # question_llm = f"""
    #     # Task:
    #     You are provided with a timestamp sequence of bullet comments in a video and the comments sequence. Please analyze in detail the causes of the comment burst peaks within the time window: {time_window}.

    #     # Timestamp sequence:
    #     {time_sequence}

    #     # Comment Sequence:
    #     {comment_sequence}

    #     # Output Format:
    #     **ONLY** output the analysis report.
    #     **DO NOT** print any additional information.
    #     **DO NOT** exceed 320 words.
    #     Answer in **English**.
    #     """

    ######################
    # question_mllm = f"""
    # # Task:
    # You are provided with a timestamp sequence of bullet comments in a video and randomly sampled video frames, count the burst peaks in the bullet comments.

    # # Timestamp sequence:
    # {time_sequence}

    # # Output Format:
    # **ONLY** output the number of peaks.
    # **DO NOT** print any additional information.
    # """

    question = question_llm
    # question = '/no_think\n' + question 

    strict = '\n**STRICTLY** follow the output format. **DO NOT** print any analysis or thinking process.'
    question = question + strict
    
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {'role': 'system', 'content': 'You are a helpful assistant.'},
            {'role': 'user', 'content': question}
        ],
        stream=False,
        max_tokens=64,
        temperature=0.9,
    )
    response = completion.choices[0].message.content
    response = response.strip()
    if i <= 10:
        print(response)
    
    result_save.append(
        {
            "video": video,
            "ground-truth": gt,
            "response": response
        }
    )

save_path = ''

with open(save_path, 'w', encoding='utf-8') as f:
    json.dump(result_save, f, ensure_ascii=False, indent=2)
