import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import savgol_filter
import os
from tqdm import tqdm

plt.rcParams['font.sans-serif'] = ['DejaVu Sans'] 
plt.rcParams['axes.unicode_minus'] = False  

file_dir = '/data01/jy/work-dir/TPP/annotation_result/annontation_data_2305_csv'
file_path = []
for root, dirs, files in os.walk(file_dir):
    for file in files:
        if file.endswith('.csv'):
            file_path.append(os.path.join(root, file))

csv_count = 0
save_dir = "/data01/jy/work-dir/TPP/graph_result/data_2305_graph"
for csv_path in tqdm(file_path):
    csv_count += 1

    file_name = csv_path.split('annontation_data_2305_csv/')[-1].replace('.csv', '')
    save_sub_dir = os.path.join(save_dir, file_name)
    os.makedirs(save_sub_dir, exist_ok=True)
    try:
        df = pd.read_csv(csv_path, 
                        header=0, 
                        names=["time","int_time","text","time_since_start",
                                "time_since_last_event","type_event","interactivity",
                                "humor_level","sentiment_Polarity","sentiment_Intensity"],
                        engine='python')
    except Exception as e:
        print(f"Data loading failed: {str(e)}  &&&  {file_name}")
        raise

    required_columns = ["time","int_time","text","time_since_start",
                        "time_since_last_event","type_event","interactivity",
                        "humor_level","sentiment_Polarity","sentiment_Intensity"]

    if not all(col in df.columns for col in required_columns):
        raise ValueError("Missing required columns. Please check CSV format") [[2]][[6]]

    plt.figure(figsize=(12,6))
    df['time_since_start'] = pd.to_datetime(df['time_since_start'], unit='s')
    df.set_index('time_since_start', inplace=True)
    minutely_count = df.resample('min').size()
    minutely_count.plot(label='Original Data')
    minutely_count.rolling(window=30).mean().plot(label='30s Moving Average')

    # Event type analysis - Bar chart
    plt.figure(figsize=(10,6))
    sns.countplot(data=df, y='type_event')
    plt.title('Distribution of Event Types')
    plt.tight_layout()
    plt.savefig(f'{save_sub_dir}/event_type_distribution.png', dpi=300)
    plt.close()

    # Event type analysis - Stacked area chart
    stacked_df = df.groupby([df.index.floor('min'), 'type_event']).size().unstack(fill_value=0)
    stacked_df = stacked_df.cumsum()
    stacked_df.plot.area(stacked=True, figsize=(12,8), alpha=0.4)
    plt.title('Event Type Evolution Over Time')
    plt.tight_layout()
    plt.savefig(f'{save_sub_dir}/stacked_area_chart.png', dpi=300)
    plt.close()

    # Interactivity analysis - Pie chart
    plt.figure(figsize=(8,8))
    df['interactivity'].value_counts().plot.pie(autopct='%1.1f%%')
    plt.title('Proportional Distribution of Interaction Types')
    plt.tight_layout()
    plt.savefig(f'{save_sub_dir}/interactivity_pie.png', dpi=300)
    plt.close()

    # Sentiment analysis - Line chart
    plt.figure(figsize=(12,6))
    smoothed = savgol_filter(df['sentiment_Polarity'].rolling(60).mean().fillna(0), 
                            window_length=51, polyorder=3)
    plt.plot(df.index, smoothed)
    plt.title('Sentiment Polarity Trend Over Time')
    plt.tight_layout()
    plt.savefig(f'{save_sub_dir}/sentiment_trend.png', dpi=300)
    plt.close()

    # Sentiment analysis - Grouped boxplot
    plt.figure(figsize=(10,6))
    sns.boxplot(data=df, x='sentiment_Intensity', y='sentiment_Polarity')
    plt.title('Sentiment Polarity Distribution by Intensity')
    plt.tight_layout()
    plt.savefig(f'{save_sub_dir}/sentiment_boxplot.png', dpi=300)
    plt.close()

    # Humor-level analysis - Percentage stacked bar chart
    cross_tab = pd.crosstab(df['type_event'], df['humor_level'], normalize='index')
    cross_tab.plot.bar(stacked=True, figsize=(12,8))
    plt.title('Humor Levels Across Event Types')
    plt.tight_layout()
    plt.savefig(f'{save_sub_dir}/humor_stacked_bar.png', dpi=300)
    plt.close()

    # New Danmu intensity graph (5-second window)
    plt.figure(figsize=(14, 7))
    df_5s = df.resample('5s').size()
    df_5s.plot(linewidth=2)
    plt.xlabel('Time Since Video Start (Seconds)', fontsize=12)
    plt.ylabel('Number of Bullet Chat', fontsize=12)
    plt.title('Bullet Chat Burst Intensity Analysis (5s Window)', fontsize=14)
    plt.grid(True, alpha=0.3)

    # Annotate peak value
    max_idx = df_5s.idxmax()
    max_value = df_5s.loc[max_idx]
    timeee = str(max_idx).replace('1970-01-01', '')
    plt.annotate(f'Peak: {max_value} entries\nTime: {timeee}', 
                xy=(max_idx, max_value),
                xytext=(0.8, 0.9), textcoords='axes fraction',
                arrowprops=dict(facecolor='red', shrink=0.05),
                fontsize=10,
                bbox=dict(boxstyle="round,pad=0.5", fc="yellow", alpha=0.5))
    plt.tight_layout()
    plt.savefig(f'{save_sub_dir}/Bullet_Chat_Intensity.png', dpi=300)
    plt.close()

    print(f"Done: video: {csv_count}  &&&  {file_name}")
