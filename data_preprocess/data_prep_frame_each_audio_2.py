
# This script is working perfect and it is considering all type of audio file patterns.

import os
import pandas as pd
import subprocess

FFMPEG_PATH = "/home/taruns/mmer/MMER/ffmpeg-git-20240301-amd64-static/ffmpeg"
iemocap_base_path = "/home/taruns/project/IEMOCAP_full_release"
output_dir = "/home/taruns/project/clip/data/frames_each_audio"
os.makedirs(output_dir, exist_ok=True)

def extract_video_path(audio_file_name):
    parts = audio_file_name.split('_')
    Ses = parts[0][:5]
    session = f"Session{audio_file_name[4:5]}"
    actor = parts[0][5]

    if len(parts) > 2 and parts[2].endswith('b'):  # Check if it ends with 'b', e.g., "script01_1b"
        impro_or_script = '_'.join(parts[1:3])
    elif len(parts) > 2 and parts[2].isdigit():
        impro_or_script = '_'.join(parts[1:3])
    else:
        impro_or_script = parts[1]

    video_file_name = f"{Ses}{actor}_{impro_or_script}.avi"
    video_path = os.path.join(iemocap_base_path, session, "dialog", "avi", "DivX", video_file_name)
    return video_path

def extract_times(audio_file_name):
    session = f"Session{audio_file_name[4:5]}"
    base_file_name = '_'.join(audio_file_name.split('_')[:-1])
    transcription_file_path = os.path.join(iemocap_base_path, session, "dialog", "transcriptions", base_file_name + ".txt")

    with open(transcription_file_path, 'r') as file:
        for line in file:
            if line.startswith(audio_file_name):
                timing_info = line.split('[')[1].split(']')[0].split('-')
                start_time = float(timing_info[0])
                end_time = float(timing_info[1])
                return start_time, end_time
    return None, None

csv_file = "../iemocap.csv"
df = pd.read_csv(csv_file)

for index, row in df.iterrows():
    audio_file_name = row['FileName']
    video_file_path = extract_video_path(audio_file_name)
    start_time, end_time = extract_times(audio_file_name)

    hours = int(start_time // 3600)
    minutes = int(start_time % 3600 // 60)
    seconds = int(start_time % 60)
    start_time_formatted = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    hours = int(end_time // 3600)
    minutes = int(end_time % 3600 // 60)
    seconds = int(end_time % 60)
    end_time_formatted = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    output_dir1 = os.path.join(output_dir, audio_file_name)
    if not os.path.exists(output_dir1) or not os.listdir(output_dir1):
        os.makedirs(output_dir1, exist_ok=True)
        output_pattern = os.path.join(output_dir1, f"{audio_file_name}_%04d.jpg")
        ffmpeg_command = f"{FFMPEG_PATH} -i {video_file_path} -ss {start_time_formatted} -t {end_time_formatted} -vf fps=1 {output_pattern} -loglevel quiet"
        try:
            subprocess.run(ffmpeg_command, shell=True, check=True)
        except subprocess.CalledProcessError:
            print(f"Failed to process video: {audio_file_name}")
    else:
        print(f"Skipping {audio_file_name}, output folder already exists and is non-empty.")

    print(f"Processed {index+1}/{len(df)}: {audio_file_name}")
