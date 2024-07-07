import cv2
import csv
import math
import pandas as pd
import os
from pathlib import Path
import subprocess

# Global variables
drawing = False
paused = False
cog_values = []

def calculate_seconds_from_hmsms(time_str):
    """
    Calculate the total seconds from a time string in the format "HH:MM:SS:MS".
    
    Parameters:
    - time_str (str): Time string in the format "HH:MM:SS:MS".
    
    Returns:
    - float: Total seconds with milliseconds as a floating point number.
    """
    # Split the time string into its components
    hours, minutes, seconds, milliseconds = map(int, time_str.split(':'))
    
    # Calculate the total seconds
    total_seconds = hours * 3600 + minutes * 60 + seconds + milliseconds / 1000.0
    
    return total_seconds

def annotate_frame(event, x, y, flags, param):
    global drawing, paused, cog_values, frame, current_frame

    if event == cv2.EVENT_LBUTTONDOWN:
        cog_values[current_frame] = (current_frame, x, y)
        cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
        cv2.putText(frame, "Annotated Point", (x - 20, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.imshow('Frame', frame)
        print(f"Annotated point: Frame={current_frame}, X={x}, Y={y}")

def timecode_to_seconds(timecode, fps):
    h, m, s, f = map(int, timecode.split(':'))
    return h * 3600 + m * 60 + s + f / fps

def cut_video_based_on_markers(video_path, annotations_path, output_path):
    try:
        annotations_df = pd.read_csv(annotations_path, index_col=0)
    except Exception as e:
        print(f"Error reading annotations file: {e}")
        return None
    
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        print("Error: Could not open video.")
        return None

    fps = 30  # Fixed to 30 as per the provided annotation format

    try:
        start_timecode = annotations_df.loc[annotations_df["Markers"] == "climbing", "Start"].iat[0]
        end_timecode = annotations_df.loc[annotations_df["Markers"] == "climbing", "End"].iat[0]
    except IndexError:
        print("Error: Climbing markers not found in annotations.")
        return None
    start_seconds = timecode_to_seconds(start_timecode, fps)
    print(start_seconds)
    end_seconds = timecode_to_seconds(end_timecode, fps)
    print(end_seconds)
    #00:02:11:683	00:06:58:600
    start_seconds = calculate_seconds_from_hmsms("00:03:12:933")
    print(start_seconds)
    end_seconds = calculate_seconds_from_hmsms("00:06:22:700")
    print(end_seconds)

    command = [
        'ffmpeg',
        '-i', str(video_path),
        '-ss', str(start_seconds),
        '-to', str(end_seconds),
        '-c', 'copy',
        str(output_path)
    ]
    
    result = subprocess.run(command, capture_output=True, text=True)
    
    if result.returncode != 0:
        print("FFmpeg error:", result.stderr)
        return None

    return output_path

def check_fps(video_path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print("Error: Could not open video.")
        return None
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps

if __name__ == "__main__":
    participant = 6
    climbing_style = "toprope"
    base_path = Path(__file__).parent / "data" / "trial_1" / f"participant_{participant}"
    #base_path = Path("/Volumes/NO NAME/climbers-grip/data/trial_1/participant_16")
    #path_to_video = base_path / f"participant_{participant}_{climbing_style}.MP4"
    path_to_video = base_path / f"participant_{participant}_{climbing_style}.MOV"
    path_to_annotation_file = base_path / f"{climbing_style}_annotations_alligned.csv"
    path_to_output = base_path / f"{climbing_style}_clipped_file.mp4"

    if not os.path.isfile(path_to_output):
        cut_video_based_on_markers(path_to_video, path_to_annotation_file, path_to_output)

    cap = cv2.VideoCapture(str(path_to_output))

    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()

    original_fps = check_fps(path_to_output)
    print(f"Original FPS: {original_fps}")

    desired_fps = 30  # Change this to 1 if you want 1 fps playback
    if desired_fps not in [1, original_fps]:
        print("Invalid FPS. Defaulting to 30 FPS.")
        desired_fps = 30

    fps = desired_fps
    delay = int(1000 / fps)

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cog_values = [(i, math.nan, math.nan) for i in range(frame_count)]

    cv2.namedWindow('Frame')
    cv2.setMouseCallback('Frame', annotate_frame)

    current_frame = 0

    while cap.isOpened():
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (640, 480))
            cv2.imshow('Frame', frame)
            current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1

        key = cv2.waitKey(delay)
        if key & 0xFF == ord('q'):
            break
        elif key & 0xFF == ord('p'):
            paused = not paused

    cap.release()
    cv2.destroyAllWindows()

    output_csv = Path(__file__).parent / "data" / "trajectories" / f"participant_{participant}_{climbing_style}_trajectories.csv"
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, 'w', newline='') as csvfile:
        cog_writer = csv.writer(csvfile)
        cog_writer.writerow(['Frame', 'COG_X', 'COG_Y'])
        for cog in cog_values:
            cog_writer.writerow(cog)

    print(f"COG values have been saved to {output_csv}")