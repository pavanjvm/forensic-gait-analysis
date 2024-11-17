import cv2
import os

def preprocess_videos(video_path1, video_path2, output_dir, target_fps=20):
    # Create output directory if not exists
    os.makedirs(output_dir, exist_ok=True)

    # Open the video files
    cap1 = cv2.VideoCapture(video_path1)
    cap2 = cv2.VideoCapture(video_path2)

    # Check if videos are loaded
    if not cap1.isOpened():
        raise ValueError(f"Error: Cannot open video file {video_path1}")
    if not cap2.isOpened():
        raise ValueError(f"Error: Cannot open video file {video_path2}")

    # Get video properties
    fps1 = int(cap1.get(cv2.CAP_PROP_FPS))
    fps2 = int(cap2.get(cv2.CAP_PROP_FPS))
    frame_count1 = int(cap1.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_count2 = int(cap2.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps1 == 0 or fps2 == 0:
        raise ValueError("Error: One of the video files has an invalid FPS value.")

    duration1 = frame_count1 / fps1
    duration2 = frame_count2 / fps2

    # Determine the minimum duration
    min_duration = min(duration1, duration2)

    # Determine the frame counts for the minimum duration at target FPS
    target_frame_count = int(min_duration * target_fps)

    # Helper function to process a single video
    def process_video(cap, output_path, source_fps):
        frames = []
        success, frame = cap.read()
        count = 0
        while success and len(frames) < target_frame_count:
            # Adjust to target FPS
            if count % int(source_fps / target_fps) == 0:
                frames.append(frame)
            success, frame = cap.read()
            count += 1

        # Write processed frames to a new video
        if len(frames) == 0:
            raise ValueError(f"No frames extracted from {output_path}.")
        
        height, width, _ = frames[0].shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, target_fps, (width, height))
        for frame in frames:
            out.write(frame)
        out.release()

    # Process both videos
    output_video1 = os.path.join(output_dir, "processed_video1.mp4")
    output_video2 = os.path.join(output_dir, "processed_video2.mp4")
    process_video(cap1, output_video1, fps1)
    process_video(cap2, output_video2, fps2)

    # Release video captures
    cap1.release()
    cap2.release()

    print(f"Preprocessed videos saved to:\n{output_video1}\n{output_video2}")

# Example usage
video1 = r'C:\Users\pavan\OneDrive\Desktop\gait anlaysis\forensic-gait-analysis\set1.0.mp4'
video2 = r'C:\Users\pavan\OneDrive\Desktop\gait anlaysis\forensic-gait-analysis\set1.1.mp4'
output_videos = r'C:\Users\pavan\OneDrive\Desktop\gait anlaysis\forensic-gait-analysis\processed_videos'

preprocess_videos(video1, video2, output_videos, target_fps=20)
