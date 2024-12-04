import cv2
import os

def preprocess_videos(video_paths, output_dir, target_fps=20, target_duration=6):
    """
    Preprocess multiple videos to have the same FPS, duration, and frame count.
    
    Args:
        video_paths (list): List of paths to input videos
        output_dir (str): Directory to save processed videos
        target_fps (int): Desired frames per second
        target_duration (int): Desired duration in seconds
    """
    # Create output directory if not exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate target frame count
    target_frame_count = target_fps * target_duration
    
    # Open all video captures
    caps = []
    for video_path in video_paths:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Error: Cannot open video file {video_path}")
        caps.append(cap)
    
    # Get video properties
    video_properties = []
    for cap in caps:
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        if fps == 0:
            raise ValueError("Error: One of the video files has an invalid FPS value.")
        video_properties.append({
            'fps': fps,
            'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        })
def preprocess_videos(video_paths, output_dir, target_fps=20, target_duration=6):
    """
    Preprocess multiple videos to have the same FPS, duration, and frame count.
    
    Args:
        video_paths (list): List of paths to input videos
        output_dir (str): Directory to save processed videos
        target_fps (int): Desired frames per second
        target_duration (int): Desired duration in seconds
    """
    # Create output directory if not exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate target frame count
    target_frame_count = target_fps * target_duration
    
    # Open all video captures
    caps = []
    for video_path in video_paths:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Error: Cannot open video file {video_path}")
        caps.append(cap)
    
    # Get video properties
    video_properties = []
    for cap in caps:
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        if fps == 0:
            raise ValueError("Error: One of the video files has an invalid FPS value.")
        video_properties.append({
            'fps': fps,
            'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        })
    
    def process_video(cap, output_path, source_fps):
        """
        Process a single video to match target specifications.
        """
        frames = []
        success, frame = cap.read()
        count = 0
        
        while success and len(frames) < target_frame_count:
            # Adjust to target FPS
            if count % int(source_fps / target_fps) == 0:
                frames.append(frame)
            success, frame = cap.read()
            count += 1
        
        # If we don't have enough frames, loop the video
        while len(frames) < target_frame_count:
            frames.extend(frames[:target_frame_count - len(frames)])
        
        # Trim to exact number of frames if we have too many
        frames = frames[:target_frame_count]
        
        if len(frames) == 0:
            raise ValueError(f"No frames extracted from {output_path}")
        
        # Write processed frames to new video
        height, width, _ = frames[0].shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, target_fps, (width, height))
        
        for frame in frames:
            out.write(frame)
        out.release()
        
        return len(frames)
    
    # Process all videos
    processed_frames = []
    for i, (cap, props) in enumerate(zip(caps, video_properties)):
        output_path = os.path.join(output_dir, f"processed_video{i+1}.mp4")
        frame_count = process_video(cap, output_path, props['fps'])
        processed_frames.append(frame_count)
        print(f"Processed video {i+1}: {frame_count} frames")
        
    # Release all captures
    for cap in caps:
        cap.release()
    
    # Verify all videos have the same number of frames
    if len(set(processed_frames)) != 1:
        raise ValueError("Error: Processed videos have different frame counts")
    
    print(f"\nAll videos processed successfully:")
    print(f"Target FPS: {target_fps}")
    print(f"Target duration: {target_duration} seconds")
    print(f"Frame count per video: {processed_frames[0]}")
    print(f"Output directory: {output_dir}")

# Example usage
video_paths = [
    r'C:\Users\pavan\OneDrive\Desktop\gait anlaysis\forensic-gait-analysis\videos\person1-a.mp4',
    r'C:\Users\pavan\OneDrive\Desktop\gait anlaysis\forensic-gait-analysis\videos\person1-b.mp4',
    r'C:\Users\pavan\OneDrive\Desktop\gait anlaysis\forensic-gait-analysis\videos\person2-a.mp4',
    r'C:\Users\pavan\OneDrive\Desktop\gait anlaysis\forensic-gait-analysis\videos\person2-b.mp4'
]

output_dir = r'C:\Users\pavan\OneDrive\Desktop\gait anlaysis\forensic-gait-analysis\processed_videos'

preprocess_videos(video_paths, output_dir, target_fps=20, target_duration=6)
