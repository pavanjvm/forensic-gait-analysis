from ultralytics import YOLO
import cv2
from scipy.spatial.distance import euclidean
import numpy as np

# Load YOLOv8 pose model
model = YOLO(r'C:\Users\pavan\forensic-gait-anlysis\yolov8n-pose.pt')

def get_keypoints(video_path):
    cap = cv2.VideoCapture(video_path)
    keypoints = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame)  # Perform inference on the frame
        for result in results:
            if hasattr(result, 'keypoints'):  # Check if keypoints are available
                keypoints.append(result.keypoints.xy.cpu().numpy())  # x, y coordinates
    cap.release()
    return keypoints

def normalize_keypoints(keypoints):
    normalized = []
    for frame in keypoints:
        center = frame.mean(axis=0)  # Mean of all keypoints
        normalized.append(frame - center)  # Center the keypoints
    return normalized

def compute_similarity(video1_keypoints, video2_keypoints):
    scores = []
    for k1, k2 in zip(video1_keypoints, video2_keypoints):
        distances = [euclidean(p1, p2) for p1, p2 in zip(k1, k2)]
        scores.append(sum(distances) / len(distances))  # Average distance
    return scores

# Input video files
video1_path = r"C:\path\to\crime_scene_video.mp4"
video2_path = r"C:\path\to\suspect_video.mp4"

# Extract and normalize keypoints
keypoints_video1 = get_keypoints(video1_path)
keypoints_video2 = get_keypoints(video2_path)

if not keypoints_video1 or not keypoints_video2:
    print("Failed to extract keypoints from one or both videos.")
else:
    similarity_scores = compute_similarity(
        normalize_keypoints(keypoints_video1),
        normalize_keypoints(keypoints_video2)
    )
    average_similarity = sum(similarity_scores) / len(similarity_scores)
    print(f"Average Similarity Score: {average_similarity}")
