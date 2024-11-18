import numpy as np
import pandas as pd
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt
import os

def load_sequence_keypoints(folder_path):
    """
    Load all keypoint files from a folder and arrange them in sequence
    """
    all_frames = []
    
    # Get all txt files in the folder
    files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]
    files.sort()  # Ensure files are in sequence
    
    for file in files:
        with open(os.path.join(folder_path, file), 'r') as f:
            data = f.readline().strip().split()  # Read first line of each file
            frame_data = [float(x) for x in data]
            all_frames.append(frame_data)
    
    return np.array(all_frames)

def extract_key_points(sequence_data):
    """
    Extract relevant keypoints for gait comparison
    """
    print("Frame data shape:", sequence_data.shape)
    print("Number of values in first frame:", len(sequence_data[0]))
    
    relevant_points = []
    
    for frame in sequence_data:
        keypoints = frame[7:]  # All keypoints data
        
        if len(relevant_points) == 0:
            print("First frame keypoints length:", len(keypoints))
        
        left_hip_idx = 29
        right_hip_idx = 31
        left_knee_idx = 33
        right_knee_idx = 35
        left_ankle_idx = 37
        right_ankle_idx = 39
        
        try:
            frame_points = {
                'left_hip': (keypoints[left_hip_idx], keypoints[left_hip_idx + 1]),
                'right_hip': (keypoints[right_hip_idx], keypoints[right_hip_idx + 1]),
                'left_knee': (keypoints[left_knee_idx], keypoints[left_knee_idx + 1]),
                'right_knee': (keypoints[right_knee_idx], keypoints[right_knee_idx + 1]),
                'left_ankle': (keypoints[left_ankle_idx], keypoints[left_ankle_idx + 1]),
                'right_ankle': (keypoints[right_ankle_idx], keypoints[right_ankle_idx + 1])
            }
            relevant_points.append(frame_points)
        except IndexError as e:
            print(f"Warning: Frame has insufficient keypoints. Expected indices up to {right_ankle_idx + 1}")
            print(f"Available keypoints length: {len(keypoints)}")
            print(f"Frame data: {frame}")
            raise e
    
    return relevant_points

def calculate_angle(point1, point2, point3):
    """
    Calculate angle between three points with improved error handling and debugging
    """
    try:
        p1 = np.array(point1)
        p2 = np.array(point2)
        p3 = np.array(point3)
        
        vector1 = p1 - p2
        vector2 = p3 - p2
        
        if np.all(vector1 == 0) or np.all(vector2 == 0):
            print(f"Warning: Zero vector detected!")
            print(f"Points: {point1}, {point2}, {point3}")
            return 0
        
        dot_product = np.dot(vector1, vector2)
        magnitude1 = np.linalg.norm(vector1)
        magnitude2 = np.linalg.norm(vector2)
        
        if magnitude1 == 0 or magnitude2 == 0:
            print(f"Warning: Zero magnitude detected!")
            print(f"Vector1 magnitude: {magnitude1}, Vector2 magnitude: {magnitude2}")
            return 0
        
        cos_angle = dot_product / (magnitude1 * magnitude2)
        
        if cos_angle > 1:
            cos_angle = 1
        elif cos_angle < -1:
            cos_angle = -1
        
        angle = np.degrees(np.arccos(cos_angle))
        
        if angle == 0:
            print(f"Zero angle detected!")
            print(f"Points: Hip: {point1}, Knee: {point2}, Ankle: {point3}")
            print(f"Vectors: {vector1}, {vector2}")
            print(f"Dot product: {dot_product}")
            print(f"Magnitudes: {magnitude1}, {magnitude2}")
            print(f"Cosine: {cos_angle}")
        
        return angle
        
    except Exception as e:
        print(f"Error in angle calculation: {str(e)}")
        print(f"Points: {point1}, {point2}, {point3}")
        return 0

def calculate_gait_features(points_sequence):
    """
    Calculate key gait features for each frame
    """
    features = []
    
    for i, points in enumerate(points_sequence):
        step_length = euclidean(points['left_ankle'], points['right_ankle'])
        stance_width = euclidean(points['left_hip'], points['right_hip'])
        
        left_knee_angle = calculate_angle(
            points['left_hip'],
            points['left_knee'],
            points['left_ankle']
        )
        
        if left_knee_angle == 0:
            print(f"\nFrame {i} Left Knee Debug:")
            print(f"Left Hip: {points['left_hip']}")
            print(f"Left Knee: {points['left_knee']}")
            print(f"Left Ankle: {points['left_ankle']}")
        
        right_knee_angle = calculate_angle(
            points['right_hip'],
            points['right_knee'],
            points['right_ankle']
        )
        
        features.append([step_length, stance_width, left_knee_angle, right_knee_angle])
    
    features_array = np.array(features)
    print("\nFeature Statistics:")
    print("Step Length - Mean:", np.mean(features_array[:, 0]), "Max:", np.max(features_array[:, 0]))
    print("Stance Width - Mean:", np.mean(features_array[:, 1]), "Max:", np.max(features_array[:, 1]))
    print("Left Knee Angle - Mean:", np.mean(features_array[:, 2]), "Max:", np.max(features_array[:, 2]))
    print("Right Knee Angle - Mean:", np.mean(features_array[:, 3]), "Max:", np.max(features_array[:, 3]))
    
    return np.array(features)

def compare_sequences(features1, features2):
    """
    Compare two gait sequences
    """
    differences = np.mean(np.abs(features1 - features2), axis=0)
    
    max_diff = np.max(differences)
    if max_diff == 0:
        similarities = np.array([100.0] * len(differences))
    else:
        similarities = 100 * (1 - differences / max_diff)
    
    feature_names = ['Step Length', 'Stance Width', 'Left Knee Angle', 'Right Knee Angle']
    detailed_metrics = dict(zip(feature_names, similarities))
    
    overall_score = np.mean(similarities)
    
    return overall_score, detailed_metrics

def visualize_comparison(features1, features2):
    """
    Visualize the comparison between two gait sequences
    """
    feature_names = ['Step Length', 'Stance Width', 'Left Knee Angle', 'Right Knee Angle']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Gait Pattern Comparison')
    
    for i, (ax, name) in enumerate(zip(axes.flat, feature_names)):
        ax.plot(features1[:, i], label='Person 1', alpha=0.7)
        ax.plot(features2[:, i], label='Person 2', alpha=0.7)
        ax.set_title(name)
        ax.set_xlabel('Frame')
        ax.legend()
        ax.grid(True)
    
    plt.tight_layout()
    plt.show()

def main(folder1, folder2):
    """
    Main function to process and compare two gait sequences
    """
    # Load sequences
    print("Loading sequences...")
    seq1 = load_sequence_keypoints(folder1)
    seq2 = load_sequence_keypoints(folder2)
    
    # Extract relevant points
    print("Extracting keypoints...")
    points1 = extract_key_points(seq1)
    points2 = extract_key_points(seq2)
    
    # Calculate features
    print("Calculating gait features...")
    features1 = calculate_gait_features(points1)
    features2 = calculate_gait_features(points2)
    
    # Compare sequences
    print("Comparing sequences...")
    similarity_score, detailed_metrics = compare_sequences(features1, features2)
    
    # Visualize comparison
    visualize_comparison(features1, features2)
    
    # Print results
    print(f"\nOverall Similarity Score: {similarity_score:.2f}%")
    print("\nDetailed Metrics (higher is more similar):")
    for feature, score in detailed_metrics.items():
        print(f"{feature}: {score:.2f}%")
    
    return similarity_score, detailed_metrics

if __name__ == "__main__":
    folder1 = r"C:\Users\pavan\OneDrive\Desktop\gait anlaysis\forensic-gait-analysis\runs\pose\predict3\labels"
    folder2 = r"C:\Users\pavan\OneDrive\Desktop\gait anlaysis\forensic-gait-analysis\runs\pose\predict4\labels"
    
    similarity_score, metrics = main(folder1, folder2)