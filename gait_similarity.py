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
    Calculate angle between three points with enhanced debugging and validation
    """
    try:
        p1 = np.array(point1)
        p2 = np.array(point2)
        p3 = np.array(point3)
        
        # Validate input points
        if np.any(np.isnan(p1)) or np.any(np.isnan(p2)) or np.any(np.isnan(p3)):
            print(f"Warning: NaN values detected in points")
            return None
            
        vector1 = p1 - p2
        vector2 = p3 - p2
        
        # Early validation of vectors
        if np.allclose(vector1, 0) or np.allclose(vector2, 0):
            print(f"Warning: Zero vector detected")
            print(f"Vector1: {vector1}, Vector2: {vector2}")
            return None
            
        # Calculate angle using arctan2 for more stable results
        angle = np.abs(np.degrees(
            np.arctan2(np.linalg.norm(np.cross(vector1, vector2)), np.dot(vector1, vector2))
        ))
        
        # Validate angle
        if not (0 <= angle <= 180):
            print(f"Warning: Invalid angle calculated: {angle}")
            return None
            
        return angle
        
    except Exception as e:
        print(f"Error in angle calculation: {str(e)}")
        print(f"Points: {point1}, {point2}, {point3}")
        return None

def calculate_gait_features(points_sequence):
    """
    Calculate key gait features with enhanced error handling
    """
    features = []
    valid_frames = 0
    invalid_frames = 0
    
    for i, points in enumerate(points_sequence):
        try:
            # Calculate basic measurements
            step_length = euclidean(points['left_ankle'], points['right_ankle'])
            stance_width = euclidean(points['left_hip'], points['right_hip'])
            
            # Calculate angles with validation
            left_knee_angle = calculate_angle(
                points['left_hip'],
                points['left_knee'],
                points['left_ankle']
            )
            
            right_knee_angle = calculate_angle(
                points['right_hip'],
                points['right_knee'],
                points['right_ankle']
            )
            
            # Validate all measurements
            if all(x is not None and not np.isnan(x) for x in [step_length, stance_width, left_knee_angle, right_knee_angle]):
                features.append([step_length, stance_width, left_knee_angle, right_knee_angle])
                valid_frames += 1
            else:
                print(f"\nSkipping frame {i} due to invalid measurements:")
                print(f"Step length: {step_length}")
                print(f"Stance width: {stance_width}")
                print(f"Left knee angle: {left_knee_angle}")
                print(f"Right knee angle: {right_knee_angle}")
                invalid_frames += 1
                
        except Exception as e:
            print(f"Error processing frame {i}: {str(e)}")
            invalid_frames += 1
            continue
    
    if not features:
        raise ValueError("No valid frames were processed!")
        
    features_array = np.array(features)
    
    print(f"\nProcessing Summary:")
    print(f"Valid frames: {valid_frames}")
    print(f"Invalid frames: {invalid_frames}")
    print(f"Valid frame percentage: {(valid_frames/(valid_frames+invalid_frames))*100:.2f}%")
    
    print("\nFeature Statistics:")
    print("Step Length - Mean:", np.mean(features_array[:, 0]), "Std:", np.std(features_array[:, 0]))
    print("Stance Width - Mean:", np.mean(features_array[:, 1]), "Std:", np.std(features_array[:, 1]))
    print("Left Knee Angle - Mean:", np.mean(features_array[:, 2]), "Std:", np.std(features_array[:, 2]))
    print("Right Knee Angle - Mean:", np.mean(features_array[:, 3]), "Std:", np.std(features_array[:, 3]))
    
    return features_array

def compare_sequences(features1, features2):
    """
    Compare two gait sequences with improved normalization
    """
    # Ensure sequences are of same length
    min_length = min(len(features1), len(features2))
    features1 = features1[:min_length]
    features2 = features2[:min_length]
    
    # Calculate differences with normalization
    differences = []
    for i in range(4):  # For each feature
        f1 = features1[:, i]
        f2 = features2[:, i]
        
        # Normalize each feature to its range
        range_val = max(np.ptp(f1), np.ptp(f2))
        if range_val > 0:
            diff = np.mean(np.abs(f1 - f2)) / range_val
        else:
            diff = 0
        differences.append(diff)
    
    differences = np.array(differences)
    
    # Convert to similarity scores
    similarities = 100 * (1 - differences)
    
    feature_names = ['Step Length', 'Stance Width', 'Left Knee Angle', 'Right Knee Angle']
    detailed_metrics = dict(zip(feature_names, similarities))
    
    overall_score = np.mean(similarities)
    
    return overall_score, detailed_metrics

def visualize_comparison(features1, features2):
    """
    Visualize the comparison between two gait sequences with simplified plotting
    """
    feature_names = ['Step Length', 'Stance Width', 'Left Knee Angle', 'Right Knee Angle']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Gait Pattern Comparison', fontsize=16)
    
    for i, (ax, name) in enumerate(zip(axes.flat, feature_names)):
        frames = range(len(features1[:, i]))
        ax.plot(frames, features1[:, i], label='Person 1', alpha=0.7, linewidth=2)
        ax.plot(frames, features2[:, i], label='Person 2', alpha=0.7, linewidth=2)
        ax.set_title(name, fontsize=12)
        ax.set_xlabel('Frame', fontsize=10)
        ax.set_ylabel('Value', fontsize=10)
        ax.legend(fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.tick_params(labelsize=8)
    
    plt.tight_layout()
    plt.show()

def main(folder1, folder2):
    """
    Main function to process and compare two gait sequences
    """
    try:
        # Load sequences
        print("Loading sequences...")
        seq1 = load_sequence_keypoints(folder1)
        seq2 = load_sequence_keypoints(folder2)
        
        # Extract relevant points
        print("\nExtracting keypoints...")
        points1 = extract_key_points(seq1)
        points2 = extract_key_points(seq2)
        
        # Calculate features
        print("\nCalculating gait features for sequence 1...")
        features1 = calculate_gait_features(points1)
        print("\nCalculating gait features for sequence 2...")
        features2 = calculate_gait_features(points2)
        
        # Compare sequences
        print("\nComparing sequences...")
        similarity_score, detailed_metrics = compare_sequences(features1, features2)
        
        # Print results
        print(f"\nOverall Similarity Score: {similarity_score:.2f}%")
        print("\nDetailed Metrics (higher is more similar):")
        for feature, score in detailed_metrics.items():
            print(f"{feature}: {score:.2f}%")
        
        # Visualize comparison
        print("\nGenerating visualization...")
        visualize_comparison(features1, features2)
        
        return similarity_score, detailed_metrics
        
    except Exception as e:
        print(f"\nError in main processing: {str(e)}")
        raise

if __name__ == "__main__":
    # Configure folders
    folder1 = r"C:\Users\pavan\OneDrive\Desktop\gait anlaysis\forensic-gait-analysis\runs\pose\predict\labels"
    folder2 = r"C:\Users\pavan\OneDrive\Desktop\gait anlaysis\forensic-gait-analysis\runs\pose\predict2\labels"
    
    try:
        similarity_score, metrics = main(folder1, folder2)
    except Exception as e:
        print(f"\nProgram terminated with error: {str(e)}")