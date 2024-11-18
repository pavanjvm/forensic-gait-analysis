# Forensic Gait Analysis

Forensic Gait Analysis is a tool that compares two video footages, one from a crime scene and another from a suspect, to determine if they depict the same person based on their gait patterns.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
  - [Step 1: Preprocess the Videos](#step-1-preprocess-the-videos)
  - [Step 2: Calculate Keypoints](#step-2-calculate-keypoints)
  - [Step 3: Calculate Gait Similarity](#step-3-calculate-gait-similarity)
- [Example](#example)
- [Contributing](#contributing)
- [License](#license)

## Installation

1. Clone this repository:

    ```bash
    git clone https://github.com/your-username/forensic-gait-analysis.git
    cd forensic-gait-analysis
    ```

2. Install the dependencies:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Step 1: Preprocess the Videos

1. Update the paths in `preprocess.py` to point to the crime scene and suspect videos.
2. Run the preprocessing script to convert each video to 20 fps and trim them to 6 seconds:

    ```bash
    python preprocess.py
    ```

### Step 2: Calculate Keypoints

1. Update the paths in `pose.py` to point to the processed videos.
2. Run the keypoint calculation script to generate keypoint data for each frame:

    ```bash
    python pose.py
    ```

3. **Example Output (Keypoints Visualization):**
   ![Keypoints Visualization](images/keypoints_example.png)

   This image represents the detected keypoints on a person in each frame. Keypoints such as joints, shoulders, and limbs are highlighted to capture the gait pattern.

### Step 3: Calculate Gait Similarity

1. Update the paths in `gait-similarity.py` to point to the folder containing the keypoint labels.
2. Run the gait similarity calculation and visualization script:

    ```bash
    python gait-similarity.py
    ```

3. **Example Output (Gait Similarity):**
   ![Gait Similarity Visualization](images/gait_similarity_example.png)

   The above visualization shows a comparison of the gait patterns from the crime scene and suspect videos. The similarity score reflects the degree of match.

## Example

After completing the steps above, the `gait-similarity.py` script will output a similarity score and visualizations to help identify if the two footages depict the same person based on gait analysis.


