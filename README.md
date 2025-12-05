# Object Detection Comparison

## MSBA 503 — Take-Home Assignment

This repository contains a structured experiment comparing two modern object detection architectures, YOLOv8 and Faster R-CNN (ResNet-50 FPN), across a curated image set.
The purpose is not simply to run two models, but to analyze how architecture, speed, confidence, and image characteristics interact, generating insights suitable for Part A of the assignment.

## 1. Purpose of the Project
The assignment requires comparing at least two deep learning algorithms in terms of:
* inference time
* number of objects detected
* confidence/probability scores
This implementation extends the requirement by also extracting non-deep-learning image features (entropy, edge count, brightness, colorfulness), enabling a more analytical interpretation of why models behave differently across images.

## 2. How This Repository Adds Analytical Value
### 2.1 Understanding Architectural Differences Through Behavior
YOLOv8 (a one-stage detector) and Faster R-CNN (a two-stage detector) prioritize different trade-offs:
* YOLOv8 aims for speed and broad coverage.
* Faster R-CNN favors selective, proposal-based detection.
By measuring inference time, counts, and confidence in parallel, the experiment reveals how these architectural choices manifest in practice—something not visible by reading model descriptions alone.

### 2.2 Linking Image Characteristics to Model Performance
The script computes classical image statistics, allowing comments such as:
* Higher entropy and edge count images often generate more detections for both models.
* Low-brightness images may reduce Faster R-CNN scores more than YOLO’s.
* Strong color separation sometimes leads YOLO to detect more small objects.
These relationships provide meaningful insights for the written assignment without writing overly long explanations.

### 2.3 Reproducible and Configurable Design
The experiment is built around a .env file so the user can change dataset locations without touching code:

`PROJECT_FOLDER=path/to/project`
`IMAGE_FOLDER=path/to/images`

This makes the project easier to review and re-run on new images—a requirement for good data science practice and a benefit when an instructor tests the script.

## 3. Workflow Overview
### 3.1 Processing Steps

1. Read project and image folder paths from `.env`.
2. Validate directory structure.
3. Load image files with supported extensions.
4. Extract non-deep-learning features from each image:
    * entropy
    * edge pixel count
    * colorfulness
    * brightness
    * aspect ratio
5. Run YOLOv8 on each image.
6. Run Faster R-CNN on each image.
7. Collect and store results in a single CSV file.

## 4. Running the Experiment
### 4.1 Install Required Packages
`pip install ultralytics python-dotenv opencv-python`
`pip install torch torchvision pandas numpy pillow`

### 4.2 Create Environment File
Create:
`projectrootfolderpath.env`
Add:
`PROJECT_FOLDER=C:/path/to/project`
`IMAGE_FOLDER=C:/path/to/images`

### 4.3 Run the Script
`python object_detection_script.py`

Output file created:
`object_detection_comparison_results.csv`

## 5. Acknowledgments
Some structural improvements and feature ideas were assisted by ChatGPT.
Help was provided by Brandon Christenson for guidance, and I liked his use of helper functions, so I wanted to implement that.

## 6. Versions of This Project (Assignment Notebook vs. Script Version)

This repository contains two functional versions of the object detection workflow:
#### 1. Notebook Version (for the MSBA 503 assignment)
  Designed to show intermediate outputs, printed debugging logs, and explanations suitable for an academic submission. This version emphasizes transparency and readability.

#### 2. Standalone Script Version (object_detection_script.py)
  A production-style version wrapped in main(), intended for clean execution, portability, and reusability on new image sets. It removes verbose notebook-style output and uses environment variables for configuration.

Both versions perform the same core analysis but differ in presentation and execution style depending on the intended audience.
