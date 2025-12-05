#!/usr/bin/env python
# coding: utf-8

# In[2]:


# Install Packages
#pip install ultralytics
#pip install python-dotenv
#pip install opencv-python

# Libraries
import os
import time
from dotenv import load_dotenv
import numpy as np
import pandas as pd
import cv2
from PIL import Image
import torch
import torchvision.transforms as T
from ultralytics import YOLO
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn,
    FasterRCNN_ResNet50_FPN_Weights,
)

# Configuration Settings
PROJECT_ENV_FILE = "projectrootfolderpath.env" # environment file with folder paths
PROJECT_FOLDER_VAR = "PROJECT_FOLDER" # project folder path
IMAGE_FOLDER_VAR = "IMAGE_FOLDER" # image folder path
OUTPUT_CSV_FILE = "object_detection_comparison_results.csv"
# FILE_EXTS = () # If not specified, defaults (".jpg", ".jpeg", ".png", ".bmp")

# Helper Functions
def load_image_paths(folder_path, valid_exts=None):
    if folder_path is None:
        raise ValueError(f"ERROR: Image folder path is None.")
    if not os.path.isdir(folder_path):
        raise ValueError(f"ERROR: '{folder_path}' is not a valid directory.")
        
    # set default extension types
    if valid_exts is None: 
        valid_exts = (".jpg", ".jpeg", ".png", ".bmp") 

    # image files
    image_files = sorted(
        f for f in os.listdir(folder_path)
        if f.lower().endswith(valid_exts)
    )
    
    image_paths = [os.path.join(folder_path, f) for f in image_files]

    # summary info
    detected_exts = {os.path.splitext(f)[1].lower() for f in image_files}
    summary = {
        "count": len(image_files),
        "extensions": detected_exts,
        "filenames": image_files,
    }

    return image_files, image_paths, summary

def compute_entropy(gray_array: np.ndarray) -> float:
    """
    Compute Shannon entropy of a grayscale image.
    gray_array should be a 2D uint8 array (0–255).
    """
    # Histogram with 256 bins for 0–255
    hist, _ = np.histogram(gray_array.ravel(), bins=256, range=(0, 255))
    # Convert to probabilities
    p = hist.astype("float32")
    total = p.sum()
    if total == 0:
        return 0.0
    p /= total
    # Keep only non-zero probs to avoid log(0)
    p = p[p > 0]
    entropy = -np.sum(p * np.log2(p))
    return float(entropy)

def compute_edge_pixel_count(gray_array: np.ndarray,
                             low_threshold: int = 100,
                             high_threshold: int = 200) -> int:
    """
    Count edge pixels using Canny edge detection.
    gray_array should be a 2D uint8 array (0–255).
    """
    edges = cv2.Canny(gray_array, low_threshold, high_threshold)
    edge_count = int(np.count_nonzero(edges))
    return edge_count

def compute_colorfulness(rgb_array: np.ndarray) -> float:
    """
    Compute the Hasler–Süsstrunk colorfulness metric.
    rgb_array should be in shape (H, W, 3) with values in [0, 255].
    """
    # Separate channels
    R = rgb_array[:, :, 2].astype("float32")
    G = rgb_array[:, :, 1].astype("float32")
    B = rgb_array[:, :, 0].astype("float32")

    # R-G and Y-B components
    Rg = R - G
    Yb = 0.5 * (R + G) - B

    # Mean and standard deviation
    mean_rg = np.mean(Rg)
    mean_yb = np.mean(Yb)
    std_rg = np.std(Rg)
    std_yb = np.std(Yb)

    # Colorfulness formula
    std_root = np.sqrt(std_rg**2 + std_yb**2)
    mean_root = np.sqrt(mean_rg**2 + mean_yb**2)
    colorfulness = std_root + 0.3 * mean_root
    return float(colorfulness)

def extract_basic_image_features(image_path):
    """
    Compute simple image statistics that do NOT use deep learning:
    - width, height (pixels)
    - aspect ratio (width / height)
    - mean_brightness (0–1)
    - entropy (Shannon entropy of grayscale)
    - edge_pixel_count (Canny edge detector)
    - colorfulness (Hasler–Süsstrunk metric)
    """
    img = Image.open(image_path).convert("RGB")
    width, height = img.size

    # Convert to NumPy arrays
    rgb_array = np.asarray(img)               # shape (H, W, 3) in [0..255]
    norm_array = rgb_array.astype("float32") / 255.0

    # Brightness: mean over all pixels/channels (0–1)
    mean_brightness = float(norm_array.mean())

    # Aspect ratio
    aspect_ratio = width / height

    # Grayscale array for entropy and edges
    gray_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2GRAY)

    entropy = compute_entropy(gray_array)
    edge_pixel_count = compute_edge_pixel_count(gray_array)
    colorfulness = compute_colorfulness(rgb_array)

    return (width, height, aspect_ratio, mean_brightness, entropy, edge_pixel_count, colorfulness)

# convert image to tensor
to_tensor = T.ToTensor()

def run_yolov8_on_image(image_path, model, conf_threshold=0.25):
    """
    Run YOLOv8 on a single image.

    Returns:
    - elapsed_time (seconds)
    - num_detections (count of boxes)
    - avg_confidence (mean of box confidences, 0 if no detections)
    """
    start_time = time.time()

    # Let the model use its own internal device setting.
    results = model(
        source=image_path,
        conf=conf_threshold,
        verbose=False,
    )

    end_time = time.time()
    elapsed = end_time - start_time

    result = results[0]
    boxes = result.boxes

    num_detections = len(boxes)
    if num_detections > 0:
        confidences = boxes.conf.detach().cpu().numpy()
        avg_conf = float(confidences.mean())
    else:
        avg_conf = 0.0

    return elapsed, num_detections, avg_conf

def run_fasterrcnn_on_image(image_path, model, device, score_threshold=0.5):
    """
    Run Faster R-CNN on a single image.

    Returns:
    - elapsed_time (seconds)
    - num_detections (count of boxes with score >= threshold)
    - avg_score (mean score of kept detections, 0 if none)
    """
    img = Image.open(image_path).convert("RGB")
    img_tensor = to_tensor(img).to(device)

    start_time = time.time()
    with torch.no_grad():
        outputs = model([img_tensor])
    end_time = time.time()
    elapsed = end_time - start_time

    output = outputs[0]
    scores = output["scores"].detach().cpu().numpy()

    keep_mask = scores >= score_threshold
    num_detections = int(keep_mask.sum())

    if num_detections > 0:
        avg_score = float(scores[keep_mask].mean())
    else:
        avg_score = 0.0

    return elapsed, num_detections, avg_score

def main():
    # load environment variables
    load_dotenv(PROJECT_ENV_FILE)
    
    # get folder paths
    project_folder = os.getenv(PROJECT_FOLDER_VAR)
    if project_folder is None:
        raise ValueError(f"ERROR: Environment variable '{PROJECT_FOLDER_VAR}' not set in {PROJECT_ENV_FILE}")
    if not os.path.isdir(project_folder):
        raise ValueError(f"ERROR: '{project_folder}' is not a valid directory.")
        
    # get image folder paths
    image_folder = os.getenv(IMAGE_FOLDER_VAR)
    if image_folder is None:
        raise ValueError("Environment variable IMAGE_FOLDER is not set.")
    if not os.path.isdir(image_folder):
        raise ValueError(f"Path '{image_folder}' is not a valid directory.")
    
    #print(f"Project folder: {project_folder}")
    print(f"Project folder contains {len(os.listdir(project_folder))} files/directories")
    #print(f"Image folder:   {image_folder}")
    print(f"Image folder contains   {len(os.listdir(image_folder))} files")

    # load image paths
    image_files, image_paths, summary = load_image_paths(image_folder)
    if summary["count"] == 0:
        raise RuntimeError("No images found in the image folder.")
    
    #print("\nImage files found:")
    #for f in image_files:
    #    print(" -", f)
    #print(f"Total files: {len(image_files)}\n")
    
    print(f"Found {summary['count']} images")
    print("Extensions detected:", summary["extensions"])

    # Choose GPU if available, otherwise use CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    
    print(f"Using device: {device}")

    # Load the metadata for the pre-trained model
    yolo_weights_path = "yolov8m.pt" if device.type == "cuda" else "yolov8n.pt"
        
    # Load the pre-trained YOLO model (pick model size based on runtime capabilities
    yolo_model = YOLO(yolo_weights_path)
    yolo_model.to(device)    
    print("Loaded YOLO model:", yolo_weights_path)
    
    # Load the metadata for the pre-trained model
    frcnn_weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    
    # Load the pre-trained Faster R-CNN model
    fasterrcnn_model = fasterrcnn_resnet50_fpn(weights=frcnn_weights)
    fasterrcnn_model.to(device)
    fasterrcnn_model.eval()
    
    print("Loaded Faster R-CNN model with COCO weights")

    # run models on each image
    results_rows = []
    
    for img_name, img_path in zip(image_files, image_paths):
        print(f"\nProcessing {img_name} ...")
    
        # Non–deep-learning features
        (width, height, aspect_ratio, mean_brightness, entropy, edge_pixel_count, colorfulness) = extract_basic_image_features(img_path)
    
        # YOLOv8 detection
        yolo_time, yolo_count, yolo_avg_conf = run_yolov8_on_image(img_path, yolo_model, conf_threshold=0.25)        
        print(
            f"[YOLOv8]    time={yolo_time:.3f}s | "
            f"detections={yolo_count} | "
            f"avg_conf={yolo_avg_conf:.3f}"
        )
    
        # Faster R-CNN detection
        frcnn_time, frcnn_count, frcnn_avg_score = run_fasterrcnn_on_image(img_path, fasterrcnn_model, device=device, score_threshold=0.5)
        print(
            f"[FasterRCNN] time={frcnn_time:.3f}s | "
            f"detections={frcnn_count} | "
            f"avg_score={frcnn_avg_score:.3f}"
        )
    
        # Collect everything in a row for later tabular analysis
        results_rows.append(
            {
                "image": img_name,
                # Non–deep-learning features
                "width": width,
                "height": height,
                "aspect_ratio": aspect_ratio,
                "mean_brightness": mean_brightness,
                "entropy": entropy,
                "edge_pixel_count": edge_pixel_count,
                "colorfulness": colorfulness,
                # YOLO metrics
                "yolo_time_sec": yolo_time,
                "yolo_objects": yolo_count,
                "yolo_avg_conf": yolo_avg_conf,
                # Faster R-CNN metrics
                "frcnn_time_sec": frcnn_time,
                "frcnn_objects": frcnn_count,
                "frcnn_avg_score": frcnn_avg_score,
            }
        )

    # Build dataframe
    results_df = pd.DataFrame(results_rows)
    print(f"\nProcessed {len(results_df)} images.")
    results_df.info()

    # Save to CSV
    output_csv = os.path.join(project_folder, OUTPUT_CSV_FILE)
    results_df.to_csv(output_csv, index=False)    
    print(f"\nResults Saved as {OUTPUT_CSV_FILE} in project_folder")
    print("Script finished")

if __name__ == "__main__":
    main()

