import os
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from transformers import AutoImageProcessor, DeformableDetrForObjectDetection
from ultralytics import YOLO
import time
import json
from tabulate import tabulate
import cv2

# Paths
test_dir = r"C:\Users\pavan\Documents\Test_data"
results_dir = r"C:\Users\pavan\Documents\CV_environment\.venv\CV_Project3\obj_detection\Results_Final1"
os.makedirs(results_dir, exist_ok=True)

# Set confidence threshold to be the same for both models
CONFIDENCE_THRESHOLD = 0.6
# Load models
# DETR model
detr_processor = AutoImageProcessor.from_pretrained("SenseTime/deformable-detr")
detr_model = DeformableDetrForObjectDetection.from_pretrained("SenseTime/deformable-detr")

# YOLOv8 model
yolo_model = YOLO("yolov8x.pt")

# Results dictionary
all_results = {}
# Table data for comparison
table_data = []
table_headers = ["Image", "Model", "Inference Time (s)", "FPS", "Detections", "Memory (MB)", "Avg Confidence"]

# Process each image
image_files = [f for f in os.listdir(test_dir) if f.endswith(('.jpg', '.jpeg', '.png')) and not f.startswith('.')]

for img_file in image_files:
    img_path = os.path.join(test_dir, img_file)
    img = Image.open(img_path)
    
    # Store results for this image
    img_results = {"detr": {}, "yolo": {}}
    
    # Convert to RGB if needed (ensure same preprocessing)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # For DETR, explicitly print preprocessing details
    print("DETR preprocessing details:", detr_processor)

    # For YOLO, we can try to match some preprocessing parameters
    # For example, set the same image size for both models
    img_size = 640  # Common size used in object detection
    
    # ---- DETR Processing ----
    # Reset GPU memory stats
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    
    start_time = time.time()
    # Process image with DETR
    inputs = detr_processor(images=img, return_tensors="pt")
    
    # Move to GPU if available
    if torch.cuda.is_available():
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
        detr_model = detr_model.to("cuda")
    
    # Run inference
    with torch.no_grad():
        outputs = detr_model(**inputs)
    
    # Post-process
    target_sizes = torch.tensor([img.size[::-1]])
    if torch.cuda.is_available():
        target_sizes = target_sizes.to("cuda")
    
    detr_results = detr_processor.post_process_object_detection(
        outputs, target_sizes=target_sizes, threshold=CONFIDENCE_THRESHOLD
    )[0]
    
    detr_end_time = time.time()
    
    # Calculate DETR metrics
    detr_inference_time = detr_end_time - start_time
    detr_fps = 1 / detr_inference_time
    detr_num_detections = len(detr_results["scores"])
    detr_avg_confidence = np.mean([float(s) for s in detr_results["scores"]]) if detr_num_detections > 0 else 0
    
    # Get memory usage for DETR
    if torch.cuda.is_available():
        detr_memory_allocated = torch.cuda.memory_allocated() / (1024 * 1024)  # MB
        detr_max_memory = torch.cuda.max_memory_allocated() / (1024 * 1024)  # MB
    else:
        detr_memory_allocated = 0
        detr_max_memory = 0
    
    img_results["detr"] = {
        "inference_time": detr_inference_time,
        "fps": detr_fps,
        "num_detections": detr_num_detections,
        "memory_allocated_mb": detr_memory_allocated,
        "max_memory_mb": detr_max_memory,
        "avg_confidence": detr_avg_confidence,
        "scores": [float(s) for s in detr_results["scores"]],
        "labels": [int(l) for l in detr_results["labels"]],
        "boxes": [[float(c) for c in box] for box in detr_results["boxes"]]
    }
    
    # Add to table data
    table_data.append([
        img_file, 
        "DETR", 
        f"{detr_inference_time:.4f}", 
        f"{detr_fps:.2f}", 
        detr_num_detections, 
        f"{detr_max_memory:.2f}",
        f"{detr_avg_confidence:.4f}" if detr_num_detections > 0 else "N/A"
    ])
    
    # ---- YOLOv8 Processing ----
    # Reset GPU memory stats
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    
    start_time = time.time()
    # Run YOLOv8 with the same confidence threshold and image size
    yolo_results = yolo_model(img, conf=CONFIDENCE_THRESHOLD, imgsz=img_size)
    yolo_end_time = time.time()
    
    # Calculate YOLO metrics
    yolo_inference_time = yolo_end_time - start_time
    yolo_fps = 1 / yolo_inference_time
    yolo_num_detections = len(yolo_results[0].boxes)
    yolo_avg_confidence = np.mean([float(s) for s in yolo_results[0].boxes.conf]) if yolo_num_detections > 0 else 0
    
    # Get memory usage for YOLO
    if torch.cuda.is_available():
        yolo_memory_allocated = torch.cuda.memory_allocated() / (1024 * 1024)  # MB
        yolo_max_memory = torch.cuda.max_memory_allocated() / (1024 * 1024)  # MB
    else:
        yolo_memory_allocated = 0
        yolo_max_memory = 0
    
    img_results["yolo"] = {
        "inference_time": yolo_inference_time,
        "fps": yolo_fps,
        "num_detections": yolo_num_detections,
        "memory_allocated_mb": yolo_memory_allocated,
        "max_memory_mb": yolo_max_memory,
        "avg_confidence": yolo_avg_confidence,
        "scores": [float(s) for s in yolo_results[0].boxes.conf],
        "labels": [int(s) for s in yolo_results[0].boxes.cls],
        "boxes": [[float(x) for x in box] for box in yolo_results[0].boxes.xyxy.tolist()]
    }
    
    # Add to table data
    table_data.append([
        img_file, 
        "YOLOv8", 
        f"{yolo_inference_time:.4f}", 
        f"{yolo_fps:.2f}", 
        yolo_num_detections, 
        f"{yolo_max_memory:.2f}",
        f"{yolo_avg_confidence:.4f}" if yolo_num_detections > 0 else "N/A"
    ])
    
    # Create annotated images with metrics
    # DETR image
    detr_img = img.copy()
    draw = ImageDraw.Draw(detr_img)
    try:
        font = ImageFont.truetype("arial.ttf", 15)
    except:
        font = ImageFont.load_default()
    
    # Add metrics text to image including memory usage
    metrics_text = f"DETR - Time: {detr_inference_time:.3f}s, FPS: {detr_fps:.1f}, Memory: {detr_max_memory:.1f}MB, Detections: {detr_num_detections}"
    draw.text((10, 10), metrics_text, fill=(255, 0, 0), font=font)
    
    # Draw boxes with improved visibility
    for score, label, box in zip(detr_results["scores"], detr_results["labels"], detr_results["boxes"]):
        box = [round(i) for i in box.tolist()]
        # Draw rectangle with thicker width
        draw.rectangle(box, outline=(255, 0, 0), width=3)
        
        # Add background for text
        label_text = f"{detr_model.config.id2label[label.item()]}: {score:.2f}"
        text_width, text_height = draw.textsize(label_text, font=font) if hasattr(draw, 'textsize') else (len(label_text)*8, 15)
        draw.rectangle((box[0], box[1]-20, box[0]+text_width, box[1]), fill=(255, 255, 0))
        draw.text((box[0], box[1]-15), label_text, fill=(0, 0, 0), font=font)
    
    # Save DETR result
    detr_img.save(os.path.join(results_dir, f"detr_{img_file}"))
    
    # YOLO image
    # yolo_img = Image.fromarray(yolo_results[0].plot())
    # draw = ImageDraw.Draw(yolo_img)
    
    yolo_img_array = yolo_results[0].plot()
    # Convert BGR to RGB if needed
    if hasattr(yolo_results[0], 'plot_format') and yolo_results[0].plot_format == "bgr":
        yolo_img_array = cv2.cvtColor(yolo_img_array, cv2.COLOR_BGR2RGB)
    yolo_img = Image.fromarray(yolo_img_array)
    draw = ImageDraw.Draw(yolo_img)
    
    # Add metrics text to image including memory usage
    metrics_text = f"YOLOv8 - Time: {yolo_inference_time:.3f}s, FPS: {yolo_fps:.1f}, Memory: {yolo_max_memory:.1f}MB, Detections: {yolo_num_detections}"
    draw.text((10, 10), metrics_text, fill=(0, 0, 255), font=font)
    
    # Save YOLO result
    yolo_img.save(os.path.join(results_dir, f"yolo_{img_file}"))
    
    # Store results
    all_results[img_file] = img_results
    
    print(f"Processed {img_file}")

# Process video if needed
video_file = "Video_1.mp4"
if os.path.exists(os.path.join(test_dir, video_file)):
    # Video processing code would go here
    # For videos, you'd need to process frame by frame
    pass

# Save all metrics to a JSON file
with open(os.path.join(results_dir, "detection_metrics.json"), "w") as f:
    json.dump(all_results, f, indent=4)

# Generate summary report
summary = {
    "detr": {
        "avg_inference_time": np.mean([all_results[img]["detr"]["inference_time"] for img in all_results]),
        "avg_fps": np.mean([all_results[img]["detr"]["fps"] for img in all_results]),
        "avg_detections": np.mean([all_results[img]["detr"]["num_detections"] for img in all_results]),
        "avg_memory_mb": np.mean([all_results[img]["detr"]["max_memory_mb"] for img in all_results]),
        "avg_confidence": np.mean([all_results[img]["detr"]["avg_confidence"] for img in all_results if all_results[img]["detr"]["num_detections"] > 0])
    },
    "yolo": {
        "avg_inference_time": np.mean([all_results[img]["yolo"]["inference_time"] for img in all_results]),
        "avg_fps": np.mean([all_results[img]["yolo"]["fps"] for img in all_results]),
        "avg_detections": np.mean([all_results[img]["yolo"]["num_detections"] for img in all_results]),
        "avg_memory_mb": np.mean([all_results[img]["yolo"]["max_memory_mb"] for img in all_results]),
        "avg_confidence": np.mean([all_results[img]["yolo"]["avg_confidence"] for img in all_results if all_results[img]["yolo"]["num_detections"] > 0])
    }
}

with open(os.path.join(results_dir, "summary_metrics.json"), "w") as f:
    json.dump(summary, f, indent=4)

# Create per-image comparison table
table_str = tabulate(table_data, headers=table_headers, tablefmt="grid")
with open(os.path.join(results_dir, "metrics_comparison_table.txt"), "w") as f:
    f.write(table_str)
print("\nMetrics Comparison Table:")
print(table_str)

# Create metrics calculation explanation table
metrics_explanation = [
    ["Inference Time (s)", "End time - Start time", "Measures the total time taken for model inference"],
    ["FPS", "1 / Inference Time", "Frames per second - higher is better for real-time applications"],
    ["Detections", "len(results['scores'])", "Number of objects detected above confidence threshold"],
    ["Memory (MB)", "torch.cuda.max_memory_allocated() / (1024 * 1024)", "Peak GPU memory usage during inference"],
    ["Avg Confidence", "mean(results['scores'])", "Average confidence score of all detections"]
]
explanation_table = tabulate(metrics_explanation, headers=["Metric", "Calculation", "Description"], tablefmt="grid")
with open(os.path.join(results_dir, "metrics_explanation.txt"), "w") as f:
    f.write(explanation_table)
print("\nMetrics Calculation Explanation:")
print(explanation_table)

# Create summary comparison table
summary_data = [
    ["Average Inference Time (s)", f"{summary['detr']['avg_inference_time']:.4f}", f"{summary['yolo']['avg_inference_time']:.4f}"],
    ["Average FPS", f"{summary['detr']['avg_fps']:.2f}", f"{summary['yolo']['avg_fps']:.2f}"],
    ["Average Detections", f"{summary['detr']['avg_detections']:.2f}", f"{summary['yolo']['avg_detections']:.2f}"],
    ["Average Memory (MB)", f"{summary['detr']['avg_memory_mb']:.2f}", f"{summary['yolo']['avg_memory_mb']:.2f}"],
    ["Average Confidence", f"{summary['detr']['avg_confidence']:.4f}", f"{summary['yolo']['avg_confidence']:.4f}"]
]
summary_table = tabulate(summary_data, headers=["Metric", "DETR", "YOLOv8"], tablefmt="grid")
with open(os.path.join(results_dir, "summary_comparison_table.txt"), "w") as f:
    f.write(summary_table)
print("\nSummary Comparison Table:")
print(summary_table)

print("\nTesting complete. Results saved to", results_dir)
