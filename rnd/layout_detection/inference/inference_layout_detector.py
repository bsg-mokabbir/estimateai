import os
from pathlib import Path
from ultralytics import YOLO
import cv2
import numpy as np
from typing import List, Tuple, Dict
import torch

class TiledInference:
    def __init__(self, model_path: str, tile_size: int = 1280, overlap: float = 0.00, conf_threshold: float = 0.25, nms_threshold: float = 0.3):
        """
        Initialize tiled inference for large images
        
        Args:
            model_path: Path to the trained YOLO model
            tile_size: Size of each tile (larger for better small object detection)
            overlap: Overlap ratio between tiles (0.0 to 1.0) - reduced for less duplicates
            conf_threshold: Confidence threshold for detections
            nms_threshold: NMS threshold for removing duplicate detections - stricter
        """
        self.model = YOLO(model_path)
        self.tile_size = tile_size
        self.overlap = overlap
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.stride = int(tile_size * (1 - overlap))
        
    def create_tiles(self, image: np.ndarray) -> List[Dict]:
        """
        Create overlapping tiles from the input image
        
        Args:
            image: Input image array
            
        Returns:
            List of dictionaries containing tile information
        """
        height, width = image.shape[:2]
        tiles = []
        
        # Calculate the number of tiles needed
        y_positions = list(range(0, height - self.tile_size + 1, self.stride))
        x_positions = list(range(0, width - self.tile_size + 1, self.stride))
        
        # Handle case where image is smaller than tile size
        if not y_positions:
            y_positions = [0]
        if not x_positions:
            x_positions = [0]
        
        # Add the last positions if they don't align perfectly
        if len(y_positions) > 0 and y_positions[-1] + self.tile_size < height:
            y_positions.append(height - self.tile_size)
        if len(x_positions) > 0 and x_positions[-1] + self.tile_size < width:
            x_positions.append(width - self.tile_size)
        
        for y in y_positions:
            for x in x_positions:
                # Ensure we don't go out of bounds
                y_end = min(y + self.tile_size, height)
                x_end = min(x + self.tile_size, width)
                y_start = max(0, y_end - self.tile_size)
                x_start = max(0, x_end - self.tile_size)
                
                tile = image[y_start:y_end, x_start:x_end]
                
                tiles.append({
                    'tile': tile,
                    'x_offset': x_start,
                    'y_offset': y_start,
                    'original_size': (width, height)
                })
        
        return tiles
    
    def process_tile(self, tile_info: Dict) -> List[Dict]:
        """
        Process a single tile and return detections
        
        Args:
            tile_info: Dictionary containing tile information
            
        Returns:
            List of detection dictionaries
        """
        tile = tile_info['tile']
        x_offset = tile_info['x_offset']
        y_offset = tile_info['y_offset']
        
        # Run inference on the tile
        results = self.model(tile, conf=self.conf_threshold, verbose=False)
        
        detections = []
        if results[0].boxes is not None:
            for box in results[0].boxes:
                # Convert to global coordinates
                xyxy = box.xyxy[0].cpu().numpy()
                xyxy[0] += x_offset  # x1
                xyxy[1] += y_offset  # y1
                xyxy[2] += x_offset  # x2
                xyxy[3] += y_offset  # y2
                
                detection = {
                    'bbox': xyxy,
                    'confidence': float(box.conf[0]),
                    'class_id': int(box.cls[0]),
                    'class_name': self.model.names[int(box.cls[0])]
                }
                detections.append(detection)
        
        return detections
    
    def calculate_iou(self, box1: np.ndarray, box2: np.ndarray) -> float:
        """
        Calculate Intersection over Union (IoU) between two boxes
        
        Args:
            box1: First bounding box [x1, y1, x2, y2]
            box2: Second bounding box [x1, y1, x2, y2]
            
        Returns:
            IoU value
        """
        # Calculate intersection area
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        
        # Calculate union area
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def apply_nms(self, detections: List[Dict]) -> List[Dict]:
        """
        Apply Non-Maximum Suppression to remove duplicate detections
        with improved overlap handling
        
        Args:
            detections: List of detection dictionaries
            
        Returns:
            Filtered list of detections
        """
        if not detections:
            return []
        
        # Group detections by class
        class_detections = {}
        for det in detections:
            class_id = det['class_id']
            if class_id not in class_detections:
                class_detections[class_id] = []
            class_detections[class_id].append(det)
        
        filtered_detections = []
        
        # Apply NMS for each class separately
        for class_id, class_dets in class_detections.items():
            if not class_dets:
                continue
            
            # Sort by confidence score (descending)
            class_dets.sort(key=lambda x: x['confidence'], reverse=True)
            
            # Custom NMS implementation with stricter IoU threshold
            keep_detections = []
            
            for i, det in enumerate(class_dets):
                should_keep = True
                
                for kept_det in keep_detections:
                    iou = self.calculate_iou(det['bbox'], kept_det['bbox'])
                    
                    # Use stricter threshold for similar confidence scores
                    if det['confidence'] > 0.8 and kept_det['confidence'] > 0.8:
                        threshold = 0.3  # Stricter for high confidence
                    else:
                        threshold = self.nms_threshold
                    
                    if iou > threshold:
                        should_keep = False
                        break
                
                if should_keep:
                    keep_detections.append(det)
            
            filtered_detections.extend(keep_detections)
        
        return filtered_detections
    
    def post_process_detections(self, detections: List[Dict]) -> List[Dict]:
        """
        Additional post-processing to remove remaining duplicates
        
        Args:
            detections: List of detection dictionaries
            
        Returns:
            Further filtered detections
        """
        if not detections:
            return []
        
        # Sort by confidence (highest first)
        detections.sort(key=lambda x: x['confidence'], reverse=True)
        
        final_detections = []
        
        for det in detections:
            is_duplicate = False
            
            for existing_det in final_detections:
                # Check if same class
                if det['class_id'] == existing_det['class_id']:
                    iou = self.calculate_iou(det['bbox'], existing_det['bbox'])
                    
                    # Very strict duplicate removal
                    if iou > 0.5:  # 50% overlap threshold
                        is_duplicate = True
                        break
                    
                    # Check for nearly identical boxes (center distance)
                    det_center = [(det['bbox'][0] + det['bbox'][2])/2, (det['bbox'][1] + det['bbox'][3])/2]
                    exist_center = [(existing_det['bbox'][0] + existing_det['bbox'][2])/2, 
                                   (existing_det['bbox'][1] + existing_det['bbox'][3])/2]
                    
                    center_distance = np.sqrt((det_center[0] - exist_center[0])**2 + 
                                            (det_center[1] - exist_center[1])**2)
                    
                    avg_size = (np.sqrt((det['bbox'][2] - det['bbox'][0])**2 + (det['bbox'][3] - det['bbox'][1])**2) + 
                               np.sqrt((existing_det['bbox'][2] - existing_det['bbox'][0])**2 + 
                                      (existing_det['bbox'][3] - existing_det['bbox'][1])**2)) / 2
                    
                    # If centers are very close relative to object size
                    if center_distance < avg_size * 0.3:
                        is_duplicate = True
                        break
            
            if not is_duplicate:
                final_detections.append(det)
        
        return final_detections
    
    def auto_adjust_tile_size(self, image: np.ndarray, min_object_size: int = 20) -> int:
        """
        Automatically adjust tile size based on image dimensions and expected object size
        
        Args:
            image: Input image
            min_object_size: Minimum expected object size in pixels
            
        Returns:
            Recommended tile size
        """
        height, width = image.shape[:2]
        
        # Calculate scale factor compared to training size (640)
        scale_factor = max(width, height) / 640
        
        # Recommended tile sizes based on image size
        if scale_factor > 20:  # Very large images (>12800px)
            return 2048
        elif scale_factor > 10:  # Large images (>6400px)
            return 1536
        elif scale_factor > 5:   # Medium-large images (>3200px)
            return 1280
        else:
            return 640
    
    def simple_inference(self, image: np.ndarray) -> List[Dict]:
        """
        Simple inference for smaller images without tiling
        
        Args:
            image: Input image array
            
        Returns:
            List of detections
        """
        print("Using simple inference for smaller image")
        results = self.model(image, conf=self.conf_threshold, verbose=False)
        detections = []
        
        if results[0].boxes is not None:
            for box in results[0].boxes:
                detection = {
                    'bbox': box.xyxy[0].cpu().numpy(),
                    'confidence': float(box.conf[0]),
                    'class_id': int(box.cls[0]),
                    'class_name': self.model.names[int(box.cls[0])]
                }
                detections.append(detection)
        
        return detections
    
    def predict_with_multiple_scales(self, image: np.ndarray, tile_sizes: List[int] = None) -> List[Dict]:
        """
        Run inference with multiple tile sizes and combine results
        Check image size first - if smaller than 1500x1500, use simple inference
        
        Args:
            image: Input image
            tile_sizes: List of tile sizes to try
            
        Returns:
            Combined detections from all scales
        """
        height, width = image.shape[:2]
        
        # If image is smaller than 1500x1500, use simple inference
        if height < 1500 and width < 1500:
            return self.simple_inference(image)
        
        # For larger images, use tiled inference with multiple scales
        if tile_sizes is None:
            # Auto-determine tile sizes
            base_size = self.auto_adjust_tile_size(image)
            tile_sizes = [base_size, int(base_size * 1.5), int(base_size * 0.75)]
        
        all_detections = []
        
        for tile_size in tile_sizes:
            print(f"Processing with tile size: {tile_size}")
            
            # Temporarily change tile size
            original_tile_size = self.tile_size
            original_stride = self.stride
            
            self.tile_size = tile_size
            self.stride = int(tile_size * (1 - self.overlap))
            
            # Get detections for this tile size
            detections = self.predict_large_image(image)
            all_detections.extend(detections)
            
            # Restore original values
            self.tile_size = original_tile_size
            self.stride = original_stride
        
        # Apply NMS on combined results
        final_detections = self.apply_nms(all_detections)
        final_detections = self.post_process_detections(final_detections)
        return final_detections
    
    def predict_large_image(self, image: np.ndarray) -> List[Dict]:
        """
        Perform tiled inference on a large image
        
        Args:
            image: Input image array
            
        Returns:
            List of final detections after NMS
        """
        height, width = image.shape[:2]
        
        # If image is smaller than tile size, process normally
        if height <= self.tile_size and width <= self.tile_size:
            results = self.model(image, conf=self.conf_threshold, verbose=False)
            detections = []
            if results[0].boxes is not None:
                for box in results[0].boxes:
                    detection = {
                        'bbox': box.xyxy[0].cpu().numpy(),
                        'confidence': float(box.conf[0]),
                        'class_id': int(box.cls[0]),
                        'class_name': self.model.names[int(box.cls[0])]
                    }
                    detections.append(detection)
            return detections
        
        # Create tiles
        tiles = self.create_tiles(image)
        print(f"Created {len(tiles)} tiles for image of size {width}x{height}")
        
        # Process each tile
        all_detections = []
        for i, tile_info in enumerate(tiles):
            tile_detections = self.process_tile(tile_info)
            all_detections.extend(tile_detections)
            if i % 10 == 0:  # Progress update
                print(f"Processed {i+1}/{len(tiles)} tiles")
        
        print(f"Total detections before NMS: {len(all_detections)}")
        
        # Apply NMS to remove duplicates
        final_detections = self.apply_nms(all_detections)
        
        # Additional post-processing for stubborn duplicates
        final_detections = self.post_process_detections(final_detections)
        
        print(f"Final detections after post-processing: {len(final_detections)}")
        
        return final_detections
    
    def draw_detections(self, image: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """
        Draw detection boxes on the image
        
        Args:
            image: Input image
            detections: List of detection dictionaries
            
        Returns:
            Image with drawn detections
        """
        img_copy = image.copy()
        
        for det in detections:
            bbox = det['bbox'].astype(int)
            confidence = det['confidence']
            class_name = det['class_name']
            
            # Draw bounding box
            color = (0, 255, 0)  # Green
            cv2.rectangle(img_copy, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            
            # Draw label
            label = f"{class_name} {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            cv2.rectangle(img_copy, (bbox[0], bbox[1] - label_size[1] - 10), 
                         (bbox[0] + label_size[0], bbox[1]), color, -1)
            cv2.putText(img_copy, label, (bbox[0], bbox[1] - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return img_copy


# Configuration for different scenarios
CONFIGS = {
    'high_precision': {
        'tile_size': 1280,
        'overlap': 0.0,
        'conf_threshold': 0.3,
        'nms_threshold': 0.25
    },
    'balanced': {
        'tile_size': 1280,
        'overlap': 0.0,
        'conf_threshold': 0.25,
        'nms_threshold': 0.3
    },
    'high_recall': {
        'tile_size': 1536,
        'overlap': 0.0,
        'conf_threshold': 0.2,
        'nms_threshold': 0.35
    }
}

def create_tiled_inference(model_path: str, config_name: str = 'balanced'):
    """
    Create TiledInference with predefined configuration
    
    Args:
        model_path: Path to model
        config_name: 'high_precision', 'balanced', or 'high_recall'
    """
    config = CONFIGS[config_name]
    return TiledInference(
        model_path=model_path,
        tile_size=config['tile_size'],
        overlap=config['overlap'],
        conf_threshold=config['conf_threshold'],
        nms_threshold=config['nms_threshold']
    )


def main():
    # Configuration
    model_path = r"best_iteration_2.1.pt"
    input_folder = Path("test_input")
    output_folder = Path("test_outpu")
    output_folder.mkdir(parents=True, exist_ok=True)
    
    # Initialize with balanced config (less overlapping)
    tiled_inference = create_tiled_inference(model_path, 'high_precision')
    
    # Alternative: Manual initialization
    tiled_inference = TiledInference(
         model_path=model_path,
         tile_size=1280,
         overlap=0.2,     # Further reduced overlap
         conf_threshold=0.85, 
         nms_threshold=0.25  # Very strict NMS
     )
    
    # Supported image extensions
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    
    # Process each image
    for image_path in input_folder.glob("*"):
        if image_path.suffix.lower() not in image_extensions:
            continue
        
        print(f"\nProcessing: {image_path.name}")
        
        # Load image
        img = cv2.imread(str(image_path))
        if img is None:
            print(f"Failed to load image: {image_path}")
            continue
        
        height, width = img.shape[:2]
        print(f"Image size: {width}x{height}")
        
        # Perform inference with automatic size-based processing
        detections = tiled_inference.predict_with_multiple_scales(img)
        
        # Draw detections
        result_img = tiled_inference.draw_detections(img, detections)
        
        # Save result
        output_path = output_folder / image_path.name
        cv2.imwrite(str(output_path), result_img)
        print(f"Saved: {output_path}")
        
        # Optionally save detection results as text file
        txt_path = output_folder / f"{image_path.stem}_detections.txt"
        with open(txt_path, 'w') as f:
            f.write(f"Image: {image_path.name}\n")
            f.write(f"Size: {width}x{height}\n")
            f.write(f"Total detections: {len(detections)}\n\n")
            
            for i, det in enumerate(detections):
                bbox = det['bbox']
                f.write(f"Detection {i+1}:\n")
                f.write(f"  Class: {det['class_name']}\n")
                f.write(f"  Confidence: {det['confidence']:.3f}\n")
                f.write(f"  Bbox: [{bbox[0]:.1f}, {bbox[1]:.1f}, {bbox[2]:.1f}, {bbox[3]:.1f}]\n\n")


if __name__ == "__main__":
    main()