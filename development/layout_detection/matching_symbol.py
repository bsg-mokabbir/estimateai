import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
import json
from skimage.metrics import structural_similarity as ssim
from scipy import ndimage

class EnhancedSymbolMatcher:
    def __init__(self, symbol_folder: str):
        """Enhanced symbol matcher with multiple matching techniques"""
        
        # Initialize ORB detector with tuned parameters for better sensitivity.
        self.orb = cv2.ORB_create(
            nfeatures=500,
            scaleFactor=1.2,
            nlevels=8,
            edgeThreshold=10,
            firstLevel=0,
            scoreType=cv2.ORB_HARRIS_SCORE,
            patchSize=31
        )

        # Initialize Brute-Force matcher (used in new feature matching method)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        self.symbol_folder = Path(symbol_folder)
        self.symbols = self.load_and_preprocess_symbols()
        print(f"Loaded {len(self.symbols)} reference symbols with multiple features")
    
    def common_preprocessing(self, img: np.ndarray, target_size=(64, 64)) -> np.ndarray:
        """
        SAME preprocessing function for both reference symbols and detected regions
        This ensures consistent processing
        """
        # Convert to grayscale if needed - with proper channel checking
        if len(img.shape) == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        elif len(img.shape) == 3 and img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        elif len(img.shape) == 2:
            # Already grayscale, keep as is
            pass
        else:
            print(f"Warning: Unexpected image shape {img.shape}")
            # If it's already 2D, keep as is
            if len(img.shape) > 2:
                img = img[:, :, 0]  # Take first channel
        
        # Resize to standard size
        img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
        
        # Denoise before other ops
        img = cv2.bilateralFilter(img, 9, 75, 75)
        
        # Binary thresholding for simple symbols
        _, img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        
        return img
    
    def load_and_preprocess_symbols(self) -> Dict[str, Dict]:
        """Load symbols and extract all features"""
        symbols = {}
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        
        for symbol_path in self.symbol_folder.glob("*"):
            if symbol_path.suffix.lower() in image_extensions:
                # Try loading as color first, then grayscale if that fails
                symbol_img = cv2.imread(str(symbol_path), cv2.IMREAD_COLOR)
                if symbol_img is None:
                    symbol_img = cv2.imread(str(symbol_path), cv2.IMREAD_GRAYSCALE)
                
                if symbol_img is not None:
                    try:
                        # Use SAME preprocessing function
                        symbol_clean = self.common_preprocessing(symbol_img, target_size=(128, 128))
                        
                        # Extract features
                        keypoints, descriptors = self.orb.detectAndCompute(symbol_clean, None)
                        
                        symbols[symbol_path.stem] = {
                            'image': symbol_clean,
                            'edges': cv2.Canny(symbol_clean, 50, 150),
                            'keypoints': keypoints,
                            'descriptors': descriptors,
                            'moments': cv2.moments(symbol_clean),
                            'contour_features': self.extract_contour_features(symbol_clean)
                        }
                        print(f"Loaded: {symbol_path.stem}")
                    except Exception as e:
                        print(f"Error processing {symbol_path.name}: {e}")
                        continue
                else:
                    print(f"Could not load: {symbol_path.name}")
        return symbols
    
    def preprocess_detected_region(self, region: np.ndarray) -> np.ndarray:
        """Enhanced preprocessing for detected regions - USES SAME FUNCTION"""
        
        # Handle very small regions first
        h, w = region.shape[:2]
        if h < 20 or w < 20:
            region = cv2.resize(region, (max(128, w*3), max(128, h*3)), interpolation=cv2.INTER_CUBIC)
        
        # Use the EXACT SAME preprocessing function as reference symbols
        region = self.common_preprocessing(region, target_size=(128, 128))
        
        return region
    
    def extract_contour_features(self, img: np.ndarray) -> Dict:
        """Extract contour-based features"""
        contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return {'area': 0, 'perimeter': 0, 'circularity': 0, 'aspect_ratio': 1.0}
        
        main_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(main_contour)
        perimeter = cv2.arcLength(main_contour, True)
        circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
        
        # Bounding rectangle for aspect ratio
        x, y, w, h = cv2.boundingRect(main_contour)
        aspect_ratio = float(w) / h if h > 0 else 1.0
        
        return {
            'area': area, 
            'perimeter': perimeter, 
            'circularity': circularity,
            'aspect_ratio': aspect_ratio
        }
    
    def rotate_image(self, img: np.ndarray, angle: int) -> np.ndarray:
        """Rotate image by specified angle (0, 90, 180, 270)"""
        if angle == 0:
            return img
        elif angle == 90:
            return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        elif angle == 180:
            return cv2.rotate(img, cv2.ROTATE_180)
        elif angle == 270:
            return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        else:
            return img
    
    def template_matching_score_with_rotation(self, detected: np.ndarray, symbol: np.ndarray) -> Tuple[float, int]:
        """Enhanced template matching with 4-way rotation"""
        best_score = 0.0
        best_angle = 0
        
        # Test all 4 rotations: 0°, 90°, 180°, 270°
        angles = [0, 90, 180, 270]
        methods = [cv2.TM_CCOEFF_NORMED, cv2.TM_CCORR_NORMED, cv2.TM_SQDIFF_NORMED]
        
        for angle in angles:
            rotated_detected = self.rotate_image(detected, angle)
            
            scores = []
            for method in methods:
                try:
                    result = cv2.matchTemplate(rotated_detected, symbol, method)
                    score = cv2.minMaxLoc(result)[1] if method != cv2.TM_SQDIFF_NORMED else (1 - cv2.minMaxLoc(result)[0])
                    scores.append(max(0, score))
                except:
                    scores.append(0)
            
            avg_score = np.mean(scores)
            if avg_score > best_score:
                best_score = avg_score
                best_angle = angle
        
        return best_score, best_angle
    
    def feature_matching_score(self, detected: np.ndarray, symbol_data: Dict) -> float:
        """ORB feature matching without rotation"""
        desc_ref = symbol_data['descriptors']
        
        if desc_ref is None or len(desc_ref) < 10:
            return 0.0
        
        kp_query, desc_query = self.orb.detectAndCompute(detected, None)
        
        if desc_query is None or len(desc_query) < 10:
            return 0.0
            
        try:
            # Match features using BFMatcher with cross-check
            matches = self.bf.match(desc_query, desc_ref)
            if not matches:
                return 0.0
                
            matches = sorted(matches, key=lambda x: x.distance)
            good_matches = [m for m in matches if m.distance < 50]
            num_good_matches = len(good_matches)

            if num_good_matches > 0:
                avg_distance = np.mean([m.distance for m in good_matches])
                # Normalize confidence score
                confidence = num_good_matches / (1 + avg_distance / 20.0)
                confidence = min(confidence / 15.0, 1.0)
                return float(confidence)
            else:
                return 0.0

        except cv2.error:
            return 0.0

    def ssim_score(self, detected: np.ndarray, symbol: np.ndarray) -> float:
        """SSIM similarity score without rotation"""
        try:
            # Ensure same size
            if detected.shape != symbol.shape:
                detected = cv2.resize(detected, symbol.shape[::-1])
            
            # Calculate SSIM
            score, _ = ssim(detected, symbol, full=True)
            return max(0, score)
                    
        except Exception:
            return 0.0
    
    def moments_similarity(self, detected: np.ndarray, symbol_moments: Dict) -> float:
        """Hu moments similarity"""
        try:
            detected_moments = cv2.moments(detected)
            hu1 = cv2.HuMoments(detected_moments).flatten()
            hu2 = cv2.HuMoments(symbol_moments).flatten()
            
            # Handle log transform safely
            hu1 = np.sign(hu1) * np.log10(np.abs(hu1) + 1e-10)
            hu2 = np.sign(hu2) * np.log10(np.abs(hu2) + 1e-10)
            
            # Calculate similarity
            diff = np.abs(hu1 - hu2)
            similarity = 1.0 / (1.0 + np.mean(diff))
            return similarity
            
        except Exception as e:
            return 0.0
    
    def contour_similarity(self, detected: np.ndarray, symbol_contour: Dict) -> float:
        """Contour features similarity"""
        try:
            detected_contour = self.extract_contour_features(detected)
            
            if detected_contour['perimeter'] == 0 or symbol_contour['perimeter'] == 0:
                return 0.0
            
            # Compare different aspects
            area_sim = min(detected_contour['area'], symbol_contour['area']) / max(detected_contour['area'], symbol_contour['area'])
            
            aspect_diff = abs(detected_contour['aspect_ratio'] - symbol_contour['aspect_ratio'])
            aspect_sim = 1.0 / (1.0 + aspect_diff)
            
            circ_diff = abs(detected_contour['circularity'] - symbol_contour['circularity'])
            circ_sim = 1.0 / (1.0 + circ_diff * 5)
            
            return (area_sim + aspect_sim + circ_sim) / 3.0
            
        except Exception as e:
            return 0.0
    
    def calculate_combined_score(self, detected: np.ndarray, symbol_name: str, symbol_data: Dict) -> Tuple[float, Dict]:
        """Combined scoring with template matching rotation only"""
        scores = {}
        
        # 1. Template Matching with rotation (30% weight)
        template_score, template_angle = self.template_matching_score_with_rotation(detected, symbol_data['image'])
        scores['template'] = template_score
        scores['template_angle'] = template_angle
        
        # 2. Feature Matching without rotation (25% weight)
        feature_score = self.feature_matching_score(detected, symbol_data)
        scores['features'] = feature_score
        
        # 3. SSIM without rotation (25% weight)
        ssim_score = self.ssim_score(detected, symbol_data['image'])
        scores['ssim'] = ssim_score
        
        # 4. Moments (10% weight) - no rotation
        scores['moments'] = self.moments_similarity(detected, symbol_data['moments'])
        
        # 5. Contour (10% weight) - no rotation
        scores['contour'] = self.contour_similarity(detected, symbol_data['contour_features'])
        
        # Weighted combination
        weights = {
            'template': 0.60,
            'features': 0.20,
            'ssim': 0.20,
            'moments': 0,
            'contour': 0
        }
        
        final_score = sum(scores[method] * weights[method] for method in scores if method != 'template_angle')
        
        return final_score, scores
    
    def match_symbol(self, detected_region: np.ndarray, top_k=3) -> List[Dict]:
        """Enhanced symbol matching with all techniques"""
        # Preprocess detected region using SAME function as reference symbols
        detected_processed = self.preprocess_detected_region(detected_region)
        
        matches = []
        
        for symbol_name, symbol_data in self.symbols.items():
            # Calculate combined score
            final_score, individual_scores = self.calculate_combined_score(
                detected_processed, symbol_name, symbol_data
            )
            
            # Process scores properly
            processed_scores = {}
            for k, v in individual_scores.items():
                if k == 'template_angle':
                    processed_scores[k] = int(v)  # Convert to int
                else:
                    processed_scores[k] = float(v)  # Convert to float
            
            matches.append({
                'symbol_name': symbol_name,
                'confidence': float(final_score),
                'scores': processed_scores
            })
        
        # Sort by confidence
        matches.sort(key=lambda x: x['confidence'], reverse=True)
        return matches[:top_k]

def enhanced_inference_with_matching(model_path: str, symbol_folder: str, input_folder: str, output_folder: str):
    """Main processing function with enhanced matching"""
    # Import existing inference
    try:
        from inference import create_tiled_inference
        tiled_inference = create_tiled_inference(model_path, 'high_precision')
    except ImportError:
        print("Warning: inference.py not found. Please ensure the inference module is available.")
        return
    
    # Initialize enhanced matcher
    symbol_matcher = EnhancedSymbolMatcher(symbol_folder)
    
    input_folder = Path(input_folder)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    
    for image_path in input_folder.glob("*"):
        if image_path.suffix.lower() not in image_extensions:
            continue
        
        print(f"\n{'='*50}")
        print(f"Processing: {image_path.name}")
        print(f"{'='*50}")
        
        img = cv2.imread(str(image_path))
        if img is None:
            continue
        
        height, width = img.shape[:2]
        print(f"Image size: {width}x{height}")
        
        # Get detections from inference
        detections = tiled_inference.predict_with_multiple_scales(img)
        
        print(f"Found {len(detections)} detections")
        
        # Enhanced matching
        enhanced_detections = []
        for i, det in enumerate(detections):
            print(f"\nMatching detection {i+1}/{len(detections)}")
            
            # Extract detected region
            bbox = det['bbox'].astype(int)
            x1, y1, x2, y2 = np.clip([bbox[0], bbox[1], bbox[2], bbox[3]], 
                                     [0, 0, 0, 0], [width, height, width, height])
            
            if x2 <= x1 or y2 <= y1:
                continue
            
            detected_region = img[y1:y2, x1:x2]
            
            # Enhanced symbol matching
            symbol_matches = symbol_matcher.match_symbol(detected_region, top_k=3)
            
            # Print results for debugging
            if symbol_matches:
                best_match = symbol_matches[0]
                print(f"  Best match: {best_match['symbol_name']} (confidence: {best_match['confidence']:.3f})")
                print(f"  Individual scores: {best_match['scores']}")
            
            enhanced_det = det.copy()
            enhanced_det['symbol_matches'] = symbol_matches
            enhanced_det['best_match'] = symbol_matches[0] if symbol_matches else None
            enhanced_detections.append(enhanced_det)
        
        # Save results
        save_enhanced_results(enhanced_detections, img, image_path, output_folder, width, height)


def save_enhanced_results(detections, img, image_path, output_folder, width, height):
    """Save results with updated label style (based on confidence threshold)"""
    result_img = img.copy()

    # Statistics counters
    high_conf_count = 0
    medium_conf_count = 0
    low_conf_count = 0

    for i, det in enumerate(detections):
        bbox = det['bbox'].astype(int)
        best_match = det['best_match']

        # Determine color based on confidence
        if best_match:
            conf = best_match['confidence']
            if conf > 0.7:
                color = (0, 255, 0)  # Green
                high_conf_count += 1
            elif conf > 0.4:
                color = (0, 255, 255)  # Yellow
                medium_conf_count += 1
            else:
                color = (0, 0, 255)  # Red
                low_conf_count += 1
        else:
            color = (128, 128, 128)  # Gray
            low_conf_count += 1

        # Draw bounding box
        cv2.rectangle(result_img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)

        # Updated Label Style
        if best_match and best_match['confidence'] > 0.3:
            label = f"{best_match['symbol_name']} ({best_match['confidence']:.2f})"
        else:
            label = f"Unknown {det['confidence']:.2f}"

        # Draw label with background
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        cv2.rectangle(result_img, 
                      (bbox[0], bbox[1] - label_size[1] - 10), 
                      (bbox[0] + label_size[0], bbox[1]), 
                      color, -1)
        cv2.putText(result_img, label, 
                    (bbox[0], bbox[1] - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    # Add top-left summary (H:M:L)
    summary = f"H:{high_conf_count} M:{medium_conf_count} L:{low_conf_count}"
    cv2.putText(result_img, summary, (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Save result image
    output_path = output_folder / f"{image_path.stem}_enhanced_matched.jpg"
    cv2.imwrite(str(output_path), result_img)

    # Save JSON result
    json_path = output_folder / f"{image_path.stem}_detailed_results.json"
    results_data = {
        'image_name': image_path.name,
        'image_size': {'width': width, 'height': height},
        'summary': {
            'total_detections': len(detections),
            'high_confidence': high_conf_count,
            'medium_confidence': medium_conf_count,
            'low_confidence': low_conf_count
        },
        'detections': []
    }

    for i, det in enumerate(detections):
        detection_data = {
            'id': i + 1,
            'bbox': det['bbox'].tolist(),
            'yolo_confidence': det['confidence'],
            'best_match': det['best_match'],
            'all_matches': det['symbol_matches']
        }
        results_data['detections'].append(detection_data)

    with open(json_path, 'w') as f:
        json.dump(results_data, f, indent=2)

    # Print save status
    print(f"\nResults saved:")
    print(f"  Image: {output_path}")
    print(f"  JSON: {json_path}")
    print(f"  High confidence: {high_conf_count}")
    print(f"  Medium confidence: {medium_conf_count}")
    print(f"  Low confidence: {low_conf_count}")


# Usage
if __name__ == "__main__":
    enhanced_inference_with_matching(
        model_path=r"save_model\iteration-4 500 epochs red_black ratio_50.pt",
        symbol_folder=r"Row_Legend",
        input_folder=r"input", 
        output_folder=r"matched_output"
    )