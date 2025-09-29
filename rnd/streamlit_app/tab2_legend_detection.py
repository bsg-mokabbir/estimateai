import json
import os
from datetime import datetime
import cv2
import numpy as np
from PIL import Image
import streamlit as st

from rnd.layout_detection.inference.inference_layout_detector import create_tiled_inference
from rnd.layout_detection.inference.matching_symbol import EnhancedSymbolMatcher



class LegendDetector:
    def __init__(self, model_path: str, symbol_folder: str):
        self.model_path = model_path
        self.symbol_folder = symbol_folder

        # Initialize session state
        if 'model_loaded' not in st.session_state:
            with st.spinner('Loading model and symbol matcher...'):
                self.tiled_inference = create_tiled_inference(model_path, 'high_precision')
                self.symbol_matcher = EnhancedSymbolMatcher(symbol_folder)
                
                st.session_state.model_loaded = True
                st.session_state.tiled_inference = self.tiled_inference
                st.session_state.symbol_matcher = self.symbol_matcher
                
                st.success("✅ Model and symbol matcher loaded successfully!")
        else:
            self.tiled_inference = st.session_state.tiled_inference
            self.symbol_matcher = st.session_state.symbol_matcher

    def get_page_images_from_folder(self, page_type):
        """Get page images from the current unique extraction folder and match with PDF data"""
        # Use current extraction folder from session state
        folder_path = getattr(st.session_state, 'current_extraction_folder', 'dynamic_extracted')
        
        if not os.path.exists(folder_path):
            return []
        
        page_images = []
        
        # Get corresponding drawings from PDF data
        if hasattr(st.session_state, 'pdf_drawings') and st.session_state.pdf_drawings:
            pdf_drawings = st.session_state.pdf_drawings[page_type]
        else:
            pdf_drawings = []
        
        for filename in os.listdir(folder_path):
            if filename.endswith('.png') and page_type in filename:
                img_path = os.path.join(folder_path, filename)
                try:
                    img = Image.open(img_path)
                    # Extract drawing number from filename
                    base_name = filename.replace('.png', '')
                    if f'_{page_type}' in base_name:
                        drawing_number = base_name.replace(f'_{page_type}', '')
                    else:
                        drawing_number = base_name
                    
                    # Find matching drawing info from PDF data to get page number
                    page_number = 0
                    for i, drawing_info in enumerate(pdf_drawings):
                        if drawing_info['drawing_number'] == drawing_number:
                            # Page number is i+1 (1-based) from the PDF extraction
                            page_number = i + 1
                            break
                    
                    page_images.append({
                        'image': img,
                        'filename': filename,
                        'drawing_number': drawing_number,
                        'page_number': page_number
                    })
                except Exception as e:
                    st.warning(f"Could not load image {filename}: {e}")
        
        # Sort by page number
        page_images.sort(key=lambda x: x['page_number'])
        return page_images

    def save_detections(self, detections, image_name=None, save_format='json'):
        """Save detections in various formats"""
        save_dir = "detection_saves"
        os.makedirs(save_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"detections_{timestamp}"
        if image_name:
            clean_name = "".join(c for c in image_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
            base_filename = f"detections_{clean_name}_{timestamp}"
        
        saved_files = []
        
        # Convert detections to serializable format
        serializable_detections = []
        for i, det in enumerate(detections):
            det_dict = {
                'detection_id': i,
                'bbox': det['bbox'].tolist() if hasattr(det['bbox'], 'tolist') else det['bbox'],
                'confidence': float(det.get('confidence', 0.0)),
                'class_id': int(det.get('class_id', -1)),
                'timestamp': timestamp
            }
            for key, value in det.items():
                if key not in det_dict:
                    if isinstance(value, np.ndarray):
                        det_dict[key] = value.tolist()
                    else:
                        det_dict[key] = value
            serializable_detections.append(det_dict)
        
        if save_format in ['json', 'all']:
            json_path = os.path.join(save_dir, f"{base_filename}.json")
            with open(json_path, 'w') as f:
                json.dump({
                    'metadata': {
                        'timestamp': timestamp,
                        'image_name': image_name,
                        'total_detections': len(detections),
                        'format_version': '1.0'
                    },
                    'detections': serializable_detections
                }, f, indent=2)
            saved_files.append(json_path)
        
        return saved_files

    def process_image(self, image):
        """Process uploaded image and detect symbols"""
        if image is None:
            return None, "Please upload an image first.", []

        try:
            # Convert PIL image to OpenCV format
            img_array = np.array(image)
            if len(img_array.shape) == 3:
                img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            else:
                img = img_array

            height, width = img.shape[:2]
            
            # Run detection with progress indicator
            progress_container = st.empty()
            with progress_container:
                with st.spinner('🔍 Detecting symbols in image...'):
                    detections = self.tiled_inference.predict_with_multiple_scales(img)
            
            # Auto-save detections
            if detections:
                try:
                    image_name = f"processed_image_{datetime.now().strftime('%H%M%S')}"
                    saved_files = self.save_detections(detections, image_name, save_format='json')
                except Exception as e:
                    st.warning(f"⚠️ Could not save detections: {e}")

            matched_patches = []
            result_img = img.copy()
            high_conf_count, medium_conf_count, low_conf_count = 0, 0, 0
            debug_info = []

            # Process each detection with progress indicator
            with st.spinner(f'🧩 Matching {len(detections)} detected objects with symbol library...'):
                for i, det in enumerate(detections):
                    bbox = det['bbox'].astype(int)
                    x1, y1, x2, y2 = np.clip([bbox[0], bbox[1], bbox[2], bbox[3]],
                                             [0, 0, 0, 0], [width, height, width, height])

                    if x2 <= x1 or y2 <= y1:
                        continue

                    detected_region = img[y1:y2, x1:x2]
                    symbol_matches = self.symbol_matcher.match_symbol(detected_region, top_k=3)
                    
                    if symbol_matches:
                        best_conf = symbol_matches[0]['confidence']
                        debug_info.append(f"Detection {i}: Best confidence {best_conf:.3f}")
                    else:
                        debug_info.append(f"Detection {i}: No matches found")

                    if symbol_matches:
                        best_match = symbol_matches[0]
                        conf = best_match['confidence']

                        if conf > 0.15:
                            if conf > 0.7:
                                color = (0, 255, 0)
                                conf_category = "High"
                                high_conf_count += 1
                            elif conf > 0.3:
                                color = (0, 255, 255)
                                conf_category = "Medium"
                                medium_conf_count += 1
                            else:
                                color = (0, 0, 255)
                                conf_category = "Low"
                                low_conf_count += 1

                            # Draw bounding box and label
                            label = f"{best_match['symbol_name']} ({conf:.2f})"
                            cv2.rectangle(result_img, (x1, y1), (x2, y2), color, 2)
                            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                            cv2.rectangle(result_img, (x1, y1 - label_size[1] - 10), (x1 + label_size[0], y1), color, -1)
                            cv2.putText(result_img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

                            # Store detection info
                            detected_rgb = cv2.cvtColor(detected_region, cv2.COLOR_BGR2RGB)
                            patch_pil = Image.fromarray(detected_rgb)

                            matched_patches.append({
                                'patch': patch_pil,
                                'symbol_name': best_match['symbol_name'],
                                'confidence': conf,
                                'confidence_category': conf_category,
                                'bbox': [x1, y1, x2, y2],
                                'all_matches': symbol_matches,
                                'index': i
                            })

            # Add summary to image
            summary = f"H:{high_conf_count} M:{medium_conf_count} L:{low_conf_count}"
            cv2.putText(result_img, summary, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # Convert result image back to RGB
            result_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
            result_pil = Image.fromarray(result_rgb)

            # Add debug information to summary
            debug_summary = "\n".join(debug_info[:10])
            
            summary_text = f"""Processing Complete!
                Total Raw Detections: {len(detections)}
                Total Matched Symbols: {len(matched_patches)}
                High Confidence (>0.7): {high_conf_count}
                Medium Confidence (0.3-0.7): {medium_conf_count}
                Low Confidence (0.15-0.3): {low_conf_count}
                Image Size: {width}x{height}

                Debug Info (first 10 detections):
                {debug_summary}"""

            return result_pil, summary_text, matched_patches

        except Exception as e:
            error_msg = f"Error processing image: {str(e)}"
            st.error(error_msg)
            return None, error_msg, []

    def display_symbol_selection(self, matched_patches):
        """Create interactive symbol selection with smaller images"""
        if not matched_patches:
            st.info("No symbols detected yet. Upload and process an image first.")
            return []

        st.subheader("🧩 Detected Symbols & Selection")
        st.write("*Select the symbols you want to count in new images*")

        selected_symbols = []
        cols_per_row = 4

        for i in range(0, len(matched_patches), cols_per_row):
            cols = st.columns(cols_per_row)
            for j in range(min(cols_per_row, len(matched_patches) - i)):
                patch_data = matched_patches[i + j]
                name = patch_data['symbol_name']
                patch_img = patch_data['patch']
                key_name = f"symbol_{i+j}_{name}"  # unique & stable key

                with cols[j]:
                    st.image(patch_img, width=60)
                    st.write(f"**{name[:12]}...**" if len(name) > 12 else f"**{name}**")
                    if st.checkbox("Select", key=key_name, help=f"Select {name} for counting"):
                        selected_symbols.append(name)

        # Now update one time
        st.session_state.selected_symbols = selected_symbols
        return selected_symbols

    def run_tab2(self):
        """Tab 2: Legend Detection"""
        st.header("🔍 Legend Symbol Detection")

        # ⬇️ summary
        summary_box = st.container()

        if not st.session_state.get('pdf_processed', False):
            st.warning("⚠️ Please upload and process a PDF first.")
            return

        legend_images = self.get_page_images_from_folder('legend')
        if not legend_images:
            st.warning("No legend page images found. Make sure PDF processing completed successfully.")
            return

        st.success(f"Found {len(legend_images)} legend page images")

        # Legend page selection dropdown
        if len(legend_images) == 1:
            current_legend = legend_images[0]
            st.info(f"**Auto-selected Legend Page:** {current_legend['drawing_number']}")
        else:
            # Create dropdown options
            legend_options = {f"{img['drawing_number']} (Page {img['page_number']})": i 
                            for i, img in enumerate(legend_images)}
            
            selected_option = st.selectbox(
                "📋 Select Legend Page:",
                options=list(legend_options.keys()),
                help="Choose which legend page to use for symbol detection"
            )
            
            selected_index = legend_options[selected_option]
            current_legend = legend_images[selected_index]
            st.info(f"**Selected Legend Page:** {current_legend['drawing_number']}")

        st.subheader("📋 Legend Page")
        st.image(current_legend['image'], caption=f"Legend: {current_legend['drawing_number']}", use_container_width=True)

        if st.button("🚀 Run Legend Detection", type="primary"):
            result_img, summary, patches = self.process_image(current_legend['image'])
            if result_img is not None:
                st.session_state.matched_patches = patches
                st.subheader("🔍 Detection Results")
                st.image(result_img, caption="Detection Results", use_container_width=True)
                st.text_area("📋 Summary", summary, height=150)

        # First checkbox ready
        if st.session_state.get('matched_patches'):
            self.display_symbol_selection(st.session_state.matched_patches)

        with summary_box:
            st.subheader("✅ Selected Symbols Summary")
            st.write("🎯 Currently selected:")
            if st.session_state.get('selected_symbols'):
                for sym in st.session_state.selected_symbols:
                    st.markdown(f"- {sym}")
            else:
                st.markdown("_No symbols selected yet._")
            st.divider()

                    


def main():
    MODEL_PATH = r"save_model/iteration-4_500Epoch_50Black_50Red.pt"
    SYMBOL_FOLDER = r"Row_Legend"
    
    detector = LegendDetector(MODEL_PATH, SYMBOL_FOLDER)
    detector.run_tab2()


if __name__ == "__main__":
    main()