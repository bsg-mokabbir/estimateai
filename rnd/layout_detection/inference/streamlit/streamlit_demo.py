import streamlit as st
import cv2
import numpy as np
from PIL import Image
import base64
import json
import pickle
import os
from datetime import datetime
from io import BytesIO
import random

# Your imports
from inference import TiledInference, create_tiled_inference
from matching_symbol import EnhancedSymbolMatcher


class StreamlitSymbolMatcher:
    def __init__(self, model_path: str, symbol_folder: str):
        self.model_path = model_path
        self.symbol_folder = symbol_folder

        # Initialize session state
        if 'model_loaded' not in st.session_state:
            with st.spinner('Loading model and symbol matcher...'):
                #st.write(f"Loading model from: {model_path}")
                #st.write(f"Loading symbols from: {symbol_folder}")
                
                self.tiled_inference = create_tiled_inference(model_path, 'high_precision')
                self.symbol_matcher = EnhancedSymbolMatcher(symbol_folder)
                
                st.session_state.model_loaded = True
                st.session_state.tiled_inference = self.tiled_inference
                st.session_state.symbol_matcher = self.symbol_matcher
                
                st.success("✅ Model and symbol matcher loaded successfully!")
        else:
            self.tiled_inference = st.session_state.tiled_inference
            self.symbol_matcher = st.session_state.symbol_matcher

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
            #st.success(f"✅ Detections saved to JSON: {json_path}")
        
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
                    #st.info(f"📁 Detections automatically saved: {len(saved_files)} file(s)")
                except Exception as e:
                    st.warning(f"⚠️ Could not save detections: {e}")

            matched_patches = []
            result_img = img.copy()
            high_conf_count, medium_conf_count, low_conf_count = 0, 0, 0
            debug_info = []  # Add debug information

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
                    
                    # Debug: collect information about all matches
                    if symbol_matches:
                        best_conf = symbol_matches[0]['confidence']
                        debug_info.append(f"Detection {i}: Best confidence {best_conf:.3f}")
                    else:
                        debug_info.append(f"Detection {i}: No matches found")

                    if symbol_matches:
                        best_match = symbol_matches[0]
                        conf = best_match['confidence']

                        # Lower the threshold to capture more matches for selection
                        if conf > 0.15:  # Significantly lowered threshold
                            if conf > 0.7:
                                color = (0, 255, 0)
                                conf_category = "High"
                                high_conf_count += 1
                            elif conf > 0.3:  # Lowered medium threshold
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
            debug_summary = "\n".join(debug_info[:10])  # Show first 10 debug entries
            
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
        """Create interactive symbol selection with checkboxes"""
        if not matched_patches:
            st.info("No symbols detected yet. Upload and process an image first.")
            return []

        st.subheader("🧩 Detected Symbols & Selection")
        st.write("*Select the symbols you want to count in new images*")

        selected_symbols = []
        
        # Create selection interface
        for i, patch_data in enumerate(matched_patches):
            col1, col2, col3 = st.columns([1, 3, 1])
            
            name = patch_data['symbol_name']
            conf = patch_data['confidence']
            patch_img = patch_data['patch']
            
            # Confidence styling
            if conf > 0.7:
                conf_text = "🟢 HIGH"
                conf_color = "green"
            elif conf > 0.4:
                conf_text = "🟡 MEDIUM"
                conf_color = "orange"
            else:
                conf_text = "🔴 LOW"
                conf_color = "red"

            with col1:
                # Display thumbnail
                st.image(patch_img, width=80, caption=f"#{i+1}")
            
            with col2:
                # Display symbol info
                st.write(f"**📦 {name}**")
                #st.write(f"Confidence: {conf:.3f}")
                #st.markdown(f"<span style='color: {conf_color}'>{conf_text}</span>", unsafe_allow_html=True)
            
            with col3:
                # Checkbox for selection
                is_selected = st.checkbox(
                    "Select",
                    key=f"symbol_{i}_{name}",
                    help=f"Select {name} for counting"
                )
                
                if is_selected:
                    selected_symbols.append(name)
            
            st.divider()

        return selected_symbols

    def image_to_download_link(self, img_pil, filename="detected_image.png"):
        """Generate download link for PIL image"""
        buffered = BytesIO()
        img_pil.save(buffered, format="PNG", quality=95)
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        href = f'<a href="data:image/png;base64,{img_str}" download="{filename}">' \
            f'<button style="background-color: #4CAF50; color: white; padding: 8px 16px; ' \
            f'border: none; border-radius: 4px; cursor: pointer;">📥 Download Image</button></a>'
        return href


    def count_selected_symbols(self, image, selected_symbols):
        """Count occurrences of selected symbols in new image"""
        if image is None:
            return None, "Please upload an image for counting first.", {}

        if not selected_symbols:
            return None, "Please select at least one symbol to count.", {}

        try:
            # Convert image for processing
            img_array = np.array(image)
            if len(img_array.shape) == 3:
                img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            else:
                img = img_array

            height, width = img.shape[:2]
            
            # Run detection with progress indicator
            with st.spinner(f'🔍 Detecting objects in counting image...'):
                detections = self.tiled_inference.predict_with_multiple_scales(img)
            
            # Initialize counting results
            count_results = {}
            result_img = img.copy()
            
            for symbol_name in selected_symbols:
                count_results[symbol_name] = {
                    'total_count': 0,
                    'high_confidence': 0,
                    'medium_confidence': 0,
                    'low_confidence': 0,
                    'detections': []
                }

            # Process each detection with progress indicator
            with st.spinner(f'🎯 Counting {len(selected_symbols)} selected symbols in {len(detections)} detected objects...'):
                for i, det in enumerate(detections):
                    bbox = det['bbox'].astype(int)
                    x1, y1, x2, y2 = np.clip([bbox[0], bbox[1], bbox[2], bbox[3]],
                                             [0, 0, 0, 0], [width, height, width, height])

                    if x2 <= x1 or y2 <= y1:
                        continue

                    detected_region = img[y1:y2, x1:x2]
                    symbol_matches = self.symbol_matcher.match_symbol(detected_region, top_k=10)

                    symbol_colors = {name: color for name, color in zip(selected_symbols, get_distinct_colors(len(selected_symbols)))}

                    # Check if any match belongs to selected symbols
                    for match in symbol_matches:
                        if match['symbol_name'] in selected_symbols and match['confidence'] > 0.3:
                            symbol_name = match['symbol_name']
                            conf = match['confidence']

                            # Use the color assigned to this symbol
                            color = symbol_colors[symbol_name]
                            
                            # Determine confidence category and color
                            if conf > 0.7:
                                #color = (0, 255, 0)  # Green
                                conf_category = "high_confidence"
                            elif conf > 0.4:
                                #color = (0, 255, 255)  # Yellow
                                conf_category = "medium_confidence"
                            else:
                                #color = (0, 0, 255)  # Red
                                conf_category = "low_confidence"

                            # Update counts
                            count_results[symbol_name]['total_count'] += 1
                            count_results[symbol_name][conf_category] += 1
                            count_results[symbol_name]['detections'].append({
                                'bbox': [x1, y1, x2, y2],
                                'confidence': conf
                            })

                            # Draw on result image
                            cv2.rectangle(result_img, (x1, y1), (x2, y2), color, 2)
                            label = f"{symbol_name} ({conf:.2f})"
                            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                            cv2.rectangle(result_img, (x1, y1 - label_size[1] - 10), (x1 + label_size[0], y1), color, -1)
                            cv2.putText(result_img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                            break  # Only count the best match per detection

            # Add summary to image
            total_found = sum(result['total_count'] for result in count_results.values())
            summary = f"Total Found: {total_found}"
            cv2.putText(result_img, summary, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # Convert result image
            result_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
            result_pil = Image.fromarray(result_rgb)

            # Create summary text
            summary_lines = [f"Counting Results for {len(selected_symbols)} selected symbol(s):"]
            for symbol_name in selected_symbols:
                result = count_results[symbol_name]
                summary_lines.append(f"• {symbol_name}: {result['total_count']} found (H:{result['high_confidence']} M:{result['medium_confidence']} L:{result['low_confidence']})")

            summary_text = "\n".join(summary_lines)

            return result_pil, summary_text, count_results

        except Exception as e:
            error_msg = f"Error during counting: {str(e)}"
            st.error(error_msg)
            return None, error_msg, {}

    def run_app(self):
        """Main Streamlit application"""
        st.set_page_config(
            page_title="Symbol Matching & Counting System",
            page_icon="🔍",
            layout="wide",
            initial_sidebar_state="expanded"
        )

        st.title("🔍 Symbol Matching & Counting System")
        st.write("Upload images to detect symbols, select specific symbols, and count them in new images.")

        # Sidebar for configuration
        with st.sidebar:
            st.header("📋 Instructions")
            st.write("""
            1. **Upload an image** for symbol detection
            2. **Click 'Run Detection'** to find symbols
            3. **Select symbols** you want to count
            4. **Upload a new image** for counting
            5. **Click 'Count Symbols'** to see results
            """)
            
            st.header("ℹ️ About")
            st.write("This tool uses machine learning to detect and count symbols in engineering drawings and technical documents.")

        # Initialize session state
        if 'matched_patches' not in st.session_state:
            st.session_state.matched_patches = []
        if 'selected_symbols' not in st.session_state:
            st.session_state.selected_symbols = []

        # Main interface
        tab1, tab2 = st.tabs(["🔍 Symbol Detection", "🔢 Symbol Counting"])

        with tab1:
            st.header("📸 Upload Image for Detection")
            
            col1, col2 = st.columns(2)
            
            with col1:
                uploaded_file = st.file_uploader(
                    "Choose an image file",
                    type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
                    help="Upload an image containing symbols to detect"
                )
                
                if uploaded_file is not None:
                    image = Image.open(uploaded_file)
                    st.image(image, caption="Uploaded Image", use_container_width=True)
                    
                    if st.button("🚀 Run Legend Detection", type="primary"):
                        # Create progress tracking
                        progress_placeholder = st.empty()
                        status_placeholder = st.empty()
                        
                        with progress_placeholder:
                            progress_bar = st.progress(0)
                            
                        with status_placeholder:
                            st.info("🔍 Starting symbol detection process...")
                            
                        # Update progress
                        progress_bar.progress(10)
                        status_placeholder.info("📷 Processing uploaded image...")
                        
                        progress_bar.progress(30)
                        status_placeholder.info("🤖 Running AI object detection...")
                        
                        result_img, summary, patches = self.process_image(image)
                        
                        progress_bar.progress(90)
                        status_placeholder.info("✨ Finalizing results...")
                        
                        if result_img is not None:
                            st.session_state.matched_patches = patches
                            progress_bar.progress(100)
                            status_placeholder.success(f"✅ Detection complete! Found {len(patches)} matched symbols!")
                            
                            with col2:
                                st.image(result_img, caption="Detection Results", use_container_width=True)
                                st.text_area("📋 Summary", summary, height=200)
                        else:
                            status_placeholder.error("❌ Detection failed. Please try again.")
                        
                        # Clear progress indicators after a short delay
                        import time
                        time.sleep(1)
                        progress_placeholder.empty()
                        status_placeholder.empty()

            # Display symbol selection if patches are available
            if st.session_state.matched_patches:
                st.session_state.selected_symbols = self.display_symbol_selection(st.session_state.matched_patches)
                
                if st.session_state.selected_symbols:
                    st.success(f"✅ Selected {len(st.session_state.selected_symbols)} symbols: {', '.join(st.session_state.selected_symbols)}")

        with tab2:
            st.header("📷 Upload Image for Counting")
            
            if not st.session_state.selected_symbols:
                st.warning("⚠️ Please detect symbols and make selections in the 'Symbol Detection' tab first.")
            else:
                st.info(f"🎯 Ready to count: {', '.join(st.session_state.selected_symbols)}")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    count_file = st.file_uploader(
                        "Choose an image file for counting",
                        type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
                        help="Upload an image where you want to count the selected symbols",
                        key="count_uploader"
                    )
                    
                    if count_file is not None:
                        count_image = Image.open(count_file)
                        st.image(count_image, caption="Image for Counting", use_container_width=True)
                        
                        if st.button("🎯 Count Symbols", type="primary"):
                            # Create progress tracking
                            progress_placeholder = st.empty()
                            status_placeholder = st.empty()
                            
                            with progress_placeholder:
                                progress_bar = st.progress(0)
                                
                            with status_placeholder:
                                st.info("🔢 Starting symbol counting process...")
                                
                            # Update progress
                            progress_bar.progress(15)
                            status_placeholder.info("📷 Processing counting image...")
                            
                            progress_bar.progress(30)
                            status_placeholder.info("🤖 Running object detection...")
                            
                            progress_bar.progress(60)
                            status_placeholder.info(f"🎯 Counting {len(st.session_state.selected_symbols)} selected symbols...")
                            
                            result_img, summary, count_data = self.count_selected_symbols(
                                count_image, st.session_state.selected_symbols
                            )
                            
                            progress_bar.progress(90)
                            status_placeholder.info("📊 Generating results...")
                            
                            if result_img is not None:
                                progress_bar.progress(100)
                                total_found = sum(result['total_count'] for result in count_data.values())
                                status_placeholder.success(f"✅ Counting complete! Found {total_found} total symbols!")
                                
                                with col2:
                                    # Image with download button overlay
                                    st.markdown('<div style="position: relative;">', unsafe_allow_html=True)
                                    st.image(result_img, caption="Counting Results", use_container_width=True)
                                    
                                    # Download button (positioned overlay)
                                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                                    download_link = self.image_to_download_link(result_img, f"counting_results_{timestamp}.png")
                                    st.markdown(download_link, unsafe_allow_html=True)
                                    st.markdown('</div>', unsafe_allow_html=True)
                                    
                                    #st.text_area("📊 Counting Summary", summary, height=200)
                                
                                # Display detailed results
                                if count_data:
                                    st.subheader("📈 Detailed Results")
                                    for symbol_name, results in count_data.items():
                                        with st.expander(f"📦 {symbol_name}"):
                                            col_h, col_m, col_l = st.columns(3)
                                            with col_h:
                                                st.metric("Count: ", results['total_count'])
                                            with col_m:
                                                st.write("")
                                                #st.metric("🟡 Medium Confidence", results['medium_confidence'])
                                            with col_l:
                                                st.write("")
                                                #st.metric("🔴 Low Confidence", results['low_confidence'])
                            else:
                                status_placeholder.error("❌ Counting failed. Please try again.")
                            
                            # Clear progress indicators after a short delay
                            import time
                            time.sleep(1)
                            progress_placeholder.empty()
                            status_placeholder.empty()


def get_distinct_colors(n):
    colors = []
    for i in range(n):
        hue = int(180 * i / n)  # OpenCV uses hue in [0,180]
        color = cv2.cvtColor(np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2BGR)[0][0]
        colors.append(tuple(int(c) for c in color))
    return colors


def main():
    MODEL_PATH = r"/workspace/iteration-4/legent-count-detection/layout_detection/runs/train/experiment_20250808_092107/train/weights/last.pt"
    SYMBOL_FOLDER = r"Row_Legend"

    # Create and run the Streamlit app
    matcher = StreamlitSymbolMatcher(MODEL_PATH, SYMBOL_FOLDER)
    matcher.run_app()


if __name__ == "__main__":
    main()

#how to run it?   
#streamlit run streamlit_demo.py